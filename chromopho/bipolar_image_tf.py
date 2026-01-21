import numpy as np
import tensorflow as tf

from .utils import _parse_cone_string


def _round_to_decimals(x, decimals):
    """
    Round to a fixed number of decimals while preserving gradients via STE.
    Forward matches tf.round; backward passes gradient through unchanged.
    """
    factor = tf.cast(10.0 ** decimals, x.dtype)

    @tf.custom_gradient
    def _round_with_ste(z):
        y = tf.round(z * factor) / factor

        def grad(dy):
            return dy

        return y, grad

    return _round_with_ste(x)


def _gaussian_kernel1d(sigma, truncate=4.0, dtype=tf.float64):
    if sigma <= 0:
        return tf.constant([1.0], dtype=dtype)
    radius = int(truncate * float(sigma) + 0.5)
    x = tf.range(-radius, radius + 1, dtype=dtype)
    kernel = tf.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel = kernel / tf.reduce_sum(kernel)
    return kernel


def _gaussian_blur_tf(x, sigma, truncate=4.0):
    if sigma is None or sigma <= 0:
        return tf.convert_to_tensor(x)
    x = tf.convert_to_tensor(x)
    dtype = x.dtype
    kernel = _gaussian_kernel1d(sigma, truncate=truncate, dtype=dtype)
    k_len = int(kernel.shape[0])
    radius = k_len // 2

    rank = x.shape.rank
    if rank == 2:
        x4 = x[tf.newaxis, :, :, tf.newaxis]
    elif rank == 3:
        x4 = x[tf.newaxis, :, :, :]
    elif rank == 4:
        x4 = x
    else:
        raise ValueError("Input must have rank 2, 3, or 4 for Gaussian blur.")

    channels = tf.shape(x4)[-1]
    k_y = tf.reshape(kernel, [k_len, 1, 1, 1])
    k_y = tf.tile(k_y, [1, 1, channels, 1])
    k_x = tf.reshape(kernel, [1, k_len, 1, 1])
    k_x = tf.tile(k_x, [1, 1, channels, 1])

    # SciPy ndimage reflect matches TF symmetric padding (edge-inclusive).
    padded = tf.pad(x4, [[0, 0], [radius, radius], [0, 0], [0, 0]], mode="SYMMETRIC")
    blurred = tf.nn.depthwise_conv2d(padded, k_y, strides=[1, 1, 1, 1], padding="VALID")
    padded = tf.pad(blurred, [[0, 0], [0, 0], [radius, radius], [0, 0]], mode="SYMMETRIC")
    blurred = tf.nn.depthwise_conv2d(padded, k_x, strides=[1, 1, 1, 1], padding="VALID")

    if rank == 2:
        return tf.squeeze(blurred, axis=[0, 3])
    if rank == 3:
        return tf.squeeze(blurred, axis=0)
    return blurred


def gaussian_blur_reflect_mask_tf(arr, sigma):
    """
    Differentiable masked Gaussian blur.
    Uses normalized convolution to avoid invalid cells contributing.
    Invalid entries remain -1 in the output.
    """
    arr = tf.convert_to_tensor(arr)
    valid = tf.math.is_finite(arr) & (arr >= 0)
    mask = tf.cast(valid, arr.dtype)

    filled = tf.where(valid, arr, tf.zeros_like(arr))
    num = _gaussian_blur_tf(filled, sigma)
    den = _gaussian_blur_tf(mask, sigma)
    blurred = tf.math.divide_no_nan(num, tf.maximum(den, tf.cast(1e-12, arr.dtype)))

    invalid_fill = tf.cast(-1.0, arr.dtype)
    return tf.where(valid, blurred, invalid_fill)


def _srgb_to_linear_tf(x):
    a = tf.cast(0.055, x.dtype)
    return tf.where(x <= tf.cast(0.04045, x.dtype), x / tf.cast(12.92, x.dtype), tf.pow((x + a) / (1 + a), 2.4))


def bipolar_image_filter_tf(
    rgb_image,
    center_cones,
    surround_cones,
    center_sigma=1.0,
    surround_sigma=3.0,
    cone_center_sigma=1,
    alpha_center=1.0,
    alpha_surround=1.0,
    apply_rectification=True,
    on_k=0.7,
    on_n=2.0,
    off_k=0.7,
    off_n=1.5,
    nonlin_adapt_cones=True,
    sigma_adapt=4.0,
    rgb_to_lms=np.array([
        [0.313, 0.639, 0.048],
        [0.155, 0.757, 0.088],
        [0.017, 0.109, 0.874],
    ]),
):
    """
    TensorFlow version of bipolar_image_filter with differentiable ops.
    Returns a grayscale image of bipolar cell response.
    """
    rgb_image = tf.convert_to_tensor(rgb_image)
    rgb_image = tf.cast(rgb_image, tf.float64)

    if rgb_image.shape.rank is None:
        raise ValueError("rgb_image must have known rank.")

    if rgb_image.shape[-1] == 4:
        alpha_mask = tf.equal(rgb_image[..., -1], 0)
        rgb = rgb_image[..., :3]
        rgb = tf.where(alpha_mask[..., tf.newaxis], tf.ones_like(rgb), rgb)
        rgb_image = rgb
    elif rgb_image.shape[-1] > 3:
        rgb_image = rgb_image[..., :3]

    rgb_image = _srgb_to_linear_tf(rgb_image)

    rgb_to_lms_tf = tf.constant(rgb_to_lms, dtype=rgb_image.dtype)
    lms_img = tf.tensordot(rgb_image, rgb_to_lms_tf, axes=[-1, 1])

    L = lms_img[..., 0]
    M = lms_img[..., 1]
    S = lms_img[..., 2]

    def cone_stage_with_adaptation(
        lms,
        lam=np.array([0.2, 0.2, 0.2], dtype=np.float32),
        gamma=np.array([1.2, 1.2, 1.2], dtype=np.float32),
        sigma=np.array([0.25, 0.25, 0.25], dtype=np.float32),
        sigma_adapt=4.0,
    ):
        lam_tf = tf.constant(lam, dtype=lms.dtype)
        gamma_tf = tf.constant(gamma, dtype=lms.dtype)
        sigma_tf = tf.constant(sigma, dtype=lms.dtype)

        lms_nonlin = tf.pow(lms + lam_tf, gamma_tf)
        y0 = tf.pow(lam_tf, gamma_tf)
        y1 = tf.pow(1.0 + lam_tf, gamma_tf)
        endpoint_range = tf.maximum(y1 - y0, tf.cast(1e-12, lms.dtype))
        lms_nonlin = (lms_nonlin - y0) / endpoint_range

        adaptation_signal = _gaussian_blur_tf(lms_nonlin, sigma_adapt)
        gain = tf.reshape(sigma_tf, [1, 1, 3])
        lms_nonlin_adapt = tf.clip_by_value(
            lms_nonlin / (gain + adaptation_signal + tf.cast(1e-12, lms.dtype)),
            0.0,
            1.0,
        )
        return lms_nonlin_adapt

    if nonlin_adapt_cones:
        lms = tf.stack([L, M, S], axis=-1)
        lms = cone_stage_with_adaptation(lms, sigma_adapt=sigma_adapt)
        L = lms[..., 0]
        M = lms[..., 1]
        S = lms[..., 2]

    cL, cM, cS = _parse_cone_string(center_cones)
    sL, sM, sS = _parse_cone_string(surround_cones)

    center_img = tf.stack([cL * L, cM * M, cS * S], axis=-1)
    surround_img = tf.stack([sL * L, sM * M, sS * S], axis=-1)

    keep_center = np.array([cL, cM, cS], dtype=bool)
    keep_surround = np.array([sL, sM, sS], dtype=bool)
    center_indices = np.flatnonzero(keep_center).astype(np.int32)
    surround_indices = np.flatnonzero(keep_surround).astype(np.int32)

    center_img = tf.gather(center_img, center_indices, axis=-1)
    surround_img = tf.gather(surround_img, surround_indices, axis=-1)

    avg_center = tf.reduce_mean(center_img, axis=-1)
    avg_surround = tf.reduce_mean(surround_img, axis=-1)

    center_blur = _gaussian_blur_tf(avg_center, center_sigma)
    surround_blur = _gaussian_blur_tf(avg_surround, surround_sigma)
    center_blur = _round_to_decimals(center_blur, 12)
    surround_blur = _round_to_decimals(surround_blur, 12)

    output = alpha_center * center_blur + alpha_surround * surround_blur

    pol_center = np.sign(np.sum([cL, cM, cS]))
    pol_surround = np.sign(np.sum([sL, sM, sS]))
    abs_min = min(alpha_center * pol_center, alpha_surround * pol_surround)
    abs_max = max(alpha_center * pol_center, alpha_surround * pol_surround)

    output_normalized = tf.clip_by_value(
        (output - abs_min) / (abs_max - abs_min),
        0.0,
        1.0,
    )

    if not apply_rectification:
        return output_normalized

    def _nr_base(x, k, n):
        x = tf.clip_by_value(x, 0.0, 1.0)
        k = max(float(k), 1e-12)
        n = float(n)
        xn = tf.pow(x, n)
        kn = k**n
        return xn / (xn + kn + 1e-12)

    def on_rectifier_nr(x, k, n):
        s = 1.0 + k**n
        return tf.clip_by_value(s * _nr_base(x, k, n), 0.0, 1.0)

    def off_rectifier_nr_high(x, k, n):
        s = 1.0 + k**n
        return tf.clip_by_value(1.0 - s * _nr_base(1.0 - x, k, n), 0.0, 1.0)

    if pol_center > 0:
        output_rectified = on_rectifier_nr(output_normalized, on_k, on_n)
    else:
        output_rectified = off_rectifier_nr_high(output_normalized, off_k, off_n)

    return output_rectified


class BipolarImageProcessorTF:
    """
    TensorFlow-based, differentiable, vectorized version of BipolarImageProcessor.
    """

    def __init__(
        self,
        mosaic,
        image,
        fit_option="fit_mosaic",
        return_minimum_rf=False,
        method="greyscale",
        stimulation_mosaic=None,
        save_flat=True,
        amacrine_sigma_blur=None,
        build_receptive_fields=True,
    ):
        self.mosaic = mosaic
        self.image_tf = tf.convert_to_tensor(image)
        self.image = self.image_tf
        if self.image_tf.shape.rank is None or self.image_tf.shape[0] is None or self.image_tf.shape[1] is None:
            shape = tf.shape(self.image_tf)
            self.image_shape = (int(shape[0].numpy()), int(shape[1].numpy()))
        else:
            self.image_shape = (int(self.image_tf.shape[0]), int(self.image_tf.shape[1]))
        self.stimulation_mosaic = stimulation_mosaic
        self.fit_option = fit_option
        self.bipolar_images = {}
        self.amacrine_sigma_blur = amacrine_sigma_blur
        self.build_receptive_fields = build_receptive_fields

        self._fit_image_and_mosaic(return_minimum_rf)
        self.get_all_average_colors(method=method, save_flat=save_flat, blur_sigma=self.amacrine_sigma_blur)
        if save_flat is False:
            self.avg_subtype_response_per_pixel = {}
            self.get_avg_color_map_per_pixel()

    def _fit_image_and_mosaic(self, return_minimum=False):
        img_height, img_width = self.image_shape
        mosaic_height, mosaic_width = self.mosaic.grid.shape[:2]
        if img_height < mosaic_height or img_width < mosaic_width:
            raise ValueError("Image is too small to fit the mosaic")

        if self.fit_option == "fit_mosaic":
            square_dim = min(img_height // mosaic_height, img_width // mosaic_width)
            self._minimum_overlap_square_dim = square_dim
            img_cutout_dim = (square_dim * mosaic_height, square_dim * mosaic_width)
            img_centerpt = (img_height // 2, img_width // 2)

            if (
                (img_cutout_dim[0] % 2 != 0)
                and (img_cutout_dim[1] % 2 != 0)
                and (mosaic_height % 2 != 0)
                and (mosaic_width % 2 != 0)
            ):
                img_cutout_i_range = [
                    int(img_centerpt[0] - img_cutout_dim[0] // 2),
                    int(img_centerpt[0] + img_cutout_dim[0] // 2 - 1),
                ]
                img_cutout_j_range = [
                    int(img_centerpt[1] - img_cutout_dim[1] // 2),
                    int(img_centerpt[1] + img_cutout_dim[1] // 2 - 1),
                ]
            else:
                img_cutout_i_range = [
                    int((img_centerpt[0] - img_cutout_dim[0] / 2) - 1),
                    int(img_centerpt[0] + img_cutout_dim[0] // 2 - 1),
                ]
                img_cutout_j_range = [
                    int((img_centerpt[1] - img_cutout_dim[1] / 2) - 1),
                    int(img_centerpt[1] + img_cutout_dim[1] // 2 - 1),
                ]

            start_rows = img_cutout_i_range[0] + np.arange(mosaic_height) * square_dim
            start_cols = img_cutout_j_range[0] + np.arange(mosaic_width) * square_dim
            end_rows = np.minimum(start_rows + square_dim, img_height)
            end_cols = np.minimum(start_cols + square_dim, img_width)

            row_min = np.maximum(start_rows, 0)
            row_max = np.minimum(end_rows - 1, img_height - 1)
            col_min = np.maximum(start_cols, 0)
            col_max = np.minimum(end_cols - 1, img_width - 1)

            center_rows = ((row_min + row_max) / 2.0).astype(int)
            center_cols = ((col_min + col_max) / 2.0).astype(int)

            row_grid = np.repeat(center_rows[:, np.newaxis], mosaic_width, axis=1)
            col_grid = np.repeat(center_cols[np.newaxis, :], mosaic_height, axis=0)

            valid_mask = self.mosaic.grid != -1
            valid_coords = np.argwhere(valid_mask)
            center_coords = np.stack(
                [row_grid[valid_mask], col_grid[valid_mask]],
                axis=1,
            )

            self._valid_cell_coords = valid_coords.astype(int)
            self._center_coords = center_coords.astype(int)
            self._cell_centers = {
                (int(coord[0]), int(coord[1])): (int(center[0]), int(center[1]))
                for coord, center in zip(self._valid_cell_coords, self._center_coords)
            }

            if self.build_receptive_fields:
                mapping = {}
                for i in range(mosaic_height):
                    for j in range(mosaic_width):
                        if self.mosaic.grid[i, j] == -1:
                            continue

                        start_row = img_cutout_i_range[0] + i * square_dim
                        end_row = min(start_row + square_dim, img_height)
                        start_col = img_cutout_j_range[0] + j * square_dim
                        end_col = min(start_col + square_dim, img_width)

                        rec_field = [
                            (r, c)
                            for r in range(start_row, end_row)
                            for c in range(start_col, end_col)
                            if 0 <= r < img_height and 0 <= c < img_width
                        ]

                        if not return_minimum:
                            rec_field = self._square_to_circle_pixels(rec_field, i, j)

                        mapping[(i, j)] = rec_field
                self._receptive_field_map = mapping
            else:
                self._receptive_field_map = {}
        elif self.fit_option == "fit_image":
            raise ValueError("functionality for fit_image not yet implemented :.(")
        else:
            raise ValueError(f"Unknown fit_option: {self.fit_option}")

    def _square_to_circle_pixels(self, square_pixels, i, j):
        scale = self.mosaic.get_receptive_field_size(i, j)
        img_height, img_width = self.image_shape
        if self._minimum_overlap_square_dim in [1, 2, 3, 4, 5, 6, 7]:
            scale_min = {1: 10, 2: 2.5, 3: 1.5, 4: 1.3, 5: 1.2, 6: 1.1, 7: 1.1}
            scale = max(scale, scale_min[self._minimum_overlap_square_dim])

        square_pixels = np.array(square_pixels)
        rows, cols = square_pixels[:, 0], square_pixels[:, 1]

        row_min, row_max = rows.min(), rows.max()
        col_min, col_max = cols.min(), cols.max()
        center_row = (row_min + row_max) / 2.0
        center_col = (col_min + col_max) / 2.0

        if not hasattr(self, "_cell_centers"):
            self._cell_centers = {}
        self._cell_centers[(i, j)] = (int(float(center_row)), int(float(center_col)))

        height = row_max - row_min
        width = col_max - col_min
        half_diag = np.sqrt(height**2 + width**2) / 2.0
        if half_diag == 0:
            half_diag = 0.1
        radius = half_diag * scale

        circle_pixels = []
        row_start = max(int(center_row - radius - 1), 0)
        row_end = min(int(center_row + radius + 1), img_height)
        col_start = max(int(center_col - radius - 1), 0)
        col_end = min(int(center_col + radius + 1), img_width)
        for r in range(row_start, row_end + 1):
            if r <= img_height and r > 0:
                for c in range(col_start, col_end + 1):
                    dist_sq = (r - center_row) ** 2 + (c - center_col) ** 2
                    if dist_sq <= radius**2:
                        circle_pixels.append((r, c))

        return circle_pixels

    def get_receptive_field_of_cell(self, i, j):
        return self._receptive_field_map[(i, j)]

    def _compute_subtype_image(
        self,
        subtype,
        method="grayscale",
        rgb_to_lms=np.array([
            [0.313, 0.639, 0.048],
            [0.155, 0.757, 0.088],
            [0.017, 0.109, 0.874],
        ]),
    ):
        if subtype.name in self.bipolar_images:
            return self.bipolar_images[subtype.name]

        color_filter_dict = subtype.color_filter_params
        rf_params = subtype.rf_params

        bipolar_image_seen = bipolar_image_filter_tf(
            rgb_image=self.image_tf,
            center_cones=color_filter_dict["center"],
            surround_cones=color_filter_dict["surround"],
            center_sigma=rf_params["center_sigma"],
            surround_sigma=rf_params["surround_sigma"],
            alpha_center=rf_params["alpha_center"],
            alpha_surround=rf_params["alpha_surround"],
            apply_rectification=rf_params["apply_rectification"],
            on_k=rf_params["on_k"],
            on_n=rf_params["on_n"],
            off_n=rf_params["off_n"],
            off_k=rf_params["off_k"],
            rgb_to_lms=rgb_to_lms,
        )

        self.bipolar_images[subtype.name] = bipolar_image_seen
        return bipolar_image_seen

    def compute_cell_output(self, i, j, method="grayscale"):
        subtype = self.mosaic._index_to_subtype_dict[self.mosaic.grid[i, j]]
        bipolar_image_seen = self._compute_subtype_image(subtype, method=method)
        center_pos = self._cell_centers[(i, j)]

        center_idx = tf.constant([center_pos[0], center_pos[1]], dtype=tf.int32)
        return tf.gather_nd(bipolar_image_seen, center_idx[tf.newaxis, :])[0]

    def _build_subtype_stack(self, method="grayscale"):
        images = []
        for subtype in self.mosaic.subtypes:
            images.append(self._compute_subtype_image(subtype, method=method))
        return tf.stack(images, axis=0)

    def get_all_average_colors(self, method="grayscale", save_flat=False, blur_sigma=None):
        stimulation_mosaic = self.stimulation_mosaic

        valid_coords = self._valid_cell_coords
        valid_coords_tf = tf.convert_to_tensor(valid_coords, dtype=tf.int32)

        if stimulation_mosaic is not None:
            stim = tf.convert_to_tensor(stimulation_mosaic, dtype=tf.float64)
            cell_outputs = tf.gather_nd(stim, valid_coords_tf)
        else:
            subtype_stack = self._build_subtype_stack(method=method)

            subtype_index_to_stack = {
                self.mosaic.subtype_index_dict[subtype.name]: idx
                for idx, subtype in enumerate(self.mosaic.subtypes)
            }
            cell_subtype_indices = self.mosaic.grid[valid_coords[:, 0], valid_coords[:, 1]]
            stack_indices = np.array(
                [subtype_index_to_stack[idx] for idx in cell_subtype_indices],
                dtype=np.int32,
            )

            indices = tf.stack(
                [
                    tf.convert_to_tensor(stack_indices, dtype=tf.int32),
                    tf.convert_to_tensor(self._center_coords[:, 0], dtype=tf.int32),
                    tf.convert_to_tensor(self._center_coords[:, 1], dtype=tf.int32),
                ],
                axis=1,
            )
            cell_outputs = tf.gather_nd(subtype_stack, indices)

        self.cell_outputs = cell_outputs
        self.avg_colors_cell_map = None

        h, w = self.mosaic.grid.shape
        valid_mask = tf.convert_to_tensor(self.mosaic.grid != -1)
        base_grid = tf.scatter_nd(valid_coords_tf, cell_outputs, [h, w])
        base_grid = tf.where(valid_mask, base_grid, tf.cast(-1.0, base_grid.dtype))
        self.avg_colors_cell_grid = base_grid

        if save_flat:
            if method == "lms":
                print("unable to save flat lms output grid, set method to grayscale")
                return
            grid_outputs = base_grid

            if blur_sigma is not None:
                grid_outputs = gaussian_blur_reflect_mask_tf(grid_outputs, sigma=blur_sigma)

            self.grid_outputs = grid_outputs

    def get_avg_colors_cell_map_numpy(self):
        if self.avg_colors_cell_map is not None:
            return self.avg_colors_cell_map
        if not hasattr(self, "cell_outputs"):
            raise ValueError("cell_outputs not computed; call get_all_average_colors first.")
        avg_colors_cell_map = {}
        valid_coords = self._valid_cell_coords
        cell_outputs_np = self.cell_outputs.numpy()
        for coord, val in zip(valid_coords, cell_outputs_np):
            avg_colors_cell_map[(int(coord[0]), int(coord[1]))] = val
        self.avg_colors_cell_map = avg_colors_cell_map
        return avg_colors_cell_map

    def get_avg_color_map_per_pixel(self):
        if not self._receptive_field_map:
            raise ValueError("Receptive field map not built. Set build_receptive_fields=True.")

        avg_color_map = self.get_avg_colors_cell_map_numpy()
        for subtype in self.mosaic.subtypes:
            subtype_index = self.mosaic.subtype_index_dict[subtype.name]
            rec_fields = {
                cell: pixels
                for cell, pixels in self._receptive_field_map.items()
                if self.mosaic.grid[cell] == subtype_index
            }

            pixel_to_avg_colors = {}

            for cell, pixels in rec_fields.items():
                avg_color = avg_color_map[cell]
                for pixel in pixels:
                    if pixel[0] < self.image_shape[0] and pixel[1] < self.image_shape[1]:
                        pixel_to_avg_colors.setdefault(pixel, []).append(avg_color)

            pixel_to_final_avg = {
                pixel: np.mean(colors, axis=0) for pixel, colors in pixel_to_avg_colors.items()
            }

            self.avg_subtype_response_per_pixel[subtype.name] = pixel_to_final_avg
