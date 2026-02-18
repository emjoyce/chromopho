import numpy as np
import tensorflow as tf
from .utils import img_to_rgb, _parse_cone_string, gaussian_blur_reflect_mask
from .plot import bipolar_image_filter
import concurrent.futures
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor


# tf based functions
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

    # Cap the effective radius so that symmetric padding never exceeds
    # the spatial extent of the current image. 
    height = x4.shape[1]
    width = x4.shape[2]
    if height is not None and width is not None:
        max_radius = min(int(height) - 1, int(width) - 1)
        if radius > max_radius:
            # Slice the 1D kernel around its center to the maximal
            # supported radius and renormalize.
            center = int(kernel.shape[0]) // 2
            radius = max_radius
            k_len = 2 * radius + 1
            start = center - radius
            end = center + radius + 1
            kernel = kernel[start:end]
            kernel = kernel / tf.reduce_sum(kernel)

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


def ste_clip(x, min, max): 
    y = tf.clip_by_value(x, min, max) 
    return x + tf.stop_gradient(y - x)


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
        lms_nonlin_adapt = ste_clip(
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
    # center_blur = _round_to_decimals(center_blur, 12)
    # surround_blur = _round_to_decimals(surround_blur, 12)

    output = alpha_center * center_blur + alpha_surround * surround_blur

    pol_center = np.sign(np.sum([cL, cM, cS]))
    pol_surround = np.sign(np.sum([sL, sM, sS]))
    abs_min = min(alpha_center * pol_center, alpha_surround * pol_surround)
    abs_max = max(alpha_center * pol_center, alpha_surround * pol_surround)

    output_normalized = ste_clip(
        (output - abs_min) / (abs_max - abs_min),
        0.0,
        1.0,
    )

    if not apply_rectification:
        return output_normalized

    def _nr_base(x, k, n):
        x = ste_clip(x, 0.0, 1.0)
        k = max(float(k), 1e-12)
        n = float(n)
        xn = tf.pow(x, n)
        kn = k**n
        return xn / (xn + kn + 1e-12)

    def on_rectifier_nr(x, k, n):
        s = 1.0 + k**n
        return ste_clip(s * _nr_base(x, k, n), 0.0, 1.0)

    def off_rectifier_nr_high(x, k, n):
        s = 1.0 + k**n
        return ste_clip(1.0 - s * _nr_base(1.0 - x, k, n), 0.0, 1.0)

    if pol_center > 0:
        output_rectified = on_rectifier_nr(output_normalized, on_k, on_n)
    else:
        output_rectified = off_rectifier_nr_high(output_normalized, off_k, off_n)

    return output_rectified








class BipolarImageProcessorTF:
    """
    TensorFlow-based, differentiable, vectorized version of BipolarImageProcessor.
    Takes a bipolar mosaic and an images and processes the image through the mosaic using the 
    color filter parameter in each subtype. the image is assumed to cover the mosaic with fit_option 
    dictating if the image should be fit to the mosaic or if the entire image should be seen by the mosaic. 
    """

    def __init__(self, mosaic, image, fit_option = 'fit_mosaic', return_minimum_rf = False, method = 'greyscale', stimulation_mosaic = None, 
    amacrine_sigma_blur=None):
        """
        Parameters:
        mosaic (BipolarMosaic): a BipolarMosaic object
        image (array): pulse2percept image object
        fit_option (str): how to fit the image to the mosaic, either 'fit_image' to see the whole image but some cells may see no pixels 
            or 'fit_mosaic' so that each cell in the mosaic sees some pixels,but some pixels may not be seen by any cell 
        method (str): the method to use to compute the average color of the pixels in the receptive field of a cell. avg color could be rgb 
            (lifo coming in from cones) or grayscale (info going out from bipolar cells)
        stimulation_mosaic (array): an array same size as mosaic that holds which cells are stimulated by an electrode and to what intensity level
        amacrine_sigma_blur (float or None): optional sigma for masked gaussian blur applied to the flattened per-cell outputs
        """
        self.mosaic = mosaic
        self.image = image
        self.image_tf = tf.convert_to_tensor(image)
        self.stimulation_mosaic = stimulation_mosaic
        self.fit_option = fit_option
        self.bipolar_images = {}
        # store requested amacrine blur for later use
        self.amacrine_sigma_blur = amacrine_sigma_blur

        # flag that controls whether full receptive fields are stored (vs just centers)
        # in this tf version we default to True so downstream code can use the mapping
        self.build_receptive_fields = True

        self._fit_image_and_mosaic(return_minimum_rf)
        self.get_all_average_colors(method = method, blur_sigma=self.amacrine_sigma_blur)

        self.avg_subtype_response_per_pixel = {}
        self.get_avg_color_map_per_pixel()

        # make the receptive field map 


    def process_new_image(self, image, method='grayscale', 
                          stimulation_mosaic=None, amacrine_sigma_blur=None,
                          recompute_pixel_map=False):
        """Reuse an existing receptive field map with a new image.

        This keeps the existing mosaic-to-pixel mapping and just recomputes
        bipolar responses for a new image, as long as the image height/width
        matches the original image used to build this processor.

        Parameters
        ----------
        image : np.ndarray
            New RGB image array. Must have the same spatial dimensions as
            the original image used when constructing this instance.
        method : str, optional
            Passed through to color / response computation (e.g. 'grayscale').
        stimulation_mosaic : np.ndarray or None, optional
            Optional stimulation mosaic to override normal color processing.
        amacrine_sigma_blur : float or None, optional
            If provided, overrides ``self.amacrine_sigma_blur`` for this
            recomputation; otherwise the stored value is reused.
        recompute_pixel_map : bool, optional
            If True, recompute ``avg_subtype_response_per_pixel`` using the
            existing receptive field map but the new image responses. This is
            required before calling higher-level feature extraction that uses
            per-pixel bipolar responses.
        """
        # ensure spatial dimensions match the mapping this processor was built on
        old_h, old_w = self.image.shape[:2]
        new_h, new_w = image.shape[:2]
        if (old_h, old_w) != (new_h, new_w):
            raise ValueError(
                f"New image shape {image.shape[:2]} does not match original {self.image.shape[:2]}, new mapping failed"
            )
            return

        # update state for the new image
        self.image = image
        self.image_tf = tf.convert_to_tensor(image)
        self.stimulation_mosaic = stimulation_mosaic

        # clear cached per-subtype filtered images so they are recomputed
        self.bipolar_images = {}

        # choose blur sigma for this pass
        blur_sigma = self.amacrine_sigma_blur if amacrine_sigma_blur is None else amacrine_sigma_blur

        # recompute per-cell outputs and optional flattened grid
        self.get_all_average_colors(method=method, blur_sigma=blur_sigma)

        # optionally recompute per-pixel subtype average response maps (default is False)
        if recompute_pixel_map:
            self.avg_subtype_response_per_pixel = {}
            self.get_avg_color_map_per_pixel()


    def _fit_image_and_mosaic(self, return_minimum = False):
        """
        Fits the image and  mosaic in accordance with fit_option 
        returns mapping which has the i,j indices of the mosaic as keys and the pixels in the receptive field of the cell as values
        Parameters:
        return_minimum (bool): if True, returns the minimum square receptive field without making it a circle
        """
        # we need to ensure that the image is the same size or larger than the mosaic, as each cell cannot have less than one pixel in its receptive field
        img_height, img_width = self.image.shape[:2]
        mosaic_height, mosaic_width = self.mosaic.grid.shape[:2]
        if img_height < mosaic_height or img_width < mosaic_width:
            raise ValueError('Image is too small to fit the mosaic')
        
        if self.fit_option == 'fit_mosaic':
            # this will fit the mosaic to the image, so that the entire mosaic is seeing pixels, but maybe not all pixels will be seen by a cell



            # first calculate the nonoverlapping squares that would fit in here
            square_dim = min(img_height // mosaic_height, img_width // mosaic_width)
            # if square_dim < 1:
            #     square_dim = 1
            self._minimum_overlap_square_dim = square_dim
            # get the dimensions of the mosaic in the space of the image
            img_cutout_dim = square_dim * mosaic_height, square_dim * mosaic_width

            # get center point of image 
            img_centerpt = img_height//2, img_width//2
            # get cutout of image that mosaic cells will 'see'
            # if all dims odd, center is whole number ind, easy to get cutout
            if ((img_cutout_dim[0] % 2 != 0) and (img_cutout_dim[1] % 2 != 0)) and ((mosaic_height % 2 != 0) and (mosaic_width % 2 != 0)):
                img_cutout_i_range = [int(img_centerpt[0]-img_cutout_dim[0]//2), int(img_centerpt[0]+img_cutout_dim[0]//2-1)]
                img_cutout_j_range = [int(img_centerpt[1]-img_cutout_dim[1]//2), int(img_centerpt[1]+img_cutout_dim[1]//2-1)]
                
            # if one of the dims is even, then define cutout like so
            else:  
                img_cutout_i_range = [int((img_centerpt[0]-img_cutout_dim[0]/2)-1), int(img_centerpt[0]+img_cutout_dim[0]//2-1)]
                img_cutout_j_range = [int((img_centerpt[1]-img_cutout_dim[1]/2)-1), int(img_centerpt[1]+img_cutout_dim[1]//2-1)]
            # now assign squsare of pixels as receptive field of each cell in the mosaic
            mapping = {}     

            # Loop through the mosaic grid
            for i in range(mosaic_height):
                for j in range(mosaic_width):
                    # Skip if no cell at this point in mosaic
                    if self.mosaic.grid[i, j] == -1:
                        continue

                    # Calculate pixel 'block' boundaries
                    start_row = img_cutout_i_range[0] + i * square_dim
                    end_row = min(start_row + square_dim, img_height)
                    start_col = img_cutout_j_range[0] + j * square_dim
                    end_col = min(start_col + square_dim, img_width)
                    
                    # Build receptive field pixel indices
                    rec_field = [
                        (r, c)
                        for r in range(start_row, end_row)
                        for c in range(start_col, end_col)
                        if 0 <= r < img_height and 0 <= c < img_width
                    ]

                    # Optionally circleify the receptive field
                    if not return_minimum:
                        rec_field = self._square_to_circle_pixels(rec_field, i, j)

                    # Save the mapping
                    mapping[(i, j)] = rec_field


        # else: # TODO: do this ^ but for fit_image
        elif self.fit_option == 'fit_image':
            raise ValueError('functionality for fit_image not yet implemented :.(')

        # save the receptive field mapping (may be empty if build_receptive_fields is False)
        self._receptive_field_map = mapping

        # after mapping/centers are defined, cache useful coordinate arrays for later tf ops
        # valid cell coords in mosaic grid
        valid_mask = (self.mosaic.grid != -1)
        self._valid_cell_coords = np.argwhere(valid_mask).astype(int)

        # corresponding center coords (row, col in image space) for each valid cell
        # use the _cell_centers dict that was filled inside _square_to_circle_pixels
        center_rows = []
        center_cols = []
        for (i, j) in self._valid_cell_coords:
            cy, cx = self._cell_centers[(int(i), int(j))]
            center_rows.append(int(cy))
            center_cols.append(int(cx))
        self._center_coords = np.stack([np.array(center_rows), np.array(center_cols)], axis = 1).astype(int)










        #     # TODO: using a list is slow, refactor 
        #     available_pixel_inds = [(i,j) for i in range(img_cutout_i_range[0], img_cutout_i_range[1]) for j in range(img_cutout_j_range[0], img_cutout_j_range[1])]
        #     # chop these available_pixel_inds 
        #     for i in range(mosaic_height):
        #         for j in range(mosaic_width):
        #             # if this i,j location does not have a cell in the mosaic, skip
        #             # the square of pixels will be square_dim x square_dim per i,j element in the mosaic
        #             # as we iterate through the row via i, we need to keep track of the pixels that have already been assigned via available_pixel_inds
        #             # get the first dim pixels in available_pixel_inds
        #             # first get j indices of the first square_dim pixels 
        #             js = np.array(available_pixel_inds[:square_dim])[:,1]
        #             j_inds = [j for j in js if j >= 0 and j < img_width]
        #             # iterate through the first square_dim i indices and assign the i,j pairs and remove them from available_pixel_inds
        #             # first i ind starts at first i in available_pixel_inds
        #             first_i = available_pixel_inds[0][0]
        #             i_inds = list(range(first_i, min(first_i+square_dim, img_height)))
        #             rec_field = [(ii,jj) for ii in i_inds for jj in j_inds]

        #             # now get the diagonal 'radius' of cube 
                    
        #             # remove the pixels that have been assigned
        #             [available_pixel_inds.remove(rec_field[i]) for i in range(len(rec_field))]
        #             # if there is not a cell in the mosaic here, dont add anythign to the mapping, but still needed to remove first 
        #             # TODO: is this is also innefficient to do this after all this computation^ 
        #             if self.mosaic.grid[i,j] == -1:
        #                 continue # without adding anything to the mapping
        #             # now have to 'circleify' the square of pixels
        #             if not return_minimum:
        #                 rec_field = self._square_to_circle_pixels(rec_field, i, j)
                    
        #             mapping[(i, j)] = rec_field



        # else: # TODO: do this ^ but for fit_image
        #     raise ValueError('functionality for fit_image not yet implemented :.(')
        # self._receptive_field_map = mapping

    def _square_to_circle_pixels(self, square_pixels, i, j):
        """
        Given a list of (row, col) pixel indices that form a square,
        produce a new list of (row, col) pixel indices that form 
        a circumscribed circle (or circle-like region).
        
        Parameters:
        square_pixels (list): a list of (row, col) pixel indices that form a square
        i (int): the row index of the cell in the mosaic
        j (int): the column index of the cell in the mosaic
        """
        # pick the scaling factor from mosaic 
        scale = self.mosaic.get_receptive_field_size(i, j)
        img_height, img_width = self.image.shape[:2]
        if self._minimum_overlap_square_dim in [1, 2, 3, 4, 5, 6, 7]:
            # these need to have scale manually scaled up to a minimum value or else turning into circle will have no effect
            # and we want overlapping to result from this circle-fication!
            scale_min = {1:10, 2:2.5, 3:1.5, 4:1.3, 5:1.2, 6:1.1, 7:1.1} 
            scale = max(scale, scale_min[self._minimum_overlap_square_dim])

        square_pixels = np.array(square_pixels)
        rows, cols = square_pixels[:, 0], square_pixels[:, 1]
        
        
        # get square's bounding box
        row_min, row_max = rows.min(), rows.max()
        col_min, col_max = cols.min(), cols.max()
        center_row = (row_min + row_max) / 2.0
        center_col = (col_min + col_max) / 2.0

        # save the center pos 
        if not hasattr(self, "_cell_centers"):
            self._cell_centers = {}
        self._cell_centers[(i, j)] = (int(float(center_row)), int(float(center_col)))

        # find the circle’s center and radius via diagonal of square
        height = row_max - row_min
        width  = col_max - col_min
        half_diag = np.sqrt(height**2 + width**2) / 2.0
        if half_diag == 0: # this is for the case where the square is just one pixel
            half_diag = .1
        radius = half_diag * scale  


        # add the pixels
        circle_pixels = []
        row_start = max(int(center_row - radius - 1), 0)
        row_end   = min(int(center_row + radius + 1), img_height)
        col_start = max(int(center_col - radius - 1), 0)
        col_end   = min(int(center_col + radius + 1), img_width)
        for r in range(row_start, row_end + 1):
            if r <= self.image.shape[0] and r > 0:
                for c in range(col_start, col_end + 1):
                    dist_sq = (r - center_row)**2 + (c - center_col)**2
                    if dist_sq <= radius**2:
                        circle_pixels.append((r, c))

        return circle_pixels

    # build several functions that together will compute the information held in every subtype, in every cell in the mosaic
    # this will take all the pixels in the receptive field of a cell and use the color filter of each subtype 
    # break that up: one function that will return the pixels of a receptive field of a cell
    # one function that will create the image seen by every subtype by applying the color filter 
    # one function that will take all the pixels seen by a cell and get the average color of those pixels 

    def get_receptive_field_of_cell(self, i, j):
        """
        Returns the pixels in the receptive field of the cell at i,j location of cell mosaic 
        """
        return self._receptive_field_map[(i, j)]
    
    def _compute_subtype_image(self, subtype, method = 'grayscale', rgb_to_lms = np.array([[0.313, 0.639, 0.048],  
                            [0.155, 0.757, 0.088], [0.017, 0.109, 0.874]])):
        '''
        computes the ideal image seen by the given subtype
        '''

        # check if we have already computed this image
        if subtype.name in self.bipolar_images:
            bipolar_image_seen = self.bipolar_images[subtype.name]
        # get the color filter params
        else:
            color_filter_dict = subtype.color_filter_params
            # get the other rf params
            rf_params = subtype.rf_params
            # generate the image seen by the subtype
            # computes more of the cone info coming in 
            # s-on would compute 
            bipolar_image_seen = bipolar_image_filter_tf(
                        rgb_image = self.image_tf,
                        center_cones = color_filter_dict['center'],
                        surround_cones = color_filter_dict['surround'],
                        center_sigma = rf_params['center_sigma'],
                        surround_sigma = rf_params['surround_sigma'],
                        alpha_center = rf_params['alpha_center'],
                        alpha_surround = rf_params['alpha_surround'],
                        apply_rectification=rf_params['apply_rectification'], 
                        on_k=rf_params['on_k'], 
                        on_n=rf_params['on_n'], 
                        off_n=rf_params['off_n'],
                        off_k=rf_params['off_k'],
                        rgb_to_lms = rgb_to_lms,)
                # so the image should output a single value dependign on subtype, which if s+, would be the output of l cones minue the output of m+l

            self.bipolar_images[subtype.name] = bipolar_image_seen
        return bipolar_image_seen
    
    def compute_cell_output(self, i, j, method = 'grayscale'):
        """
        Computes the output of the cell at i,j by picking the output of the center pixel of its receptive field after DoG
        Parameters:
        i (int): row index of cell in mosaic
        j (int): column index of cell in mosaic
        method (str): method to use to compute the average color of the pixels in the receptive (grayscale for 0 to 1 graded output)

        """

        # get the subtype of the cell so we can pull the correct DoG image
        subtype = self.mosaic._index_to_subtype_dict[self.mosaic.grid[i,j]]
        # put the image through the color filter of the subtype
        bipolar_image_seen = self._compute_subtype_image(subtype, method = method)
        
        # now get the output of the pixel at the center of the receptive field
        center_idx = tf.constant([center_pos[0], center_pos[1]], dtype=tf.int32)
        return tf.gather_nd(bipolar_image_seen, center_idx[tf.newaxis, :])[0]

    def _build_subtype_stack(self, method="grayscale"):
        '''
        builds a TensorFlow tensor that stacks all subtype responses images together
        '''
        images = []
        for subtype in self.mosaic.subtypes:
            images.append(self._compute_subtype_image(subtype, method=method))
        return tf.stack(images, axis=0)
    
    # TODO: should rename this to something like get_all_cell_responses
    def get_all_average_colors(self, method = 'grayscale', blur_sigma=None):
        """
        Returns the output of each bipolar cell in the mosaic.
        If stimulation mosaic is provided, bipass normal color filter stuff and set each cell's
        avg color/output to the value in the stimulation mosaic instead of computing it from the image.

        Optional:
        - blur_sigma: if provided, apply a masked gaussian blur with this sigma to the flattened grid to
          simulate simple amacrine cells.
        """
        stimulation_mosaic = self.stimulation_mosaic

        # get all valid cell coordinates in the mosaic grid (i,j where there is a cell)
        valid_coords = self._valid_cell_coords
        valid_coords_tf = tf.convert_to_tensor(valid_coords, dtype=tf.int32)

        # if stimulation mosaic is provided, pull those values at valid coords
        if stimulation_mosaic is not None:
            stim = tf.convert_to_tensor(stimulation_mosaic, dtype=tf.float64)
            cell_outputs = tf.gather_nd(stim, valid_coords_tf)
        else:
            # build a stack of subtype images: one (H x W) resp image per subtype
            subtype_stack = self._build_subtype_stack(method = method)

            # map from mosaic subtype index -> position along first axis of subtype_stack
            subtype_index_to_stack = {
                self.mosaic.subtype_index_dict[subtype.name]: idx
                for idx, subtype in enumerate(self.mosaic.subtypes)
            }
            # look up each valid cell's subtype index in the mosaic grid
            cell_subtype_indices = self.mosaic.grid[valid_coords[:, 0], valid_coords[:, 1]]
            # convert to the corresponding indices into the subtype_stack
            stack_indices = np.array(
                [subtype_index_to_stack[idx] for idx in cell_subtype_indices],
                dtype = np.int32,
            )

            # gather the center pixel output for each cell from the appropriate subtype image
            indices = tf.stack(
                [
                    tf.convert_to_tensor(stack_indices, dtype = tf.int32),
                    tf.convert_to_tensor(self._center_coords[:, 0], dtype = tf.int32),
                    tf.convert_to_tensor(self._center_coords[:, 1], dtype = tf.int32),
                ],
                axis = 1,
            )
            cell_outputs = tf.gather_nd(subtype_stack, indices)

        # save as a 1D tensor of outputs for valid cells; no dict in tf version
        self.cell_outputs = cell_outputs
        self.avg_colors_cell_map = None

        # now scatter the cell outputs back onto the mosaic grid, -1 for empty cells
        h, w = self.mosaic.grid.shape
        valid_mask = tf.convert_to_tensor(self.mosaic.grid != -1)
        grid_outputs = tf.scatter_nd(valid_coords_tf, cell_outputs, [h, w])
        grid_outputs = tf.where(valid_mask, grid_outputs, tf.cast(-1.0, grid_outputs.dtype))
        if blur_sigma is not None:
            grid_outputs = gaussian_blur_reflect_mask_tf(grid_outputs, sigma = blur_sigma)
        self.grid_outputs = grid_outputs

    def get_avg_colors_cell_map_numpy(self):
        """
        Returns a numpy-backed dict that maps each valid mosaic cell (i,j) to its
        bipolar output value 

        Requires get_all_average_colors to have been called first so that
        self.cell_outputs has been populated.
        """
        # if we have already built this dict, just return it
        if self.avg_colors_cell_map is not None:
            return self.avg_colors_cell_map

        # need the per-cell outputs in a TF tensor to convert to numpy
        if not hasattr(self, 'cell_outputs'):
            raise ValueError('cell_outputs not computed; call get_all_average_colors first.')

        avg_colors_cell_map = {}
        valid_coords = self._valid_cell_coords
        cell_outputs_np = self.cell_outputs.numpy()

        # zip through valid cell coords and their outputs and build a plain dict
        for coord, val in zip(valid_coords, cell_outputs_np):
            avg_colors_cell_map[(int(coord[0]), int(coord[1]))] = val

        self.avg_colors_cell_map = avg_colors_cell_map
        return avg_colors_cell_map

    def get_avg_color_map_per_pixel(self):
        """
        Returns the average color/output at each pixel in the image for every subtype.

        For each subtype, pools over receptive fields to build a dict that maps
        image pixel -> average bipolar response across all cells of that subtype
        whose receptive fields include that pixel.
        """
        # make sure receptive field map exists
        if not self._receptive_field_map:
            raise ValueError('Receptive field map not built. Set build_receptive_fields=True.')

        # get (i,j) -> cell output as a numpy dict
        avg_color_map = self.get_avg_colors_cell_map_numpy()

        for subtype in self.mosaic.subtypes:
            subtype_index = self.mosaic.subtype_index_dict[subtype.name]

            # get the map of mosaic cell: image pixels for this subtype only
            rec_fields = {
                cell: pixels
                for cell, pixels in self._receptive_field_map.items()
                if self.mosaic.grid[cell] == subtype_index
            }

            # now create pixel:avg_color(s) dict for this subtype
            pixel_to_avg_colors = {}

            for cell, pixels in rec_fields.items():
                avg_color = avg_color_map[cell]
                for pixel in pixels:
                    if pixel[0] < self.image.shape[0] and pixel[1] < self.image.shape[1]:
                        pixel_to_avg_colors.setdefault(pixel, []).append(avg_color)

            # now a second dict for pixel: average of the average colors
            pixel_to_final_avg = {
                pixel: np.mean(colors, axis = 0) for pixel, colors in pixel_to_avg_colors.items()
            }

            self.avg_subtype_response_per_pixel[subtype.name] = pixel_to_final_avg





