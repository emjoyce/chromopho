import numpy as np
import os
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.stats import gaussian_kde


def _parse_cone_string(cone_string):
    '''
    Parses something like '+l', '-m', '+ls', '-lms' into an (L, M, S) tuple
    indicating +1 or -1 for each relevant cone.
    Example:
        '+ls' -> (1, 0, 1)
        '-lm' -> (-1, -1, 0)
    '''
    sign = 1 if cone_string[0] == '+' else -1
    l, m, s = 0, 0, 0
    for c in cone_string[1:]:  # skip the initial '+' or '-'
        if c == 'l':
            l += sign
        elif c == 'm':
            m += sign
        elif c == 's':
            s += sign
    return (l, m, s)


def _subtype_to_cone_string(subtype):
    '''takes subtype, gets cone string, returns center cone and surroudn cone string'''
    sub_name = subtype.name
    center_cone = ''
    surround_cone = ''
    if 'on' in sub_name:
        center_cone = '+'
        surround_cone = '-'
    if 'off' in sub_name:
        center_cone = '-'
        surround_cone = '+'
    if 's' in sub_name:
        center_cone += 's'
        surround_cone += 'lm'
    if 'm' in sub_name:
        center_cone += 'm'
        surround_cone += 'l'
    if 'l' in sub_name:
        center_cone += 'l'
        surround_cone += 'm'
    return center_cone, surround_cone

def _subtype_to_lms(subtype):
    '''
    takes subtype, gets cone string, returns center cone and surround cone lms values
    '''
    center_cone_surround_cone = _subtype_to_cone_string(subtype)
    center_cone = _parse_cone_string(center_cone_surround_cone[0])
    surround_cone = _parse_cone_string(center_cone_surround_cone[1])
    return center_cone, surround_cone


def img_to_rgb(img):
    '''
    converts a p2p image to an rgb image
    '''
    return img.reshape(img.shape)[:,:,:3]



def save_structured_features(array, output_dir, filename_base, filename_extension):
    """
    Saves a numpy array with first two columns as integers and the rest as floats,
    ensuring type consistency when saving and loading.

    Parameters:
    - array (np.ndarray): Input numpy array with at least 2 columns.
    - output_dir (str): Directory where the file should be saved.
    - filename_base (str): Base filename (without extension).
    - filename_extension (str): File extension (e.g. '_features.npy' or '_labels.npy').
    
    Returns:
    - str: Full path of the saved file.
    """
    if array.shape[1] < 3:
        raise ValueError("Array must have at least 3 columns (x,y of pixel + at least 1 subtype response column).")

    # first two columns int, rest float
    num_float_cols = array.shape[1] - 2
    dtype = [('x', 'i4'), ('y', 'i4')] + [(f'col{i+3}', 'f4') for i in range(num_float_cols)]

    # Convert to structured array
    structured_array = np.zeros(array.shape[0], dtype=dtype)
    structured_array['x'] = array[:, 0].astype(np.int32)
    structured_array['y'] = array[:, 1].astype(np.int32)

    for i in range(num_float_cols):
        structured_array[f'col{i+3}'] = array[:, i+2]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save file
    filepath = os.path.join(output_dir, filename_base + filename_extension)
    np.save(filepath, structured_array)

def gaussian_blur_reflect_mask(arr, sigma):
    """
    Blur mosaic output while preserving values inside the valid mask (non-NaN / >=0).
    - Fill invalid pixels with the nearest valid pixel (distance transform).
    - Blur the filled image with reflect mode so kernel doesn't see zeros/NaNs.
    - Return an array that contains blurred values only at originally valid positions;
      invalid positions are set to NaN.
    """
    arr = np.asarray(arr, dtype=float)
    # consider valid those that are finite and non-negative sentinel values
    # if array uses -1 sentinel for invalid cells keep that semantics
    valid = np.isfinite(arr) & (arr >= 0)
    if not np.any(valid):
        return arr.copy()

    data = arr.copy()
    inv = ~valid
    if np.any(inv):
        # indices of nearest True in valid for every position where inv is True
        _, inds = distance_transform_edt(inv, return_indices=True)
        filled = data[inds[0], inds[1]]
    else:
        filled = data

    # blur the filled image; reflect keeps interior structure at edges
    blurred = gaussian_filter(filled, sigma=sigma, mode='reflect')

    # keep only values inside original valid mask
    out = np.full_like(arr, np.nan, dtype=float)
    out[valid] = blurred[valid]
    # replace nan with -1
    out[np.isnan(out)] = -1
    return out


import numpy as np

def amacrine_crossover_minimal(x, mosaic, mosaic_subtype_dict, sigma=1.5,
    beta=0.25, same_polarity_unsharp=False, alpha=0.10, sigma_n=1.0, rectify=True
):
    """
    Minimal amacrine-like static effect with *masked, renormalized* Gaussians.
      - ON inhibited by local OFF; OFF inhibited by local ON.
      - Invalid cells (-1) never contribute to any blur.
      - Returns -1 at invalid cells; leaves non-ON/OFF types unchanged.

    Parameters
    ----------
    x : (H, W) float array
        Bipolar outputs on the mosaic grid; invalid = -1.
    mosaic : (H, W) int array
        Subtype labels per position; invalid = -1.
    mosaic_subtype_dict : dict[str, int]
        Mapping like {'dif_on': 1, 'm_off': 2, ...}. Keys must contain 'on' or 'off'.
    sigma : float
        Gaussian std in *grid cells* for crossover pooling.
    beta : float
        Crossover inhibition strength.
    same_polarity_unsharp : bool
        If True, apply a light unsharp mask within ON and within OFF after crossover.
    alpha, sigma_n : float
        Unsharp mask params.
    rectify : bool
        If True, clamp negatives to 0 within each polarity map.

    Returns
    -------
    y : (H, W) float array
        Updated mosaic; invalid = -1.
    """

    # helpers
    def masked_gaussian(src_vals, src_mask, out_where, sigma):
        """
        Normalized convolution: (G * (src_vals * src_mask)) / (G * src_mask)
        Only src_mask==True contributes; result only written where out_where==True.
        Elsewhere returns -1 sentinel.
        """
        src_vals = np.asarray(src_vals, dtype=float)
        src_mask = src_mask.astype(float)

        num = gaussian_filter(src_vals * src_mask, sigma=sigma, mode='reflect')
        den = gaussian_filter(src_mask,           sigma=sigma, mode='reflect')
        # safe division
        out_full = np.zeros_like(num, dtype=float)
        np.divide(num, np.maximum(den, 1e-12), out=out_full)

        out = np.full_like(src_vals, -1.0, dtype=float)
        out[out_where] = out_full[out_where]
        return out

    def masked_unsharp(src_vals, src_mask, alpha, sigma):
        """ y <- y - alpha * ( masked_gaussian(y, src_mask) ) within src_mask """
        blurred = masked_gaussian(src_vals, src_mask, src_mask, sigma)
        y = np.full_like(src_vals, -1.0)
        y[src_mask] = src_vals[src_mask] - alpha * blurred[src_mask]
        return y

    # polarity masks
    on_ids  = {v for k, v in mosaic_subtype_dict.items() if 'on'  in k.lower()}
    off_ids = {v for k, v in mosaic_subtype_dict.items() if 'off' in k.lower()}

    valid_mask = (mosaic != -1) & np.isfinite(x) & (x >= 0)
    on_mask    = valid_mask & np.isin(mosaic, list(on_ids))
    off_mask   = valid_mask & np.isin(mosaic, list(off_ids))
    other_mask = valid_mask & ~(on_mask | off_mask)

    # Extract polarity maps (noncell stay -1)
    x_on  = np.full_like(x, -1.0); x_on[on_mask]   = x[on_mask]
    x_off = np.full_like(x, -1.0); x_off[off_mask] = x[off_mask]

    # masked crossover blurs
    # OFF pooled only from OFF cells, sampled only at ON cells
    off_blur_at_on = masked_gaussian(
        src_vals=np.where(off_mask, x, 0.0),
        src_mask=off_mask,
        out_where=on_mask,
        sigma=sigma
    )
    # ON pooled only from ON cells, sampled only at OFF cells
    on_blur_at_off = masked_gaussian(
        src_vals=np.where(on_mask, x, 0.0),
        src_mask=on_mask,
        out_where=off_mask,
        sigma=sigma
    )

    # crossover inhibition 
    y = np.full_like(x, -1.0)
    y[on_mask]  = x[on_mask]  - beta * off_blur_at_on[on_mask]
    y[off_mask] = x[off_mask] - beta * on_blur_at_off[off_mask]
    y[other_mask] = x[other_mask]  # pass through any non-ON/OFF types

    # same-polarity unsharp (also masked & renormalized) if same_polarity_unsharp
    if same_polarity_unsharp:
        y_on_sharp  = masked_unsharp(y, on_mask,  alpha=alpha, sigma=sigma_n)
        y_off_sharp = masked_unsharp(y, off_mask, alpha=alpha, sigma=sigma_n)
        y[on_mask]  = y_on_sharp[on_mask]
        y[off_mask] = y_off_sharp[off_mask]

    # rectify within polarity maps if rectify
    if rectify:
        y[on_mask]  = np.maximum(0.0, y[on_mask])
        y[off_mask] = np.maximum(0.0, y[off_mask])

    return y


# functions that allow each subtype to sample center surround relative weight

def bounded_log_kde_curve(data, n_grid=600, bw=1.0, pad_frac=0.05, 
                          taper_power=0.35):
    """
    func that creates a fitted kde curve around data
    data is center/surround ratios for midget or diffuse cells from Dacey 2000 paper 
    
    - Fits KDE in log space
    - Evaluates density back in ratio space using the Jacobian correction
    - Forces density to zero at a lower/upper support near the data range
    - Renormalizes so the area under the curve is 1
    parameters:
    - data: 1D array of ratio samples
    - n_grid: number of points to evaluate the curve on
    - bw: bandwidth multiplier for KDE
    - pad_frac: fraction of data range to pad on each side for support limits
    - taper_power: exponent for smooth tapering to zero at support limits
    returns:
    - x: (n_grid,) array of ratio values where the curve is evaluated
    - y: (n_grid,) array of density values corresponding to x
    - lo: lower support limit
    - hi: upper support limit
    """
    data = np.asarray(data, dtype=float)
    data_range = data.max() - data.min()

    lo = max(1e-6, data.min() - pad_frac * data_range)
    hi = data.max() + pad_frac * data_range

    x = np.linspace(lo, hi, n_grid)

    z = np.log(data)
    kde = gaussian_kde(z)
    kde.set_bandwidth(kde.factor * bw)

    # log density transformed back to ratio density
    y = kde(np.log(x)) / x

    # smooth finite support taper: to 0 at lo and hi
    t = (x - lo) / (hi - lo)
    window = np.sin(np.pi * t) ** taper_power
    y = y * window

    # normalize
    area = trapezoid(y, x)
    y = y / area

    return x, y, lo, hi

def make_surround_center_bins(cell_type,n_bins=5,
    midget_samples=(1.0, 0.9, 0.7, 1.1),
    diffuse_samples=(1.7, 0.8, 1.3, 0.9, 2.0, 2.8, 0.8, 1.3),
    bw=1, pad_frac=0.16, taper_power=0.55):
    """
    func that makes n bins to create discrete sampling of center surround ratios
    for midget or diffuse cells, based on the Dacey 2000 data and a KDE fit to that data
    parameters:
    - cell_type: 'midget' or 'diffuse'
    - n_bins: number of bins to create
    - midget_samples: tuple of center/surround ratios for midget cells
    - diffuse_samples: tuple of center/surround ratios for diffuse cells
    - bw: bandwidth multiplier for KDE
    - pad_frac: fraction of data range to pad on each side for support limits
    - taper_power: exponent for smooth tapering to zero at support limits
    """

    if cell_type == "midget":
        data = np.array(midget_samples, dtype=float)
    elif cell_type == "diffuse":
        data = np.array(diffuse_samples, dtype=float)
    else:
        raise ValueError("cell_type must be 'midget' or 'diffuse'")

    x, y, lo, hi = bounded_log_kde_curve(data,
        bw=bw,pad_frac=pad_frac,taper_power=taper_power)

    # Equal width bins 
    bin_edges = np.linspace(lo, hi, n_bins + 1)

    bin_values = []
    bin_probs = []

    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]

        mask = (x >= left) & (x <= right)
        x_bin = x[mask]
        y_bin = y[mask]

        prob = trapezoid(y_bin, x_bin)

        # representative value is density-weighted mean within bin
        val = trapezoid(x_bin * y_bin, x_bin) / prob

        bin_values.append(val)
        bin_probs.append(prob)

    bin_values = np.array(bin_values)
    # round down to nearest .1
    bin_values = np.floor(bin_values * 10) / 10
    bin_probs = np.array(bin_probs)
    bin_probs = bin_probs / bin_probs.sum()

    return bin_values, bin_probs

def sample_surround_center_ratio_binned(cell_type, n=1, n_bins=5,
    midget_samples=(1.0, 0.9, 0.7, 1.1),
    diffuse_samples=(1.7, 0.8, 1.3, 0.9, 2.0, 2.8, 0.8, 1.3)):
    """
    samples n surround/center ratios for midges or diffuse cell type
    first fits a kde to fit Dacey 2000 data, then creates n_bins bins 
    sampling that curve, then samples from those bins
    parameters:
    - cell_type: 'midget' or 'diffuse'
    - n: number of samples to pull
    - n_bins: number of bins to create
    - midget_samples: tuple of center/surround ratios for midget cells
    - diffuse_samples: tuple of center/surround ratios for diffuse cells
    returns:
    - samples: (n,) array of sampled center/surround ratios
    """
    rng = np.random.default_rng(0)

    bin_values, bin_probs = make_surround_center_bins(cell_type,
        n_bins=n_bins, midget_samples=midget_samples,
        diffuse_samples=diffuse_samples)

    return rng.choice(bin_values, size=n, replace=True, p=bin_probs)