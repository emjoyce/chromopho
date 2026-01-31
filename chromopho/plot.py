import matplotlib.pyplot as plt
import numpy as np
# import chromopho.mosaic as mosaic
from scipy.ndimage import gaussian_filter, distance_transform_edt
from .utils import _parse_cone_string, gaussian_blur_reflect_mask, amacrine_crossover_minimal
from collections import defaultdict
from pathlib import Path


def center_x_plot(r, n, mosaic, n_cells_mosaic = 25000):
    '''
    picks the center pixels radius r away from center in a mosaic n times and plots the rate of subtypes 
    '''
    
    # pick the center x 
    i_center, j_center = int(mosaic.grid.shape[0]/2), int(mosaic.grid.shape[1]/2)
    # constrain search to bounding box
    i_min, i_max = i_center-r, i_center+r
    j_min, j_max = j_center-r, j_center+r

    # get an array of valid indices 
    circle_indices = []
    for i in range(i_min, i_max):
        for j in range(j_min, j_max):
            # check if this i, j pair is in the cirle
            if ((i-i_center)**2 +(j-j_center)**2) < r**2:
                circle_indices += [[i,j]]

    n_cells = len(circle_indices)
    print(f'number of cells under electrode: {n_cells}')
    circle_indices =list(zip(*circle_indices))

    # calculated the expected number of cells per subtype based on ratios 
    # this needs to have ind:expected number of cells for that subtype
    
    expected_dict = {subtype_ind:int(mosaic._index_to_subtype_dict[subtype_ind].ratio*n_cells) for subtype_ind, subtype in mosaic._index_to_subtype_dict.items() if subtype_ind != -1}
    
    # create the mosaic 
    fig, ax = plt.subplots(ncols = n, nrows = n, figsize = (10*n,10*n))
    for i in range(n):

        for j in range(n):
            # randomly shift circle_indives by a random amount that is constrained by the size of the mosaic
            circle_indices_here = np.array(circle_indices)
            circle_indices_here[0] = circle_indices_here[0] + np.random.randint(0, 20)
            circle_indices_here[1] = circle_indices_here[1] + np.random.randint(0, 20)

            
            # get the identity of all indices in circle_indices
            sub_inds = mosaic.grid[*circle_indices_here]
            subs, counts = np.unique(sub_inds, return_counts = True)
            ind_name_dict = {v:k for k,v in mosaic.subtype_index_dict.items()}
            sub_names = [ind_name_dict[ind] for ind in subs]
            expected_counts = [expected_dict[sub] for sub in subs]
            
            x_axis = np.arange(len(subs))
            width = .2
            ax[i,j].bar(x_axis - width/2, counts, width, label = 'real values', color = '#038dc7')
            ax[i,j].bar(x_axis + width/2, expected_counts, width, label = 'expected values', color = '#63e572')
            
            ax[i,j].set_xticks(x_axis, sub_names)
            ax[i,j].legend()

def graph_receptive_fields(bipolar_img, img, subtypes=None, filter=None, ax=None, title=None, show_ax = True):
    '''
    graphs the parts of the image that are seen by the bipolar cells in the mosaic
    params:
    bipolar_img: BipolarImage object
    img: pulse2percept image object
    subtypes: optional, filter by specific subtypes
    filter: optional, additional filter
    ax: optional, matplotlib axis object
    title: optional, title for the plot
    '''
    if ax is None:
        fig, ax = plt.subplots()
    # graph the part of the image that is covered here 
    image_pixels = np.array([sublist for sublist in list(bipolar_img._receptive_field_map.values()) for sublist in sublist])
    
    mosaic = bipolar_img.mosaic
    subtype_mask = None
    if subtypes:
        # Get the indices for the specified subtypes
        subtype_indices = [mosaic.subtype_index_dict[subtype] for subtype in subtypes if subtype in mosaic.subtype_index_dict]
        # get the map of the grid that has this subtype
        subtype_mask = np.isin(bipolar_img.mosaic.grid, subtype_indices)
        subtype_mosaic_inds = list(zip(*np.where(subtype_mask)))
        # so we have the cells in grid that are of the specified subtypes, now we need to get the indices of the pixels that are covered by these cells
        # we can use the subtype_mosaic_inds to get the corresponding indices in the image_pixels
        subtype_image_pixels = [bipolar_img._receptive_field_map[pair] for pair in subtype_mosaic_inds]
        subtype_image_pixels = list(set([pair for sublist in subtype_image_pixels for pair in sublist if pair[0] < img.shape[0] and pair[1] < img.shape[1]]))
        unzipped_i, unzipped_j = zip(*subtype_image_pixels)
        # now this to a mask 
        subtype_mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
        
        subtype_mask[unzipped_i, unzipped_j] = 1


    
    # graph the logo just at these indices
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
    for pair in image_pixels:
        try:
            mask[pair[0], pair[1]] = 1
        except:
            pass

    if subtype_mask is not None:
        mask = mask & subtype_mask
    rgba = img.reshape(img.shape)
    new_img = rgba * mask[..., np.newaxis]
    
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(new_img[...,:3])
    
    if title:
        ax.set_title(title)
    
    if not show_ax:
        ax.axis('off')


def plot_mosaic(mosaic, ax=None, title=None, palette = 'viridis', plot_legend = True, legend_loc = 'upper left',
                    bbox_to_anchor = (1.05, 1)):
    '''
    plots the mosaic with each subtype in a different color
    '''
    if ax is None:
        fig, ax = plt.subplots()
    
    # Create a colormap for the subtypes
    unique_subtypes = np.unique(mosaic.grid)
    colormap = plt.cm.get_cmap(palette, len(unique_subtypes))
    subtype_colors = {subtype: colormap(i) for i, subtype in enumerate(unique_subtypes)}
    
    # Create an RGB image where each subtype is colored
    rgb_image = np.zeros((*mosaic.grid.shape, 3))
    for subtype, color in subtype_colors.items():
        mask = mosaic.grid == subtype
        rgb_image[mask] = color[:3]  # Ignore the alpha channel
    
    ax.imshow(rgb_image)
    
    # Create a legend
    if plot_legend:
        flipped_dict = {v:k for k,v in mosaic.subtype_index_dict.items()}
        print(flipped_dict)
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color[:3], markersize=10, label=flipped_dict[subtype]) for subtype, color in subtype_colors.items()]
        ax.legend(handles=handles, title="Subtypes", bbox_to_anchor=bbox_to_anchor, loc=legend_loc, ncol=len(handles))
    
    if title:
        ax.set_title(title)
    return ax



def bipolar_image_filter(rgb_image, center_cones, surround_cones,
    center_sigma=1.0,
    surround_sigma=3.0,
    cone_center_sigma = 1,
    alpha_center=1.0,
    alpha_surround=1.0,
    apply_rectification = True,
    on_k=0.7,
    on_n=2.0,
    off_k=0.7,
    off_n=1.5,  
    nonlin_adapt_cones = True,
    sigma_adapt = 4.0,
    rgb_to_lms = np.array([
    [0.313, 0.639, 0.048],  # L
    [0.155, 0.757, 0.088],  # M 
    [0.017, 0.109, 0.874]])):   # S 
    '''returns a grayscale image showing how a bipolar cell of a subtype would respond to the input rgb image'''
    if rgb_image.shape[-1] == 4:
        alpha_mask = rgb_image[..., -1] == 0
        rgb_image[alpha_mask, :3] = 1
    if rgb_image.shape[2] > 3:
        rgb_image = rgb_image[:, :, :3]

    # linearize the rgb image
    def srgb_to_linear(x):
        a = 0.055
        return np.where(x <= 0.04045, x/12.92, ((x + a)/(1 + a))**2.4)
    rgb_image = srgb_to_linear(rgb_image)
    lms_img = rgb_image @ rgb_to_lms.T

    L = lms_img[..., 0]
    M = lms_img[..., 1]
    S = lms_img[..., 2]

    # apply a nonlinearity to these channels and adaptation
    def cone_stage_with_adaptation(LMS, lam = np.array([.2,.2,.2], dtype=np.float32), 
                                   gamma = np.array([1.2, 1.2, 1.2], dtype=np.float32), 
                                   sigma =np.array([.25, .25, .25], dtype=np.float32), 
                                   sigma_adapt = 4.0):
        LMS_nonlin = np.power(LMS + lam.reshape(1,1,3), gamma.reshape(1,1,3))
        # normalize LMS nonlin
        y0 = np.power(lam, gamma)      # val at LMS = 0,0,0
        y1 = np.power(1.0 + lam, gamma) # val at LMS = 1,1,1
        endpoint_range = np.maximum(y1 - y0, 1e-12) 
        LMS_nonlin = (LMS_nonlin - y0) / endpoint_range
        adaptation_signal = np.stack([gaussian_filter(LMS_nonlin[...,i], sigma_adapt) for i in range(3)], axis=-1)
        gain = np.asarray(sigma, dtype=float).reshape(1, 1, 3)
        LMS_nonlin_adapt = np.clip(LMS_nonlin / (gain + adaptation_signal + 1e-12), 0, 1)
        return LMS_nonlin_adapt

    if nonlin_adapt_cones:
        LMS = np.stack([L,M,S], axis = 2)
        LMS = cone_stage_with_adaptation(LMS, sigma_adapt = sigma_adapt)
        L = LMS[...,0]; M = LMS[...,1]; S = LMS[...,2]

    # valence/value of lms center/surround
    cL, cM, cS = _parse_cone_string(center_cones)   
    sL, sM, sS = _parse_cone_string(surround_cones)
    # apply center and surround to whole LMS images
    center_img = np.stack([cL * L, cM * M, cS * S], axis=-1)  
    surround_img = np.stack([sL * L, sM * M, sS * S], axis=-1)

    # pull out center and surround
    keep_columns_center = np.array([cL, cM, cS], dtype=bool)
    new_center_img = center_img[..., keep_columns_center]

    # drop the columns from surround_img that are 0 in sL, sM, sS
    keep_columns_surround = np.array([sL, sM, sS], dtype=bool)
    new_surround_img = surround_img[..., keep_columns_surround]

    # if there are multiple channels in the center or surround (i.e. l and m) average their values
    # will still be between 0, 1 (but with *-1 if the polarity is - for that center or surround)
    avg_center = np.mean(new_center_img, axis = -1)
    avg_surround = np.mean(new_surround_img, axis = -1)
    
    # gaussian
    # LMS space responses for center and surround based on the parseing of the cone strings
    # will still be between -1, 0 for one and 0, 1 for the other
    # rounding becuase of floating point errors so it stays in this^ range
    center_img = np.round(gaussian_filter(avg_center, center_sigma), 12)
    surround_img = np.round(gaussian_filter(avg_surround, surround_sigma), 12)
    
    output = alpha_center*center_img+alpha_surround*surround_img 
    
    # now normalize 
    # the min value after summing will be the negative one *alpha and the max will be the positive one times alpha 
    
    pol_center   = np.sign(np.sum([cL, cM, cS]))  # +1 ON, -1 OFF
    pol_surround = np.sign(np.sum([sL, sM, sS]))

    # if center is the negative one, the min will be pol_center*alpha_center,
    # if center is the positive one, the max will be pol_center*alpha_center
    abs_min = min(alpha_center*pol_center, alpha_surround*pol_surround)
    abs_max = max(alpha_center*pol_center, alpha_surround*pol_surround)
    max_amp = abs(abs_max)+abs(abs_min) # because one of them will be negative 
    output_normalized = np.clip((output - abs_min) / (abs_max - abs_min), 0.0, 1.0) # clamp for insurance purposes only
    
    # return output_normalized
    # return if no rec
    if not apply_rectification:
        return output_normalized

    # Naka–Rushton rectifiers without sign flip (bc sign is already considered) and with dif params for on and off
    def _nr_base(x, k, n):
        x = np.clip(x, 0.0, 1.0)
        k = max(float(k), 1e-12); n = float(n)
        xn = np.power(x, n); kn = np.power(k, n)
        return xn / (xn + kn + 1e-12)

    def on_rectifier_nr(x, k, n):
        # low-end compression, output [0,1]
        s = 1.0 + k**n
        return np.clip(s * _nr_base(x, k, n), 0.0, 1.0)

    def off_rectifier_nr_high(x, k, n):
        # high-end compression, output [0,1]
        s = 1.0 + k**n
        return np.clip(1.0 - s * _nr_base(1.0 - x, k, n), 0.0, 1.0)

    if pol_center > 0:
        output_rectified = on_rectifier_nr(output_normalized, on_k, on_n)
    else:
        output_rectified = off_rectifier_nr_high(output_normalized, off_k, off_n)
    
    return output_rectified




def plot_average_color_rec_field(bipolar_img, subtype_name, ax=None, show_ax = False):
    '''
    takes a bipolar linked image, plots the receptive field of the specified cell type, but each receptive field returns the average color that cell sees
    
    '''
    if ax is None:
        fig, ax = plt.subplots()
    pixel_to_final_avg = bipolar_img.avg_subtype_response_per_pixel[subtype_name]
    final_img = np.ones((*bipolar_img.image.shape[0:2], 3))
    
    for pixel, avg_color in pixel_to_final_avg.items():
        final_img[pixel[0], pixel[1]] = avg_color
    #plot
    if not show_ax:
        ax.axis('off')
    ax.imshow(final_img)

def features_img(feats, w, h):
    '''
    generates image from vector of features of each bipolar subtype response 
    
    '''
    # make a len(feats)-2 x 1 array of images
    fig, ax = plt.subplots(1, feats.shape[1]-2, figsize=(20, 20))
    i_inds = feats.T[0].astype(int)
    j_inds = feats.T[1].astype(int)
    for i, type_feats in enumerate(feats.T[2:]):
        # make an empty image of 0s 
        img = np.ones((w,h,3))
        # this will be gray, with same values in r, g, b
        img[i_inds, j_inds] = np.stack([type_feats]*3, axis=1)
        ax[i].imshow(img)


def graph_phosphenes(i, j, model, mosaic, radius=5, stim_response=1, tensor_model = True, smooth = True, gaussian_blur = False, 
                        black_encoding =  {-1:0, 1:.6, 2:.01, 3:.6, 4: .01, 5:.7, 6:.01, 7:.01, 8:.6},
                        random_state = 0):
    dummy_img = np.zeros((400, 400, 3))
    fig, axes = plt.subplots(i, j, figsize=(i*10, j*10))

    np.random.seed(random_state)
    positions = [(x, y) for x in range(i) for y in range(j)]
    seeds = np.random.randint(0, 1e9, size=len(positions))  
    for (seed, (ix, jx)) in zip(seeds, positions):
        # getting around circular import 
        import chromopho.model as modeling
        # phosphene simulation from this model
        phosphene_stim_percept = modeling.phosphene_simulation(
            radius=radius,
            mosaic=mosaic,
            model=model,
            dummy_img=dummy_img,
            stim_response=stim_response,
            seed=seed,
            tensor_model = tensor_model,
            smooth = smooth,
            gaussian_blur = gaussian_blur,
            black_encoding = black_encoding
        )
        
        # remove blue background
        # blue_mask = (phosphene_stim_percept[...,0] == 0) & (np.isclose(phosphene_stim_percept[...,1], 0.847, atol=1e-3))  & (phosphene_stim_percept[...,2] == 1)
        # test_p = phosphene_stim_percept.copy()
        # test_p[blue_mask] = np.array([0,0,0])
        
        axes[ix, jx].imshow(phosphene_stim_percept)
        axes[ix, jx].axis('off')  # optional, make it cleaner

    plt.tight_layout()



def plot_mosaic_cell_outputs(bipolar_img, method='lms', ax=None, cmap='gray', show_ax=False, amacrine_sigma=1):
    """
    Plots the mosaic with each cell colored by its graded response (output).
    Optionally applies Gaussian blur to simulate amacrine cell smoothing.
    """
    mosaic = bipolar_img.mosaic
    grid = mosaic.grid
    h, w = grid.shape

    # Get output for each cell
    # check for bipolar_img.grid_output
    if hasattr(bipolar_img, 'grid_output'):
        cell_outputs = bipolar_img.grid_output
    else:
        cell_outputs = np.full((h, w), np.nan)
        for i in range(h):
            for j in range(w):
                if grid[i, j] != -1:
                    cell_outputs[i, j] = bipolar_img.compute_cell_output(i, j, method=method).mean()

    # Apply Gaussian blur if requested
    if amacrine_sigma is not None:
        # use shared helper from utils which handles invalid (-1) cells safely
        cell_outputs = gaussian_blur_reflect_mask(cell_outputs, sigma=amacrine_sigma)

    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(cell_outputs, cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if not show_ax:
        ax.axis('off')
    ax.set_title('Graded response of each cell in mosaic')
    return ax





# local NN section


def local_phosphene(
    mosaic_grid: np.ndarray,
    mosaic_background_values,
    i: int, j: int,
    radius: float,
    shape: str = "hex",                  # "circle" or "hex" 
    values_by_subtype: dict = {
    'm_off': 0.5, 'm_on': 0.7,
    'l_off': 0.5, 'l_on': 0.7,
    's_off': 0.5, 's_on': 0.7,
    'dif_on': 0.5, 'dif_off': 0.7},         
    subtype_dict: dict = {'m_off': 1, 'm_on': 2, 'l_off': 3, 'l_on': 4, 's_off': 5, 's_on': 6,
                                 'dif_on': 7, 'dif_off': 8, 'none': -1},
    invalid_code: int = -1,
    soft_sigma: float = .1,                
    truncate: float = 5.0,                  
    response_gain_by_subtype: dict = None,  # e.g. {'s_on': 1.2, 'dif_off': 0.6}
    sigma_by_subtype: dict = None,          # absolute override per subtype (in pixels)
    scale_inside_by_gain: bool = False,     
    in_place: bool = False,
    return_cells = False, # if true return the number of cells changed 
    amacrine_blur = True
    ):
    """
    Softly blend mosaic values toward subtype-specific targets with Gaussian falloff from a circle/hex.
      mosaic_grid: 2D array of subtype type ints per cell. (mosaic.grid)
      mosaic_background_values: 2d array of background values for each cell. (e.g. black_bipolar_outputs)
        (if mosaic_background_values is None, will load black_bipolar_outputs from local file and use that as background)
      i, j: center of stimulation (in pixel coordinates)
      radius: radius of stimulation (in pixels)
      shape: "circle" or "hex"
      soft_sigma: value of sigma relative to radius (sigma = soft_sigma * radius) for Gaussian falloff.
      truncate: truncate Gaussian at this many sigmas (set to None for no truncation).
      scale_inside_by_gain: if True, inside shape weight = clip(gain,0, 1) instead of 1.
      in_place: if True, modifies mosaic_background_values in place; else returns a copy.
      return_cells: if True, prints and returns number of cells under electrode by subtype and total changed
      amacrine_blur: if True, applies Gaussian blur to simulate amacrine cell effects
    
    """
    if mosaic_background_values is None:
        data_file = Path(__file__).resolve().parent / "mosaic_outputs" / "black_bipolar_outputs.npy"
        mosaic_background_values = np.load(data_file)

    if values_by_subtype is None or soft_sigma <= 0:
        return mosaic_background_values


    H, W = mosaic_background_values.shape
    out = mosaic_background_values if in_place else mosaic_background_values.copy()
    before = out.copy()

    def _map_dict_any(d):
        if d is None:
            return {}
        mapped = {}
        for k, v in d.items():
            if isinstance(k, str):
                if k in subtype_dict:
                    mapped[subtype_dict[k]] = float(v)
            else:
                mapped[int(k)] = float(v)
        return mapped

    code_to_val   = _map_dict_any(values_by_subtype)
    code_to_gain  = _map_dict_any(response_gain_by_subtype)
    code_to_sigma = _map_dict_any(sigma_by_subtype)

    yy, xx = np.ogrid[:H, :W]
    dy = yy - i
    dx = xx - j

    # if radius = 1, change shape to circle 
    if radius == 1:
        shape = "circle"
    if shape.lower() == "circle":
        inside = (dx*dx + dy*dy) <= (radius*radius)
    elif shape.lower() == "hex":
        # Flat-bottom / flat-top hex (horizontal flats). Apothem a = (√3/2)*r
        a = radius * (np.sqrt(3) / 2.0)
        dot1 = dy
        dot2 = (np.sqrt(3)/2.0)*dx - 0.5*dy
        dot3 = -(np.sqrt(3)/2.0)*dx - 0.5*dy
        inside = (np.abs(dot1) <= a) & (np.abs(dot2) <= a) & (np.abs(dot3) <= a)

    else:
        print(shape)
        raise ValueError("shape must be 'circle' or 'hex'")

    # distance from each pixel to the nearest inside pixel (0 for inside)
    dist_outside = distance_transform_edt(~inside)

    # pick out valid cells 
    allowed = (mosaic_grid != invalid_code)
    
    inside_allowed = inside & allowed
    counts_by_name = {}
    if return_cells:
        for name, code in subtype_dict.items():
            if code == invalid_code:
                continue
            counts_by_name[name] = int(np.count_nonzero(inside_allowed & (mosaic_grid == code)))
            
        print("Cells under electrode by subtype:", counts_by_name)
    changed = sum(counts_by_name.values())

    for code, target in code_to_val.items():
        mask_code = (mosaic_grid == code) & allowed
        if not np.any(mask_code):
            continue

        # tie sigma to stimulation radius if no per-subtype override
        sigma_c = float(code_to_sigma.get(code, soft_sigma * radius))
        gain_c  = float(code_to_gain.get(code, 1.0))

        # Build per-subtype weight field
        w_code = np.zeros_like(out, dtype=float)

        # Inside: either full effect (=1) or scaled by gain (clipped)
        if scale_inside_by_gain:
            w_code[inside] = np.clip(gain_c, 0.0, 1.0)
        else:
            w_code[inside] = 1.0

        # Outside: Gaussian tail * gain, clipped to [0,1]
        if sigma_c > 0:
            w_out = gain_c * np.exp(-0.5 * (dist_outside / sigma_c) ** 2)
            if truncate is not None and truncate > 0:
                w_out = np.where(dist_outside <= (truncate * sigma_c), w_out, 0.0)
            w_out = np.clip(w_out, 0.0, 1.0)
            w_code[~inside] = w_out[~inside]

        # Apply to this subtype only
        if np.any(mask_code):
            w_eff = w_code[mask_code]
            old   = out[mask_code]
            new   = old * (1.0 - w_eff) + target * w_eff
            out[mask_code] = new

    if return_cells:
        print(f"Cells under electrode: {changed}")
        total_changed = np.count_nonzero(np.abs(out - before) > 1e-12)
        print(f"Cells affected: {total_changed}")
        return out, changed, total_changed
    if amacrine_blur:
        out = amacrine_crossover_minimal(out, mosaic_grid, subtype_dict, sigma =2, beta = .15, 
                                               same_polarity_unsharp=False)

    return out

