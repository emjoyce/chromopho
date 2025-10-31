import matplotlib.pyplot as plt
import numpy as np
# import chromopho.mosaic as mosaic
from scipy.ndimage import gaussian_filter
from .utils import _parse_cone_string
from collections import defaultdict


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




def bipolar_image_filter_rgb(
    rgb_image,
    center_cones,
    surround_cones,
    center_sigma=1.0,
    surround_sigma=3.0,
    alpha_center=1.0,
    alpha_surround=0.4,
    rgb_to_lms = np.array([
        [0.313, 0.639, 0.048],  # L
        [0.155, 0.757, 0.088],  # M 
        [0.017, 0.109, 0.874]]),   # S 
    default_value = [0.5,0.5,0.5], 
    method = 'grayscale',
    rec_kind='softplus', 
    rec_r0=0.05, 
    rec_alpha=0.05, 
    rec_beta=7.5):
    """
    Returns an rbg image showing how a center-surround bipolar cell would
    respond in color space. 
    uses a grey (0.5,0.5,0.5) as starting point in LMS color spaceto represent response so that 
    increased and decreased response can be encoded i.e. a S center, -ML surround, both the +S and -ML can be encoded
    """
    # becuase some images have an alpha value, remove the alpha value


    # now if the places where alpha == 0 is black, replace with white because we need the contrast to see black logos 
    if rgb_image.shape[-1] == 4:
        alpha_mask = rgb_image[..., -1] == 0
        rgb_image[alpha_mask, :3] = 1

    if rgb_image.shape[2] > 3:
        rgb_image = rgb_image[:, :, :3]
    
    # rgb to lms
    baseline = 0.5  
    L = np.sum(rgb_image * rgb_to_lms[0], axis=2) - baseline
    M = np.sum(rgb_image * rgb_to_lms[1], axis=2) - baseline
    S = np.sum(rgb_image * rgb_to_lms[2], axis=2) - baseline

    # valence/value of lms center/surround
    cL, cM, cS = _parse_cone_string(center_cones)   
    sL, sM, sS = _parse_cone_string(surround_cones)

    # apply center and surround to whole LMS images
    center_img = np.stack([cL * L, cM * M, cS * S], axis=-1)  
    surround_img = np.stack([sL * L, sM * M, sS * S], axis=-1)

    # apply gaussian filters to each channel
    for channel in range(3):
        # LMS space responses for center and surround based on the parseing of the cone strings
        center_img[..., channel] = gaussian_filter(center_img[..., channel], center_sigma)
        surround_img[..., channel] = gaussian_filter(surround_img[..., channel], surround_sigma)

    
    if method == 'lms':
        # combine center and surround into one image
        final_lms = alpha_center * center_img + alpha_surround * surround_img
        # make a baseline rgb image, defaults baseline to gray
        baseline_lms = np.array(default_value)

        final_lms[...,0] += baseline_lms[0]/2 #divided by 2 because adding from .5 # TODO: what if the starting point is not .5
        final_lms[...,1] += baseline_lms[1]/2
        final_lms[...,2] += baseline_lms[2]/2        
        
        # back to rgb from lms
        lms_to_rgb = np.linalg.inv(rgb_to_lms)
        rgb_out = np.dot(final_lms, lms_to_rgb.T)

        # accounting for l and m cones being close to each other, bring min val to 0, norm by range to make response between 0 and 1
        min_val, max_val = rgb_out.min(), rgb_out.max()
        rgb_out = (rgb_out-min_val)/(max_val-min_val)

        return rgb_out
    
    elif method == 'grayscale' or method == 'greyscale':
        # combine center and surroudn into one output by taking center - surround 
        # so take center response and subtract surround response
        # if i have two or more responses in center or surround, take the average of them 
        # drop the columns from center_img that are 0 in cL, cM, cS
        keep_columns_center = np.array([cL, cM, cS], dtype=bool)
        new_center_img = center_img[..., keep_columns_center]

        # drop the columns from surround_img that are 0 in sL, sM, sS
        keep_columns_surround = np.array([sL, sM, sS], dtype=bool)
        new_surround_img = surround_img[..., keep_columns_surround]

        avg_center = np.mean(new_center_img, axis = -1)
        avg_surround = np.mean(new_surround_img, axis = -1)
        # subtract the two to get the output

        # just add cause one should be -
        output = alpha_center*avg_center+alpha_surround*avg_surround 
        # need to normalize, but not in a way that would 
        # example - if i have an image of all yellow, i want the response to be different for each cone type 
        # so I wont normalize to 0-1, but I will normalize to the range of the output
        # so what is the theoretical max and min here? -1 to 1 right? 
        # ok: normalize to make -1 0 and 1 1 
        # TODO: this might be better as sigmoid or something 

        # output = (output + 1)/2
        # ok this is what it needs to be, Half-wave rectification: 
        def rectifier(x, kind='baseline', r0=0.05, alpha=0.05, beta=15.0):
            """
            Returns a softened ON and OFF pair for a linear signal *x*.
            """
            if kind == 'baseline':      
                output  = np.maximum(0, x) + r0
            elif kind == 'leaky':       
                output  = np.where(x>0, x, alpha*x)
            elif kind == 'softplus':    
                base = np.log1p(np.exp(0)) / beta
                # soft = lambda z: (np.log1p(np.exp(beta * z)) / beta) - (np.log1p(np.exp(0)) / beta)
                output = np.log1p(np.exp(beta * x)) / beta - base

            else:
                raise ValueError("kind must be 'baseline', 'leaky', or 'softplus'")
            return output

        output = rectifier(output, kind=rec_kind, r0=rec_r0, alpha=rec_alpha, beta=rec_beta)
        r_max = (alpha_center - alpha_surround) * baseline     # with α_center > α_surround
        output = np.clip(output / r_max, 0, 1)
        return output




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
                    cell_outputs[i, j] = bipolar_img.get_average_color_of_cell(i, j, method=method).mean()

    # Apply Gaussian blur if requested
    if amacrine_sigma is not None:
        cell_outputs = gaussian_filter(cell_outputs, sigma=amacrine_sigma)

    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(cell_outputs, cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if not show_ax:
        ax.axis('off')
    ax.set_title('Graded response of each cell in mosaic')
    return ax

