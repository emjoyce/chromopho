import matplotlib.pyplot as plt
import numpy as np
import chromopho.mosaic as mosaic
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

def graph_receptive_fields(bipolar_img, img, subtypes=None, filter=None, ax=None, title=None):
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
        subtype_image_pixels = list(set([pair for sublist in subtype_image_pixels for pair in sublist if pair[0] < img.img_shape[0] and pair[1] < img.img_shape[1]]))
        unzipped_i, unzipped_j = zip(*subtype_image_pixels)
        # now this to a mask 
        subtype_mask = np.zeros((img.img_shape[0], img.img_shape[1]), dtype=bool)
        
        subtype_mask[unzipped_i, unzipped_j] = 1


    
    # graph the logo just at these indices
    mask = np.zeros((img.img_shape[0], img.img_shape[1]), dtype=bool)
    for pair in image_pixels:
        try:
            mask[pair[0], pair[1]] = 1
        except:
            pass

    if subtype_mask is not None:
        mask = mask & subtype_mask
    rgba = img.data.reshape(img.img_shape)
    new_img = rgba * mask[..., np.newaxis]
    
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.imshow(new_img)
    
    if title:
        ax.set_title(title)


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
    alpha_surround=0.8,
    rgb_to_lms = np.array([
        [0.313, 0.639, 0.048],  # L
        [0.155, 0.757, 0.088],  # M 
        [0.017, 0.109, 0.874]]),   # S 
    default_value = [0.5,0.5,0.5], 
    method = 'lms'):
    """
    Returns an rbg image showing how a center-surround bipolar cell would
    respond in color space. 
    uses a grey (0.5,0.5,0.5) as starting point in LMS color spaceto represent response so that 
    increased and decreased response can be encoded i.e. a S center, -ML surround, both the +S and -ML can be encoded
    """
    # becuase some images have an alpha value, remove the alpha value
    if rgb_image.shape[2] > 3:
        rgb_image = rgb_image[:, :, :3]

    # rgb to lms
    L = np.sum(rgb_image * rgb_to_lms[0], axis=2) 
    M = np.sum(rgb_image * rgb_to_lms[1], axis=2)
    S = np.sum(rgb_image * rgb_to_lms[2], axis=2)

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

    # combine center and surround into one image
    final_lms = alpha_center * center_img + alpha_surround * surround_img
    
    if method == 'lms':

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
        output = avg_center+avg_surround
        # need to normalize, but not in a way that would 
        # example - if i have an image of all yellow, i want the response to be different for each cone type 
        # so I wont normalize to 0-1, but I will normalize to the range of the output
        # so what is the theoretical max and min here? -1 to 1 right? 
        # ok: normalize to make -1 0 and 1 1 
        output = (output + 1)/2
        #output = (output - output.min())/(output.max()-output.min())

        return output




def plot_average_color_rec_field(bipolar_img, subtype, ax=None):
    '''
    takes a bipolar linked image, plots the receptive field of the specified cell type, but each receptive field returns the average color that cell sees
    
    '''
    if ax is None:
        fig, ax = plt.subplots()
    subtype_index = bipolar_img.mosaic.subtype_index_dict[subtype]
    # get the map of mosaic cell: image pixels
    rec_fields = bipolar_img._receptive_field_map
    # remove the cells that are not of the specified subtype
    rec_fields = {cell: pixels for cell, pixels in rec_fields.items() if bipolar_img.mosaic.grid[cell] == subtype_index}
    # get the map that has mosaic cell: average color
    avg_color_map = bipolar_img.avg_colors_cell_map

    # now create pixel:avg_color(s) dict 
    # defaultdict wont return an error but will create a new empty list if the key is not found already for a given pixel
    pixel_to_avg_colors = defaultdict(list)

    for cell, pixels in rec_fields.items():
        avg_color = avg_color_map[cell]
        for pixel in pixels:
            pixel_to_avg_colors[pixel].append(avg_color)

    # now a second dict for pixel: average of the average colors
    pixel_to_final_avg = {
        pixel: np.mean(colors, axis=0) for pixel, colors in pixel_to_avg_colors.items() if pixel[0] < bipolar_img.image.img_shape[0] and pixel[1] < bipolar_img.image.img_shape[1]}
    
    # now plot this 
    img = bipolar_img._compute_subtype_image(bipolar_img.mosaic._index_to_subtype_dict[subtype_index])
    # create empty image (all white) and then fill in the pixels with the average color
    final_img = np.ones((*img.shape[0:2], 3))
    for pixel, avg_color in pixel_to_final_avg.items():
        final_img[pixel[0], pixel[1]] = avg_color
    #plot
    ax.imshow(final_img)

