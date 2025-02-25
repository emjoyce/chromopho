import matplotlib.pyplot as plt
import numpy as np
import chromopho.mosaic as mosaic

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
        
    
            # get the identity of all indices in circle_indices
            sub_inds = mosaic.grid[*circle_indices]
            subs, counts = np.unique(sub_inds, return_counts = True)
            ind_name_dict = {v:k for k,v in mosaic.subtype_index_dict.items()}
            sub_names = [ind_name_dict[ind] for ind in subs]
            expected_counts = [expected_dict[sub] for sub in subs]
            
            x_axis = np.arange(len(subs))
            width = .2
            ax[i,j].bar(x_axis - width/2, counts, width, label = 'real values')
            ax[i,j].bar(x_axis + width/2, expected_counts, width, label = 'expected values')
            
            ax[i,j].set_xticks(x_axis, sub_names)
            ax[i,j].legend()