import numpy as np
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from .utils import make_surround_center_bins



class BipolarSubtype:
    """
    Defines properties of a bipolar cell subtype.
    """
    def __init__(self, name, rf_size, rf_params = None, color_filter_params = None, ratio = None, 
                        tiling_function = None):
        """
        Parameters:
        name (str): the name of the subtype
        rf_size (float, function): a float representing the factor to scale up the minimmum receptive field.
                    if set to 1, the receptive field will be the minimum amount of overlap possible, where every cell has a 
                    circular receptive field that is the minimum diameter so that every pixel mapped under the mosaic is 
                    seen by at least 1 cell. set to 2, the circular receptive field will have twice as large as that minimum radius, etc.
                    or custom (lambda) function for calculating the size of the receptive field scaling metric that outputs same units as described above
        rf_params (dict): a dictionary of parameters for the receptive field of the subtype
            i.e. center_sigma (1 by default), surround_sigma (3.0 default), alpha_center (1.0 default),
                alpha_surround (None by default) or family ('midget' / 'diffuse') for surround alpha sampling,
                eccentricity, etc. # TODO: maybe delete this and make them all args? do something with eccentricity so can model the whole 
                                                retina or fovea+
        color_filter_params (dict): for bipolar cells, dict with 'center' and 'surround' keys, each with the naem of the cone type(s) that the center and surround 
            of the receptive field are sensitive to, '+l', '-l', '+m', '-m', '+ls', '-ls'
        ratio (float): the ratio of this subtype in the mosaic
        tiling_function (function): a custom function for more compicated tiling than simple ratio
        
        """
        
        self.name = name
        self.ratio = ratio
        self.tiling_function = tiling_function
        self.rf_size = rf_size
        if self.rf_size < 1:
            raise ValueError(f"rf_size must be between greater than 1 for subtype '{self.name}'")

        self.color_filter_params = color_filter_params
        _defaults = {
            'family': None,
            'surround_alpha_n_bins': 5,
            'alpha_center': 1.0,
            'alpha_surround': None,
            'apply_rectification': True,
            'on_k': 0.7,
            'on_n': 2.5,
            'off_k': 0.7,
            'off_n': 4.0
        }

        # copy so we do not mutate whatever dict was passed in from a script
        params = dict(rf_params or {})
        for key, val in _defaults.items():
            params.setdefault(key, val)

        # store the surround alpha setup on the subtype itself.
        # if family is passed in, alpha_surround should become the full set of
        # possible surround alpha values for that family, and
        # alpha_surround_probs should hold the matching sampling weights.
        # if no family is passed in, we expect one scalar alpha_surround and
        # store that as a length-1 array with probability 1.
        self.family = params['family']

        if self.family is not None:
            self.alpha_surround, self.alpha_surround_probs = make_surround_center_bins(
                self.family,
                n_bins=params['surround_alpha_n_bins'],
            )
        elif params['alpha_surround'] is not None:
            self.alpha_surround = np.array([params['alpha_surround']], dtype=float)
            self.alpha_surround_probs = np.array([1.0], dtype=float)
        else:
            raise ValueError(
                f"Subtype '{self.name}' must define either family or alpha_surround in rf_params"
            )

        # keep aliases with the more explicit names too.
        # these are useful when we want to be very explicit that the subtype is
        # carrying a bank of possible surround alpha values instead of one
        # scalar value.
        self.surround_alpha_values = self.alpha_surround
        self.surround_alpha_probs = self.alpha_surround_probs

        # the current tf processor path still needs one scalar surround alpha
        # until the per-cell weighted sampling step is finished.
        # keep that scalar on a separate attribute so alpha_surround itself can
        # stay as the full value bank.
        self.alpha_surround_mean = float(np.sum(
            self.alpha_surround * self.alpha_surround_probs
        ))

        self.rf_params = params
        

    # def get_receptive_field_size(self, rf_params, eccentricity = None): # TODO: TEST 
    #     """
    #     returns the size of the receptive field of the subtype at a given eccentricity
    #     """
    #     if callable(self.rf_size):
    #         if not eccentricity:
    #             raise ValueError(f"eccentricity must be defined to get receptive field size for subtype '{self.name}' as the rf_size for this subtype is a function of eccentricity")
    #         return self.rf_size(eccentricity, **self.rf_params)
    #     else:
    #         try: 
    #             return rf_size
    #         except KeyError: raise KeyError(f"base_radius must be defined to get receptive field size. not defined for subtype '{self.name}'")
        

class BipolarMosaic:
    """
    Creates a mosaic of bipolar cells with different subtypes.
    """
    def __init__(self, num_cells, shape = 'circle', width = None, height = None, 
                    radius = None, eccentricity = None, subtypes = None, density_swap = True):
        """
        Parameters:
        num_cells (int): approximate number of bipolar cells in the mosaic. note that this number will be 
                rounded to fit the grid shape
        shape (str, array): the shape of the mosaic , 'rectangle', 'circle' or enter your custom grid here(default 'rectangle')
        width (float): Optional, the width of a rectangular mosaic (default None)
        height (float): Optional, the height of a rectangular mosaic (default None)
        radius (float): Optional, the radius of a circular mosaic (default None)
        eccentricity (float): Optional, defines the subtype tiling automatically based on 
                    the eccentricity of the mosaic (default None)
        subtypes (list): Optional, a list of BipolarSubtype objects to be used in the mosaic
        """
        
        self.subtypes = subtypes
        self.eccentricity = eccentricity
        self.shape = shape
        self.width = width
        self.height = height
        self.radius = radius
        self.num_cells = num_cells
        self.subtypes = subtypes if subtypes else []
        self.subtype_index_dict = {st.name: i+1 for i, st in enumerate(self.subtypes)}
        self.subtype_index_dict['none'] = -1
        self._index_to_subtype_dict = {self.subtype_index_dict[subtype.name]:subtype for subtype in self.subtypes}
        self._index_to_subtype_dict[-1] = None

        # create grid based on shape params and num_cells
        if self.shape == 'rectangle':
            self.width, self.height = self._best_shape()
            self.grid = np.full((self.width, self.height), 0)
            self._apply_tiling()
        elif self.shape == 'circle':
            self.radius = self._best_shape()
            self.grid = self._generate_circlular_grid()
            self._apply_tiling()
            if density_swap:
                print('density swap')
                self.grid = self._density_swap()

        # once the grid is fully defined, assign each valid cell one surround
        # alpha bank index based on that cell's subtype. these assignments are
        # part of the mosaic itself, and can then be reused by any processor
        # built from this mosaic.
        self._assign_surround_alpha_bins_to_cells()
        self._generate_receptive_field_matrix()
        

    def _best_shape(self):
        """
        determines the best dimensions for the mosaic based on the number of cells and shape
        """
        if self.shape == 'rectangle':
            if self.width and self.height:
                return width, height
            else:
                # calculate the best width and height for rectangle
                width = int(np.sqrt(self.num_cells))
                height = int(self.num_cells / width)
                return width, height
        elif self.shape == 'circle':
            if self.radius:
                return self.radius
            else:
                # calculate the best radius for circle
                return int(np.round(np.sqrt(self.num_cells / np.pi)))
    
    def _generate_circlular_grid(self):
        """
        generates a circular mask for cell placement
        """
        d = 2 * self.radius
        grid = np.full((d, d), -1)
        center = (self.radius, self.radius)
        cell_count = 0
        for i in range(d): # iterate through rows
            for j in range(d): # iterate through columns
                if np.sqrt((i - center[0])**2 + (j - center[1])**2) <= self.radius: # check if point is within circle
                    grid[i, j] = 0 
                    cell_count += 1
        return grid

    def _assign_surround_alpha_bins_to_cells(self):
        """
        assigns one surround alpha bank index to each valid cell in the mosaic

        stores the bank index, the actual surround alpha value, and the cell's
        surround alpha family both as flat arrays aligned with valid_cell_coords
        and as full mosaic-shaped grids with -1 / '' in invalid cells.

        grid_surround_alphas is the simple mosaic-shaped float grid version of
        the sampled surround alpha values. for the default large circular
        mosaic, this is the full original circle grid shape.
        """
        # get the valid cell coords once from the final mosaic grid.
        # these are the coords we will align all per-cell surround alpha arrays to.
        valid_cell_coords = np.argwhere(self.grid != -1).astype(int)
        cell_subtype_indices = self.grid[valid_cell_coords[:, 0], valid_cell_coords[:, 1]]

        n_valid_cells = valid_cell_coords.shape[0]
        cell_surround_alpha_bank_indices = np.full(n_valid_cells, -1, dtype=np.int32)
        cell_surround_alpha_values = np.full(n_valid_cells, -1.0, dtype=float)
        cell_surround_alpha_families = np.full(n_valid_cells, '', dtype=object)

        # sample surround alpha bins one subtype at a time.
        # this keeps the code vectorized over all cells of a subtype instead of
        # looping through cells one-by-one.
        for subtype in self.subtypes:
            subtype_index = self.subtype_index_dict[subtype.name]
            subtype_mask = (cell_subtype_indices == subtype_index)
            n_subtype_cells = int(np.sum(subtype_mask))
            if n_subtype_cells == 0:
                continue

            subtype_alpha_bank = np.asarray(subtype.alpha_surround, dtype=float)
            subtype_alpha_probs = np.asarray(subtype.alpha_surround_probs, dtype=float)
            subtype_bank_indices = np.arange(subtype_alpha_bank.shape[0], dtype=np.int32)

            sampled_bank_indices = np.random.choice(
                subtype_bank_indices,
                size=n_subtype_cells,
                replace=True,
                p=subtype_alpha_probs,
            )

            cell_surround_alpha_bank_indices[subtype_mask] = sampled_bank_indices
            cell_surround_alpha_values[subtype_mask] = subtype_alpha_bank[sampled_bank_indices]
            cell_surround_alpha_families[subtype_mask] = (
                subtype.family if subtype.family is not None else 'scalar'
            )

        # store the per-cell surround alpha assignments aligned with
        # valid_cell_coords. any processor that builds its valid coords from the
        # same mosaic mask can reuse these directly.
        self.valid_cell_coords = valid_cell_coords
        self.cell_surround_alpha_bank_indices = cell_surround_alpha_bank_indices
        self.cell_surround_alpha_values = cell_surround_alpha_values
        self.cell_surround_alpha_families = cell_surround_alpha_families

        # also scatter the assignments back onto full mosaic-shaped grids.
        # this makes direct lookup by i,j easy when needed.
        h, w = self.grid.shape
        self.surround_alpha_bank_index_grid = np.full((h, w), -1, dtype=np.int32)

        # keep the actual sampled surround alpha values in a mosaic-shaped
        # float grid too. this is the same simple scatter logic as grid_outputs
        # in the tf processor, but it lives on the mosaic and is built once.
        self.grid_surround_alphas = np.full((h, w), -1.0, dtype=float)

        # keep the older explicit attribute name as an alias to the same grid
        # so existing code can keep working.
        self.surround_alpha_value_grid = self.grid_surround_alphas
        self.surround_alpha_family_grid = np.full((h, w), '', dtype=object)

        self.surround_alpha_bank_index_grid[valid_cell_coords[:, 0], valid_cell_coords[:, 1]] = cell_surround_alpha_bank_indices
        self.grid_surround_alphas[valid_cell_coords[:, 0], valid_cell_coords[:, 1]] = cell_surround_alpha_values
        self.surround_alpha_family_grid[valid_cell_coords[:, 0], valid_cell_coords[:, 1]] = cell_surround_alpha_families
    
    def _apply_tiling(self):
        """
        applies the tiling ratio/functions to the grid
        """
        # get available places
        available_slots = np.argwhere(self.grid == 0)
        filled_slots = np.array([[-1,-1]]) 

        # fill subtypes with custom tiling functions first
        for subtype in self.subtypes:
            if subtype.tiling_function:
                filled_slots = subtype.tiling_function(self.grid, available_slots, filled_slots) # todo: how would a custom func work? just an array of cells for that subtype?

        
        for subtype in self.subtypes:
            # get remaining slots that are still open to be filled and shuffle them 
            filled_slots_set = set(map(tuple, filled_slots))
            remaining_slots = np.array([inds for inds in available_slots if tuple(inds) not in filled_slots_set])
            np.random.shuffle(remaining_slots)
            if subtype.ratio:
                # calculate the number of slots to fill with this subtype
                num_to_place = int(subtype.ratio * len(available_slots))
                # fill them - this works because 
                subtype_slots = remaining_slots[:num_to_place]
                # for each subtype slot, fill with subtype index
                self.grid[subtype_slots[:, 0], subtype_slots[:, 1]] = self.subtype_index_dict[subtype.name]
                #self.grid[subtype_slots] = self.subtype_index_dict[subtype.name]
                filled_slots = np.vstack([filled_slots, subtype_slots])

        remaining_cells = np.argwhere(self.grid == 0).shape[0]
        if remaining_cells > 0: # TODO: could do something smarter here?
            #print(f"filling {remaining_cells} remaining slots with random subtypes")
            filled_slots_set = set(map(tuple, filled_slots))
            remaining_slots = np.array([inds for inds in available_slots if tuple(inds) not in filled_slots_set]) # TODO: maybe this would pick a subtype with larger ratios? 
                                                                                                                    # maybe dont touch the ones with custom functions
            # fill remaining slots with random subtypes
            # dont use -1 or that will fill some places with no subtype - will stay empty
            random_subtypes = np.random.choice([ind for ind in self._index_to_subtype_dict.keys() if ind != -1], remaining_cells)
            self.grid[remaining_slots[:, 0], remaining_slots[:, 1]] = random_subtypes
            filled_slots = np.vstack([filled_slots, remaining_slots])

    # optimization function for spreading out each subtype more evenly within mosaic
    def _density_swap(bipolar_mosaic, percent_swapped = .07, sigma = .88, max_iter = 1000, track_obj = False, objective_by_type = False,
                                window = 150, epsilon = .005, center_s_sigma=12, s_perc = .04):
        #######################################################################
        ### helper funcs:
        def get_extreme_density_coords(smoothed, n_pairs):
            # finds maching number of high density and low density points 
            
            valid_mask = smoothed >= 0
            flat_smoothed = smoothed[valid_mask]
            valid_coords = np.argwhere(valid_mask)
        
            # get top and bottom n_pairs indicies
            top_idx_unsorted = np.argpartition(flat_smoothed, -n_pairs)[-n_pairs:]
            top_idx = top_idx_unsorted[np.argsort(flat_smoothed[top_idx_unsorted])[::-1]]
            bottom_idx_unsorted = np.argpartition(flat_smoothed, n_pairs)[:n_pairs]
            bottom_idx = bottom_idx_unsorted[np.argsort(flat_smoothed[bottom_idx_unsorted])]
        
            # convert to coords
            top_coords = valid_coords[top_idx]
            bottom_coords = valid_coords[bottom_idx]
        
            # filter out neighbors
            top_coords = filter_out_neighbors(top_coords)
            bottom_coords = filter_out_neighbors(bottom_coords)
        
            # now, make sure to return the same number of elements so they can be paired up for swapping later
            n_elements = min(len(top_coords), len(bottom_coords))
            
            return top_coords[:n_elements], bottom_coords[:n_elements]
        
        def filter_out_neighbors(coords):
            # helper function for get_extreme_density_coords, where we will make sure our candidates are not neighbors before swapping
            # this prevents moving too many cells to a low density area, which would just create another high density area
            kept_coords = []
            excluded = set()
        
            for coord in coords:
                c = tuple(coord)
                if c in excluded:
                    continue
                kept_coords.append(c)
                # add neighbors to excluded
                x, y = c
                excluded.update([c, (x+1, y), (x-1, y), (x, y+1), (x, y-1), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)])
        
            return np.array(kept_coords)
        
        
        # def get_sigma_it(it, max_iter, sigma_start, floor = .5, p = 2):
        #     progress = it / max_iter
        #     decay = floor + (1-floor)*(1 - progress**p)
        #     return sigma_start * decay
        
        def test_stop(objective_history, window = 75, epsilon = .001):
            # checks if the objective is still making reasonable progress compared to the recent history 
        
            # dont stop if we don't have enough history collected
            if len(objective_history) <  window:
                return False
            recent_avg = np.mean(objective_history[-window:])
            current_val = objective_history[-1]
            return abs(current_val - recent_avg) < epsilon
            
        # def get_smoothed_density(bipolar_grid, target_subtype, sigma = 2):
        
        #     # get a mask of valid cells 
        #     nonviable_mask = bipolar_grid == -1
        #     # mask of cells of the subtype 
        #     subtype_mask = (bipolar_grid == target_subtype).astype(float)
        
        #     smoothed = gaussian_filter(subtype_mask, sigma = sigma)
        
        #     smoothed[nonviable_mask] = -1
        #     # also return the number of cells in that subtype for later on, when we use this function. 
        #     # prevents us from calculating this value twice 
        #     return smoothed, np.sum(subtype_mask)
        # updated to handle S scotoma
        def get_smoothed_density(bipolar_grid, target_subtype, sigma = 2, s_subtypes = [5,6], center_s_sigma=center_s_sigma, 
                                s_perc = s_perc):
        
            # get a mask of valid cells 
            nonviable_mask = bipolar_grid == -1
            # mask of cells of the subtype 
            subtype_mask = (bipolar_grid == target_subtype).astype(float)
        
            smoothed = gaussian_filter(subtype_mask, sigma = sigma)
        
            # also return the number of cells in that subtype for later on, when we use this function. 
            # prevents us from calculating this value twice 
            # add a gaussian at the center
            if target_subtype in s_subtypes:
                
                # then this cell is an s connected bipolar, and we should avoid the center a bit
                center_i, center_j = int(bipolar_grid.shape[0]/2), int(bipolar_grid.shape[1]/2)
            
        
                # calculate the number of cells to place 
                n_pts = int(np.sum(subtype_mask)*s_perc)
                
                # sample rows and columns of the mosaic according to a gaussian around the center s scotoma 
                sampled_rows = np.random.normal(loc = center_i, scale = center_s_sigma, size = n_pts).astype(int)
                sampled_cols = np.random.normal(loc = center_j, scale = center_s_sigma, size = n_pts).astype(int)
        
                # throw out anything outside of viable cells 
                rc_mask = (sampled_rows<bipolar_grid.shape[0]) & (sampled_cols<bipolar_grid.shape[1])
                sampled_rows = sampled_rows[rc_mask]
                sampled_cols = sampled_cols[rc_mask]
                sampled_viable = ~nonviable_mask[sampled_rows, sampled_cols]
                subtype_mask[sampled_rows[sampled_viable], sampled_cols[sampled_viable]] = 1.0
            
            smoothed = gaussian_filter(subtype_mask, sigma = sigma)
            smoothed[nonviable_mask] = -1
            
            return smoothed, np.sum(subtype_mask)
            
        def total_objective(grid, per_subtype = False):
            subtypes = np.unique(grid)
            # exclude -1, which indicates a blank cell
            subtypes = subtypes[subtypes > 0]
        
            total_objective = 0
        
            if per_subtype:
                obj_list = []
            for subtype in subtypes:
                coords = np.argwhere(grid == subtype)
                tree = cKDTree(coords)
                # k = 2 to skip self
                dists, _ = tree.query(coords, k = 2)
                nearest_dists = dists[:,1]
                if per_subtype:
                    obj_list.append(np.sum(nearest_dists)/len(np.argwhere(grid == subtype)))
                else:
                    # add average value per cell of the given subtype
                    total_objective += np.sum(nearest_dists)/len(np.argwhere(grid == subtype))
            if per_subtype:
                return obj_list
            return total_objective/len(subtypes)
        ### end density swap helper funcs
        #######################################################################

        
        # save sigma start for shrinking the sigma later
        sigma_start = sigma
        grid = bipolar_mosaic.grid.copy()
        h, w = grid.shape
        subtypes = np.unique(grid)
        subtypes = subtypes[(subtypes > 0)]
        if track_obj:
            obj_history = []

        for it in range(max_iter):
            for subtype in subtypes:
                smoothed, n_subtype = get_smoothed_density(grid, subtype, sigma = sigma)

                # get the top n% of cells of this subtype
                n_pairs = int(n_subtype*percent_swapped)

                # pull out coords that are at highest density points and lowest density points 
                top_coords, bottom_coords = get_extreme_density_coords(smoothed, n_pairs)
                # now, swap these coords in the grid
                for (x1, y1), (x2, y2) in zip(top_coords, bottom_coords):
                    grid[x1, y1], grid[x2, y2] = grid[x2, y2], grid[x1, y1]
            if track_obj:
                obj_history.append(total_objective(grid, per_subtype = objective_by_type))
                # check stopping criteria 
                if (it+1) % window == 0:
                    if test_stop(obj_history, window = window, epsilon = epsilon):
                        print(f'stopping after {it} iterations')
                        break
                            
        if track_obj:
            return grid, obj_history
        return grid

    def _generate_receptive_field_matrix(self):
        """
        generates a matrix of receptive field sizes for each cell in the grid
        """
        self.receptive_field_matrix = np.zeros_like(self.grid, dtype=float)
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                self.receptive_field_matrix[i,j] = self.get_receptive_field_size(i, j)

    def get_color_filter_params(self, row, col):
        """
        returns the color filter of the cell at the given location
        parameters:
        row (int): the row index of the grid to pull the color filter from 
        col (int): the column index of the grid to pull the color filter from 
        """
        subtype_index = self.grid[row, col]
        subtype = self._index_to_subtype_dict[subtype_index]
        # if it contains a function:
        if callable(subtype.color_filter_params):
            color_filter_params = subtype.color_filter_params(row, col) # TODO: maybe this should have a helper func -i dont want people to have to define
                                                            # color filter function based on i,j, by eccentricity would be nicer 
        else:
            color_filter_params = subtype.color_filter_params
        return color_filter_params

    def get_receptive_field_size(self, row, col, eccentricity = None):
        """
        returns the size of the receptive field of the cell at the given location
        parameters:
        row (int): the row index of the grid to pull the receptive field size from 
        col (int): the column index of the grid to pull the receptive field size from 
        eccentricity (float): the eccentricity of the cell. Only needed if this cell's subtype has a custom rf_size function that takes eccentricity as a parameter
        """
        subtype_index = self.grid[row, col]
        if subtype_index == -1:
            return 0
        # get the subtype 
        subtype = self._index_to_subtype_dict[subtype_index]
        # if the rec field function is defined, use it
        if callable(subtype.rf_size):
            return subtype.rf_size(eccentricity, **subtype.rf_params)
        else:
            return subtype.rf_size

    def get_subtype(self, row, col):
        """
        returns the subtype of the cell at the given location
        parameters:
        row (int): the row index of the grid to pull the subtype from 
        col (int): the column index of the grid to pull the subtype from 
        """
        return self._index_to_subtype_dict[self.grid[row, col]]

    
    def plot(self): # TODO add custom coloring for subtypes
        """
        plots the grid
        """

        plt.imshow(self.grid)
        plt.show()
        

        


        
