import numpy as np
from matplotlib.colors import ListedColormap


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
                alpha_surround (0.8 default) eccentricity, etc. # TODO: maybe delete this and make them all args? do something with eccentricity so can model the whole 
                                                retina or fovea+
        color_filter_params (dict): for bipolar cells, dict with 'center' and 'surround' keys, each with the naem of the cone type(s) that the center and surround 
            of the receptive field are sensitive to, '+l', '-l', '+m', '-m', '+ls', '-ls'
        ratio (float): the ratio of this subtype in the mosaic
        tiling_function (function): a custom function for more compicated tiling than simple ratio
        
        """
        
        self.name = name
        self.ratio = ratio
        self.tiling_function = tiling_function
        self.rf_params = rf_params or {}
        self.rf_size = rf_size
        if self.rf_size < 1:
            raise ValueError(f"rf_size must be between greater than 1 for subtype '{self.name}'")

        self.color_filter_params = color_filter_params

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
    def __init__(self, num_cells, shape = 'rectangle', width = None, height = None, 
                    radius = None, eccentricity = None, subtypes = None):
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
                return radius
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
        

        


        
