import numpy as np
from matplotlib.colors import ListedColormap

class BipolarSubtype:
    """
    Defines properties of a bipolar cell subtype.
    """
    def __init__(self, name, rf_params = None, color_filter = None, ratio = None, 
                        tiling_function = None, rf_size_function = None):
        """
        Parameters:
        name (str): the name of the subtype
        rf_params (dict): a dictionary of parameters for the receptive field of the subtype
            i.e. radius, eccentricity, etc. # TODO: do something with eccentricity so can model the whole 
                                                retina or fovea+
        color_filter (array): an RGB array color filter for the subtype
        ratio (float): the ratio of this subtype in the mosaic
        tiling_function (function): a custom function for more compicated tiling than simple ratio
        rf_size_function (function): a custom (lambda) function for calculating the size of the receptive field e.g. base_radius + ecc_factor*eccentricity
        """
        
        self.name = name
        self.ratio = ratio
        self.tiling_function = tiling_function
        self.rf_params = rf_params or {}
        self.rf_size_function = rf_size_function
        self.color_filter = color_filter

    def get_receptive_field_size(self, rf_params): # TODO: TEST 
        """
        returns the size of the receptive field of the subtype at a given eccentricity
        """
        if self.rf_size_function:
            return self.rf_size_function(eccentricity, **self.rf_params)
        else:
            return rf_params['base_radius']
        

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

        # create grid based on shape params and num_cells
        if self.shape == 'rectangle':
            self.width, self.height = self._best_shape()
            self.grid = np.full((self.width, self.height), 0)
            self._apply_tiling()
        elif self.shape == 'circle':
            self.radius = self._best_shape()
            self.grid = self._generate_circlular_grid()
            self._apply_tiling()
        

        


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
        if remaining_cells > 0: # TODO: should do something smarter here
            print(f"filling {remaining_cells} remaining slots with random subtypes")
            filled_slots_set = set(map(tuple, filled_slots))
            remaining_slots = np.array([inds for inds in available_slots if tuple(inds) not in filled_slots_set]) # TODO: maybe this would pick a subtype with larger ratios? 
                                                                                                                    # maybe dont touch the ones with custom functions
            # fill remaining slots with random subtypes
            random_subtypes = np.random.choice(list(self.subtype_index_dict.values()), remaining_cells)
            self.grid[remaining_slots[:, 0], remaining_slots[:, 1]] = random_subtypes
            filled_slots = np.vstack([filled_slots, remaining_slots])

    
    def plot(self): # TODO add custom coloring for subtypes
        """
        plots the grid
        """

        plt.imshow(self.grid)
        plt.show()
        

        


        
