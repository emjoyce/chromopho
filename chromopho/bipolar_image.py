import numpy as np
from .utils import img_to_rgb, _parse_cone_string
from .plot import bipolar_image_filter_rgb
import concurrent.futures
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

class BipolarImageProcessor:
    """
    Takes a bipolar mosaic and an images and processes the image through the mosaic using the 
    color filter parameter in each subtype. the image is assumed to cover the mosaic with fit_option 
    dictating if the image should be fit to the mosaic or if the entire image should be seen by the mosaic. 
    """

    def __init__(self, mosaic, image, fit_option = 'fit_mosaic', return_minimum_rf = False, method = 'greyscale', stimulation_mosaic = None):
        """
        Parameters:
        mosaic (BipolarMosaic): a BipolarMosaic object
        image (array): pulse2percept image object
        fit_option (str): how to fit the image to the mosaic, either 'fit_image' to see the whole image but some cells may see no pixels 
            or 'fit_mosaic' so that each cell in the mosaic sees some pixels,but some pixels may not be seen by any cell 
        method (str): the method to use to compute the average color of the pixels in the receptive field of a cell. avg color could be rgb 
            (lifo coming in from cones) or grayscale (info going out from bipolar cells)
        stimulation_mosaic (array): an array same size as mosaic that holds which cells are stimulated by an electrode and to what intensity level
        """
        self.mosaic = mosaic
        self.image = image
        self.stimulation_mosaic = stimulation_mosaic
        self.fit_option = fit_option
        self.bipolar_images = {}
        self._fit_image_and_mosaic(return_minimum_rf)
        self.get_all_average_colors(method = method)
        self.avg_subtype_response_per_pixel = {}
        self.get_avg_color_map_per_pixel()

        # make the receptive field map 


    def _fit_image_and_mosaic(self, return_minimum = False):
        """
        Fits the image and  mosaic in accordance with fit_option 
        returns mapping which has the i,j indices of the mosaic as keys and the pixels in the receptive field of the cell as values
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
        self._receptive_field_map = mapping










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
    
    def _compute_subtype_image(self, subtype, method = 'lms', default_value = [0.5,0.5,0.5], rgb_to_lms = np.array([[0.313, 0.639, 0.048],  
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
            bipolar_image_seen = bipolar_image_filter_rgb(
                        rgb_image = self.image,
                        center_cones = color_filter_dict['center'],
                        surround_cones = color_filter_dict['surround'],
                        center_sigma = rf_params['center_sigma'],
                        surround_sigma = rf_params['surround_sigma'],
                        alpha_center = rf_params['alpha_center'],
                        alpha_surround = rf_params['alpha_surround'],
                        rgb_to_lms = rgb_to_lms,   # S 
                        default_value = default_value, 
                        method = method)
                # so the image should output a single value dependign on subtype, which if s+, would be the output of l cones minue the output of m+l

            self.bipolar_images[subtype.name] = bipolar_image_seen
        return bipolar_image_seen
    
    # TODO: add a calculation of the center and surround activity of lms based on the receptive field of each bipolar cell 
    # that will be when use_ideal = False
    def get_average_color_of_cell(self, i, j, method = 'lms', use_ideal = True):
        """
        Returns the average color of the pixels in the receptive field of the cell at i,j location of cell mosaic 
        """
        # get the pixels in the receptive field of the cell
        rec_field = np.array(self.get_receptive_field_of_cell(i, j))
        rows, cols = rec_field[:, 0], rec_field[:, 1]
        # get the subtype of the cell
        subtype = self.mosaic._index_to_subtype_dict[self.mosaic.grid[i,j]]
        # put the image through the color filter of the subtype
        bipolar_image_seen = self._compute_subtype_image(subtype, method = method)
        # get the average color of the pixels in the receptive field
        try:
            avg_color = np.mean(bipolar_image_seen[rows, cols], axis = 0)
        except:
            # remove any row, col pair that is out of bounds
            # find the inds of the out of bounds rows and cols
            # TODO figure out why this happens twice per image!! 
            out_of_bounds_inds = [i for i in range(len(rows)) if rows[i] >= bipolar_image_seen.shape[0] or cols[i] >= bipolar_image_seen.shape[1]]
            # remove these inds from rows and cols
            rows = np.delete(rows, out_of_bounds_inds)
            cols = np.delete(cols, out_of_bounds_inds)
            print('had to delete something')
            avg_color = np.mean(bipolar_image_seen[rows, cols], axis = 0)
        return avg_color
    
    # TODO: should rename this to something like bipolar cell output, but currently it can also be used to 
    # output the color info SEEN by each bipolar subtype... need to think about this 
    def get_all_average_colors(self, method = 'lms'):
        """
        Returns a dictionary of the average color of the pixels in the receptive field of each cell in the mosaic
        if stimulation mosaic is provided, bipass normal color filter stuff and set each cell's
        avg color to the value in the stimulation mosaic
        """
        stimulation_mosaic = self.stimulation_mosaic
        avg_colors_cell_map = {}
        for i in range(self.mosaic.grid.shape[0]):
            for j in range(self.mosaic.grid.shape[1]):
                # only continue if this i,j belongs to an actual cell in mosaic
                if self.mosaic.grid[i,j] == -1:
                    continue
                if stimulation_mosaic is None:
                    avg_colors_cell_map[(i, j)] = self.get_average_color_of_cell(i, j, method = method)
                else:
                    stim_val = stimulation_mosaic[i, j]
                    avg_colors_cell_map[(i, j)] = np.array([stim_val])
                    
        self.avg_colors_cell_map = avg_colors_cell_map
    
    def get_avg_color_map_per_pixel(self):
        """
        Returns the average color of the pixels in the receptive field of each cell in the mosaic
        """
        for subtype in self.mosaic.subtypes:
            subtype_index = self.mosaic.subtype_index_dict[subtype.name]
            # get the map of mosaic cell: image pixels
            # remove the cells that are not of the specified subtype
            rec_fields = {cell: pixels for cell, pixels in self._receptive_field_map.items() 
                                if self.mosaic.grid[cell] == subtype_index}
            # get the map that has mosaic cell: average color
            avg_color_map = self.avg_colors_cell_map

            # now create pixel:avg_color(s) dict 
            # defaultdict wont return an error but will create a new empty list if the key is not found already for a given pixel
            pixel_to_avg_colors = defaultdict(list)
            
            # helper function for parallel computation of colors in get_avg_color_map_per_pixel
            def _gather_pixel_avgcolor_pairs(cell_and_pixels):
                cell, pixels = cell_and_pixels
                avg_color = avg_color_map[cell]
                results = []
                for pixel in pixels:
                    if pixel[0] < self.image.shape[0] and pixel[1] < self.image.shape[1]:
                        results.append((pixel, avg_color))
                return results

            # parallel compute the pixel:avg_color pairs
            with ThreadPoolExecutor() as executor:
                for results in executor.map(_gather_pixel_avgcolor_pairs, rec_fields.items()):
                    for pixel, avg_color in results:
                        pixel_to_avg_colors[pixel].append(avg_color)

            # now a second dict for pixel: average of the average colors
            pixel_to_final_avg = {
                pixel: np.mean(colors, axis=0) for pixel, colors in pixel_to_avg_colors.items()}
            
            
            self.avg_subtype_response_per_pixel[subtype.name] = pixel_to_final_avg



    
























            












