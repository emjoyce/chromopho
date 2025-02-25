import numpy as np

class BipolarImageProcessor:
    """
    Takes a bipolar mosaic and an images and processes the image through the mosaic using the 
    color filter parameter in each subtype. the image is assumed to cover the mosaic with fit_option 
    dictating if the image should be fit to the mosaic or if the entire image should be seen by the mosaic. 
    """

    def __init__(self, mosaic, image, fit_option = 'fit_mosaic'):
        """
        Parameters:
        mosaic (BipolarMosaic): a BipolarMosaic object
        image (array): an RGB image array
        fit_option (str): how to fit the image to the mosaic, either 'fit_image' to see the whole image or 'see_entire_image' to see the entire image,
            but some cells might have no pixels in their receptive field   
        """
        self.mosaic = mosaic
        self.image = image
        self.fit_option = fit_option

        # make the receptive field map 


    def _fit_image_and_mosaic_nonoverlapping(self):
        """
        Fits the image and  mosaic in accordance with fit_option 
        returns mapping which has the i,j indices of the mosaic as keys and the pixels in the receptive field of the cell as values
        """
        # we need to ensure that the image is the same size or larger than the mosaic, as each cell cannot have less than one pixel in its receptive field
        img_height, img_width = self.image.img_shape[:2]
        mosaic_height, mosaic_width = self.mosaic.grid.shape[:2]
        if img_height < mosaic_height or img_width < mosaic_width:
            raise ValueError('Image is too small to fit the mosaic')
        
        if self.fit_option == 'fit_mosaic':
            # this will fit the mosaic to the image, so that the entire mosaic is seeing pixels, but maybe not all pixels will be seen by a cell



            # first calculate the nonoverlapping squares that would fit in here
            square_dim = min(img_height // mosaic_height, img_width // mosaic_width)
            # get the dimensions of the mosaic in the space of the image
            img_cutout_dim = square_dim * mosaic_height, square_dim * mosaic_width

            # get center point of image 
            img_centerpt = img_height//2, img_width//2
            # get cutout of image that mosaic cells will 'see'
            # if all dims odd, center is whole number ind, easy to get cutout
            if ((img_cutout_dim[0] % 2 != 0) and (img_cutout_dim[1] % 2 != 0)) and ((mosaic_height % 2 != 0) and (mosaic_width % 2 != 0)):
                img_cutout_i_range = [img_centerpt[0]-img_cutout_dim[0]//2, img_centerpt[0]+img_cutout_dim[0]//2]
                img_cutout_j_range = [img_centerpt[1]-img_cutout_dim[1]//2, img_centerpt[1]+img_cutout_dim[1]//2]
                
            # if one of the dims is even, then define cutout like so
            else:  
                img_cutout_i_range = [(img_centerpt[0]-img_cutout_dim[0]/2)-1, img_centerpt[0]+img_cutout_dim[0]//2]
                img_cutout_j_range = [(img_centerpt[1]-img_cutout_dim[1]/2)-1, img_centerpt[1]+img_cutout_dim[1]//2]

            # now assign squsare of pixels as receptive field of each cell in the mosaic
            mapping = {}     
            available_pixel_inds = [i,j for i in range(img_cutout_i_range[0], img_cutout_i_range[1]) for j in range(img_cutout_j_range[0], img_cutout_j_range[1])]
            for i in range(mosaic_height):
                for j in enumerate(range(mosaic_width)):
                    # the square of pixels will be square_dim x square_dim per i,j element in the mosaic
                    # as we iterate through the row via i, we need to keep track of the pixels that have already been assigned via available_pixel_inds
                    # get the first dim pixels in available_pixel_inds
                    # first get j indices of the first square_dim pixels 
                    j_inds = available_pixel_inds[:square_dim][1]
                    # iterate through the first square_dim i indices and assign the i,j pairs and remove them from available_pixel_inds
                    # first i ind starts at first i in available_pixel_inds
                    first_i = available_pixel_inds[0][0]
                    rec_field = zip(range(first_i, first_i+square_dim), range(j_inds)) 
                    mapping[(i, j)] = rec_field
                    # remove the pixels that have been assigned
                    [available_pixel_inds.remove(rec_field[i]) for i in range(len(rec_field))]
        else: # TODO: do this ^ but for fit_image
            raise ValueError('functionality for fit_image not yet implemented')
        return mapping
            




    def _calculate_min_rec_field_radius(self):
        """
        Calculates the minimal radius of the receptive field to tile the image with the mosaic
        """
        img_height, img_width = self.image.img_shape[:2]
        # if image is 1:1 with mosaic, then the minimal radius is 0, just the pixel right in front
        # if the image is less than 1:1 and has fewer pixels than cells, return error
        # if the image is more than 1:1 with image and there are more pixels than cells, then what MINIMUM radius of 
            # pixels per circular receptive field of the cell would cover every pixel in the image?
        # depends on fit_option 





    def _compute_receptive_field_map(self):
        """
        Assigns receptive fields to the bipolar cells in the mosaic, stored in self._receptive_field_map
        """
        # receptive field will be a mapping of each bipolar cell to the pixels that it has in its receptive field
        self._receptive_field_map = np.zeros_like(self.mosaic.grid)
        for i in range(self.mosaic.grid.shape[0]):
            for j in range(self.mosaic.grid.shape[1]):
                # (i,j) will be the key to the dict
                # get the receptive field size of the subtype at the current location
                rf_size = self.mosaic.get_receptive_field_size(i, j)
                # get the pixels in the receptive field





    def process_image(self):
        """
        Processes the image through the mosaic using the color filter and receptive field parameters
        of each subtype
        """
        # create an empty image array to store the processed image
        processed_image = np.zeros_like(self.image)
        for i in range(self.mosaic.grid.shape[0]):
            for j in range(self.mosaic.grid.shape[1]):
                # get the color filter of the subtype at the current location
                color_filter = self.mosaic.get_color_filter(i, j)
                # get the receptive field size of the subtype at the current location
                rf_size = self.mosaic.get_receptive_field_size(i, j)
                # apply the color filter to the image using the receptive field size
                print(f'color_filter: {color_filter}, rf_size: {rf_size}')
                # TODO: will put a function here that has a map of cell to pixels it responds to, then will process those pixels
                #processed_image[i, j] = color_filter * rf_size



