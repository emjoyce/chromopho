import numpy as np
import os 

from .mosaic import BipolarMosaic
from .bipolar_image import BipolarImageProcessor
from .utils import save_structured_features
import matplotlib.pyplot as plt

def extract_features(bipolar_img, return_labels=True, alpha_white=False, trim_edge_type = True,
                        trim_perc = .95, smooth = True):
    
    '''
    trim perc is the percentage of the m_off x range / 2 to use as new radius 

    '''
    avg_response_dict = bipolar_img.avg_subtype_response_per_pixel

    # Step 1: collect all unique pixel locations
    pixels_seen = sorted(set().union(*[sub_d.keys() for sub_d in avg_response_dict.values()]))
    # trim the edges if trim_edge_type is not None
    if trim_edge_type is True:
        center_x, center_y = np.array(bipolar_img.image.shape[:2]) // 2
        # measure the radius of this image 
        # this should be 'm_off'
        trim_type_pix = np.array(list(set([sub_d.keys() for sub_d in avg_response_dict.values()][0])))
        # get min and max x (top, bottom, left, right of the 'circle', ish)
        min_x, max_x = np.min(trim_type_pix[:,0]), np.max(trim_type_pix[:,0])
        new_rad = (max_x - min_x)//2
        # now only allow 'pixels seen' if they are within that circle
        pixels_seen_arr = np.array(pixels_seen)  
        dists_squared = (pixels_seen_arr[:,0] - center_x)**2 + (pixels_seen_arr[:,1] - center_y)**2

        # Mask for points within the radius
        mask = dists_squared <= (new_rad*trim_perc)**2

        # Apply mask
        pixels_seen = [tuple(px) for px in pixels_seen_arr[mask]]




    n_pixels = len(pixels_seen)
    n_subtypes = len(avg_response_dict)
    features_array = np.zeros((n_pixels, n_subtypes + 2))  # +2 for x and y

    # Optional: handle labels
    if return_labels:
        labels_array = np.zeros((n_pixels, 5))  # x, y, r, g, b
        rgb_img = bipolar_img.image
        if rgb_img.shape[-1] == 4 and alpha_white:
            alpha_mask = rgb_img[..., -1] == 0
            rgb_img = rgb_img.copy()  # avoid modifying original
            rgb_img[alpha_mask, :3] = 1
        rgb_img = rgb_img[..., :3]

    subtypes = sorted(avg_response_dict.keys())
    missing_indices = []  # will store (i, j)

    # Step 2: Fill feature and label arrays
    pixel_index = {pixel: idx for idx, pixel in enumerate(pixels_seen)}  # faster lookup
    for subtype_idx, subtype in enumerate(subtypes):
        subtype_dict = avg_response_dict[subtype]
        for (px, py), value in subtype_dict.items():
            if (px, py) in pixel_index:
                i = pixel_index[(px, py)]
                features_array[i, 0] = px
                features_array[i, 1] = py
                features_array[i, subtype_idx + 2] = value
    # Step 3: Fill labels if needed
    if return_labels:
        for i, (px, py) in enumerate(pixels_seen):
            labels_array[i, 0] = px
            labels_array[i, 1] = py
            labels_array[i, 2:] = rgb_img[px, py]
    if smooth:
        # Handle missing values (-1 fill for pixels not seen by some subtypes)
        missing_mask = (features_array[:, 2:] == 0)  # everything initialized to 0, missing is still 0
        features_array[:, 2:][missing_mask] = -1  # mark missing as -1



        # Impute missing values 
        for j in range(2, features_array.shape[1]):
            missing = np.where(features_array[:, j] == -1)[0]
            for i in missing:
                px, py = features_array[i, 0], features_array[i, 1]
                # Pull out neighbors
                mask = (np.abs(features_array[:, 0] - px) <= 2) & (np.abs(features_array[:, 1] - py) <= 2)
                neighbors = features_array[mask, j]
                neighbors = neighbors[neighbors != -1]
                features_array[i, j] = np.mean(neighbors) if len(neighbors) > 0 else 0

    return (features_array, labels_array) if return_labels else features_array

def extract_features_pipeline(image_dir, features_output_dir, mosaic, analysis_complete_dir, verbose = False):

    '''
    takes an image path, a bipolar mosaic runs through BipolarImageProcessor to get the 
    average response of each cell type at each pixel location in the original image
    stores t

    '''
    if verbose:
        # get the number of .png files
        num_files = len([filename for filename in os.listdir(image_dir) if '.png' in filename])
        print(f'Extracting features from {num_files} images in {image_dir}...')
        file_count = 0 

    features, labels = [], []
    # load the image to p2pimg
    for filename in os.listdir(image_dir):            
        if '.png' in filename:
            if verbose:
                file_count += 1
                # every 5% of files we get through, print the progress
                if file_count % (num_files // 20) == 0:
                    print(f'{(file_count/num_files)*100}% of images processed...')

            img_path = os.path.join(image_dir, filename)
            img = plt.imread(img_path)

            # create a BipolarImageProcessor object
            bipolar_img = BipolarImageProcessor(mosaic, img)
            features_array, labels_array = extract_features(bipolar_img, return_labels = True)

            # save results
            filename_base = img_path.split('/')[-1].split('.')[0]
            save_structured_features(features_array, features_output_dir, filename_base, '_features.npy')
            save_structured_features(labels_array, features_output_dir, filename_base, '_labels.npy')

            # now move the image to the analysis_complete_dir
            os.rename(img_path, os.path.join(analysis_complete_dir, filename_base+'.png'))


            




