import numpy as np
import os 

from .mosaic import BipolarMosaic
from .bipolar_image import BipolarImageProcessor
from .utils import save_structured_features
import pulse2percept.stimuli.images
p2pimg = pulse2percept.stimuli.images.ImageStimulus

def extract_features(image_dir, features_output_dir, mosaic, analysis_complete_dir, verbose = False):

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
            img = p2pimg(img_path)

            img_h, img_w, img_c = img.img_shape
            rgb_img = img.data.reshape(img.img_shape)

            # now if the places where alpha == 0 is black, replace with white because we need the contrast to see black logos 
            alpha_mask = rgb_img[..., -1] == 0
            rgb_img[alpha_mask, :3] = 1

            rgb_img = rgb_img[..., :3]

            # create a BipolarImageProcessor object
            bipolar_image = BipolarImageProcessor(mosaic, img)
            # now extract dict with subtype:{pixel:avg_response}
            avg_response_dict = bipolar_image.avg_subtype_response_per_pixel
            # need all 'pixel' entries in all dict 
            pixels_seen = sorted(set([sub_d.keys() for sub_d in avg_response_dict.values()][0]))

            features_array = np.zeros((len(pixels_seen), len(avg_response_dict) + 2)) # plus 2 because pixels need x and y
            labels_array = np.zeros((len(pixels_seen), 5)) # first 2 because pixels need x and y, then r,g,b of image

            # fill in the output array
            # pixel by pixel in pixels_seen
            subtypes = sorted(avg_response_dict.keys())
            # place to keep track of missing avgs
            missing_avgs = []
            for i, (px, py) in enumerate(pixels_seen):
                features_array[i,0], labels_array[i,0] = px, px
                features_array[i,1], labels_array[i,1] = py, py
                for j, subtype in enumerate(subtypes):
                    # if the pixel was seen by the subtype, add the avg respose to that column
                    try:
                        features_array[i,j+2] = avg_response_dict[subtype][(px,py)]
                    except:
                        # this means that subtype did not see this pixel
                        features_array[i,j+2] = -1
                        # add this location to missing_avgs
                        missing_avgs.append([i, j+2, px, py])
                
                # now add the real label to the labels array in the last column
                labels_array[i,-3:] = rgb_img[px,py]
         
            # replace -1s with average of other values around it, not including -1
            
            for (i, j, px, py) in missing_avgs:
                # get the average of the other values at +/-2 in col, +/-2 in row
                # if the value is -1, don't include it in the average
                # if there are no other values, make it 0

                # pull out avg values around the missing value
                mask = (features_array[:, 0] >= px-2) & (features_array[:, 0] <= px+2) & (features_array[:, 1] >= py-2) & (features_array[:, 1] <= py+2)
                values = features_array[mask, j]
                # throw away any that are -1 
                values = values[values != -1]
                # if there are no other values, make it 0
                if len(values) == 0:
                    features_array[i,j] = 0
                else:
                    features_array[i,j] = np.mean(values)

            # save results
            filename_base = img_path.split('/')[-1].split('.')[0]
            save_structured_features(features_array, features_output_dir, filename_base, '_features.npy')
            save_structured_features(labels_array, features_output_dir, filename_base, '_labels.npy')

            # now move the image to the analysis_complete_dir
            os.rename(img_path, os.path.join(analysis_complete_dir, filename_base+'.png'))


            




