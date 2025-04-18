import numpy as np
import os
import joblib
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim
from sklearn.linear_model import Ridge, RidgeCV
from .feature_extraction import extract_features
from .bipolar_image import BipolarImageProcessor
from sklearn.model_selection import KFold


def read_XY(filename_base, features_dir):
    '''
    reads in the features and labels for a given filename_base
    '''
    features = np.load(os.path.join(features_dir, filename_base+'_features.npy'))
    labels = np.load(os.path.join(features_dir, filename_base+'_labels.npy'))
    return features, labels

def train_model(X, Y):
    '''
    trains a model on the given features and labels
    '''
    model = Ridge(alpha= 1.0)
    # remove the pixel index from X and Y
    X = X[:,2:]
    Y = Y[:,2:]
    model.fit(X, Y)
    return model

def train_model_cv(X_train, y_train, alphas=None, k=5):


    if alphas is None:
        alphas = [0.1, 1.0, 10.0, 100.0] 

    # Remove pixel indices 
    X = X_train[:, 2:]
    Y = y_train[:, 2:]

    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    model = RidgeCV(alphas=alphas, cv=kfold)

    model.fit(X, Y)

    print(f"Best alpha selected: {model.alpha_}")

    return model

def evaluate_ssim(y, y_pred):
    return ssim(y, y_pred, channel_axis = -1, data_range = 1)

def generate_img_from_feats(feats, model, img_h, img_w, output_groundtruth = None):
    '''
    generates an image from the predicted values
    '''
    y_pred = model.predict(feats[:,2:])
    y_pred = np.clip(y_pred, 0.0, 1.0)

    # make an image from the predicted values and an image for the labels
    y_pred_img = np.zeros((img_h, img_w, 3)) # output will be r,g,b
    if output_groundtruth is not None:
        y_img = np.zeros((img_h, img_w, 3))
    for i, (x, y, t1,t2,t3,t4,t5,t6) in enumerate(feats):

        # I need to take the value at y_pred at a pixel index and put it in y_pred_img 
        # the indices of y_pred are the same as the indices of Xs_test so I should just be able to match via i
        # but that is not working 

        y_pred_img[int(x), int(y)] = y_pred[i]
        if output_groundtruth is not None:
            y_img[int(x), int(y)] = output_groundtruth[i,2:]
    
    if output_groundtruth is not None:
        return y_pred_img, y_img
    else:
        return y_pred_img



def evaluate_model_image(model, Xs_test, Ys_test, img_h, img_w, return_sim = False):
    '''
    evaluates a model on the given features and labels based on ssim
    '''
    
    y_pred_img, y_img = generate_img_from_feats(Xs_test, model, img_h, img_w, output_groundtruth = Ys_test)

    
    # now get the ssim score
    # TODO : this is not working, need to debug
    if return_sim:
        ssim_score = evaluate_ssim(y_img, y_pred_img)
        #print(f'SSIM: {ssim_score}')
        return y_pred_img, y_img, ssim_score
    else:
        return y_pred_img, y_img

def phosphene_simulation(radius, mosaic, model, dummy_img, stim_response = 1, seed = None, alpha_white = False):
    '''
    simulates a phosphene with a given radius
    '''

    if seed is not None:
        np.random.seed(seed)
    
    # create a blank map for cell responses 
    
    img_h, img_w = dummy_img.shape[0], dummy_img.shape[1]
    mosaic_h, mosaic_w = mosaic.grid.shape[0], mosaic.grid.shape[1]
    bipolar_cell_responses = np.zeros((mosaic_h, mosaic_w))
    # get the center of the image
    center_x, center_y = mosaic_h//2, mosaic_w//2

    # randomly shift circle_indives by a random amount that is constrained by the size of the mosaic
    center_phosphene_x = center_x + np.random.randint(0, mosaic_h/2.1)
    center_phosphene_y = center_y + np.random.randint(0, mosaic_w/2.1)
    print(center_phosphene_x, center_phosphene_y)

    # get the indices of the cells that are within the radius
    for i in range(mosaic_h):
        for j in range(mosaic_w):
            if (i-center_phosphene_x)**2 + (j-center_phosphene_y)**2 <= radius**2:
                bipolar_cell_responses[i,j] = stim_response
    
    # now create a BipolarImageProcessor object
    bip = BipolarImageProcessor(mosaic, dummy_img, stimulation_mosaic = bipolar_cell_responses)

    # now run through model to get y_pred
    # need an array with pixel_x, pixel_y, subtype_response vector for each subtype in the order of ordered_subtypes
    avg_response_dict = bip.avg_subtype_response_per_pixel

    pixels_seen = sorted(set([sub_d.keys() for sub_d in avg_response_dict.values()][0]))
    
    # can change here blue to black 
    features_array = np.zeros((len(pixels_seen), len(avg_response_dict) + 2)) # plus 2 because pixels need x and y
    subtypes = sorted(avg_response_dict.keys())

    missing_avgs = []
    # extract features
    features_array = extract_features(bip, return_labels = False, alpha_white = alpha_white)
    # manually change blue to black ?
    # run these features through model to get y_pred
    y_pred = model.predict(features_array[:,2:])
    y_pred = np.clip(y_pred, 0.0, 1.0)

    # now create an image using generate_img_from_feats
    y_pred_img = generate_img_from_feats(features_array, model, img_h, img_w, output_groundtruth = None)
    return y_pred_img

