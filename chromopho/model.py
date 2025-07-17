import numpy as np
import os
import joblib
from collections import defaultdict

from skimage.metrics import structural_similarity as ssim
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.kernel_ridge import KernelRidge
from .feature_extraction import extract_features
from .bipolar_image import BipolarImageProcessor
from sklearn.model_selection import KFold
import torch


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

def train_model_cv(X_train, y_train, alphas=None):
    # hard coded for now
    k=5

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


def train_model_kernel_ridge(X_train, y_train, alpha=1.0, kernel='rbf', gamma=None):
    X = X_train[:, 2:]
    Y = y_train[:, 2:]

    model = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma)
    model.fit(X, Y)

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
    for i, (x, y) in enumerate(feats[:,:2]):

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


def generate_img_from_feats_pytorch(feats, model, img_h, img_w, output_groundtruth=None):
    '''
    Generates an image from the predicted values using a PyTorch model
    '''
    # Convert features
    feats_only = feats[:,2:]
    feats_tensor = torch.from_numpy(feats_only).float().to('mps')

    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(feats_tensor)

    y_pred = y_pred_tensor.cpu().numpy()
    y_pred = np.clip(y_pred, 0.0, 1.0)

    # Create blank images
    y_pred_img = np.zeros((img_h, img_w, 3))
    y_img = None
    if output_groundtruth is not None:
        y_img = np.zeros((img_h, img_w, 3))

    # Fill predicted image
    for i, (x, y) in enumerate(feats[:, :2]):
        x = int(x)
        y = int(y)
        if 0 <= x < img_h and 0 <= y < img_w:
            y_pred_img[x, y] = y_pred[i]
            if output_groundtruth is not None:
                y_img[x, y] = output_groundtruth[i, 2:]

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
    if return_sim:
        ssim_score = evaluate_ssim(y_img, y_pred_img)
        #print(f'SSIM: {ssim_score}')
        return y_pred_img, y_img, ssim_score
    else:
        return y_pred_img, y_img

def phosphene_simulation(radius, mosaic, model, dummy_img, stim_response=1, seed=None, alpha_white=False,
                         tensor_model=False, smooth = False, gaussian_blur = False,
                         black_encoding = {-1:0, 1:.55, 2:.45, 3:.55, 4: .45, 5:.55, 6:.45, 7:.45, 8:.55}):
    '''
    simulates a phosphene with a given radius
    black vec is the vector that represents black in the space of the model 
    '''
    if seed is not None:
        np.random.seed(seed)
    # create a blank map for cell responses 
    img_h, img_w = dummy_img.shape[0], dummy_img.shape[1]
    mosaic_h, mosaic_w = mosaic.grid.shape[0], mosaic.grid.shape[1]
    # bipolar_cell_responses = np.zeros((mosaic_h, mosaic_w))
    black_lookup_vec = np.zeros(len(black_encoding)+1) # plus one because the encoding starts at -1 but we want to be able to index
    for k, v in black_encoding.items():
        black_lookup_vec[k + 1] = v
    bipolar_cell_responses = black_lookup_vec[mosaic.grid + 1]
    # get the center of the image 
    center_x, center_y = mosaic_h//2, mosaic_w//2
    # randomly shift circle_indives by a random amount that is constrained by the size of the mosaic
    center_phosphene_x = center_x + np.random.randint(0, mosaic_h/2.1)
    center_phosphene_y = center_y + np.random.randint(0, mosaic_w/2.1)
    print(center_phosphene_x, center_phosphene_y)
    # get the indices of the cells that are within the radius
    for i in range(mosaic_h):
        for j in range(mosaic_w):
            if (i - center_phosphene_x)**2 + (j - center_phosphene_y)**2 <= radius**2:
                bipolar_cell_responses[i, j] = stim_response
    # now create a BipolarImageProcessor object
    bip = BipolarImageProcessor(mosaic, dummy_img, stimulation_mosaic = bipolar_cell_responses)
    # now run through model to get y_pred
    # need an array with pixel_x, pixel_y, subtype_response vector for each subtype in the order of ordered_subtypes
    avg_response_dict = bip.avg_subtype_response_per_pixel
    pixels_seen = sorted(set([sub_d.keys() for sub_d in avg_response_dict.values()][0]))
    subtypes = sorted(avg_response_dict.keys())
    # extract features
    features_array = extract_features(bip, return_labels=False, alpha_white=alpha_white, smooth = smooth, gaussian_blur = gaussian_blur)

    # the average features need to be black encoding 
    if tensor_model:
        # PyTorch model path
        import torch
        # convert features to tensor
        feats_only = features_array[:, 2:]
        feats_tensor = torch.from_numpy(feats_only).float().to('mps')

        # predict with PyTorch model
        model.eval()
        with torch.no_grad():
            y_pred_tensor = model(feats_tensor)
        # bring prediction back to cpu and numpy
        y_pred = y_pred_tensor.cpu().numpy()
        y_pred = np.clip(y_pred, 0.0, 1.0)
        # now create an image from predicted values
        img = np.zeros((img_h, img_w, 3))
        for idx, (x, y) in enumerate(features_array[:, :2]):
            x = int(x)
            y = int(y)
            if 0 <= x < img_h and 0 <= y < img_w:
                img[x, y] = y_pred[idx]
        y_pred_img = img
    else:
        # ridge model path
        y_pred = model.predict(features_array[:, 2:])
        y_pred = np.clip(y_pred, 0.0, 1.0)

        # create an image using generate_img_from_feats
        y_pred_img = generate_img_from_feats(features_array, model, img_h, img_w, output_groundtruth=None)
        
    return y_pred_img



