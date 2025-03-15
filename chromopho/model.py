import numpy as np
import os
from sklearn.linear_model import Ridge
import joblib
from skimage.metrics import structural_similarity as ssim


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

def evaluate_ssim(y, y_pred):
    return ssim(y, y_pred, multichannel=True)

def evaluate_model_image(model, Xs_test, Ys_test, img_h, img_w):
    '''
    evaluates a model on the given features and labels based on ssim
    '''
    y_pred = model.predict(Xs_test[:,2:])
    y_pred = np.clip(y_pred, 0.0, 1.0)
    # re-add the pixel index to the y_pred in col 0 and 1
    # y_pred = np.hstack((Xs_test[:,:2], y_pred))    

    # make an image from the predicted values and an image for the labels
    y_pred_img = np.zeros((img_h, img_w, 3)) # output will be r,g,b
    y_img = np.zeros((img_h, img_w, 3))
    for i, (x, y, t1,t2,t3,t4,t5,t6) in enumerate(Xs_test):

        # I need to take the value at y_pred at a pixel index and put it in y_pred_img 
        # the indices of y_pred are the same as the indices of Xs_test so I should just be able to match via i
        # but that is not working 

        y_pred_img[int(x), int(y)] = y_pred[i]
        y_img[int(x), int(y)] = Ys_test[i,2:]
    
    # now get the ssim score
    #ssim_score = evaluate_ssim(y_img, y_pred_img)
    #print(f'SSIM: {ssim_score}')
    #return ssim_score, y_pred_img, y_img
    return y_pred_img, y_img

