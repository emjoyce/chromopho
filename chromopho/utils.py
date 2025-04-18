import numpy as np
import os


def _parse_cone_string(cone_string):
    '''
    Parses something like '+l', '-m', '+ls', '-lms' into an (L, M, S) tuple
    indicating +1 or -1 for each relevant cone.
    Example:
        '+ls' -> (1, 0, 1)
        '-lm' -> (-1, -1, 0)
    '''
    sign = 1 if cone_string[0] == '+' else -1
    l, m, s = 0, 0, 0
    for c in cone_string[1:]:  # skip the initial '+' or '-'
        if c == 'l':
            l += sign
        elif c == 'm':
            m += sign
        elif c == 's':
            s += sign
    return (l, m, s)


def _subtype_to_cone_string(subtype):
    '''takes subtype, gets cone string, returns center cone and surroudn cone string'''
    sub_name = subtype.name
    center_cone = ''
    surround_cone = ''
    if 'on' in sub_name:
        center_cone = '+'
        surround_cone = '-'
    if 'off' in sub_name:
        center_cone = '-'
        surround_cone = '+'
    if 's' in sub_name:
        center_cone += 's'
        surround_cone += 'lm'
    if 'm' in sub_name:
        center_cone += 'm'
        surround_cone += 'l'
    if 'l' in sub_name:
        center_cone += 'l'
        surround_cone += 'm'
    return center_cone, surround_cone

def _subtype_to_lms(subtype):
    '''
    takes subtype, gets cone string, returns center cone and surround cone lms values
    '''
    center_cone_surround_cone = _subtype_to_cone_string(subtype)
    center_cone = _parse_cone_string(center_cone_surround_cone[0])
    surround_cone = _parse_cone_string(center_cone_surround_cone[1])
    return center_cone, surround_cone


def img_to_rgb(img):
    '''
    converts a p2p image to an rgb image
    '''
    return img.reshape(img.shape)[:,:,:3]



def save_structured_features(array, output_dir, filename_base, filename_extension):
    """
    Saves a numpy array with first two columns as integers and the rest as floats,
    ensuring type consistency when saving and loading.

    Parameters:
    - array (np.ndarray): Input numpy array with at least 2 columns.
    - output_dir (str): Directory where the file should be saved.
    - filename_base (str): Base filename (without extension).
    - filename_extension (str): File extension (e.g. '_features.npy' or '_labels.npy').
    
    Returns:
    - str: Full path of the saved file.
    """
    if array.shape[1] < 3:
        raise ValueError("Array must have at least 3 columns (x,y of pixel + at least 1 subtype response column).")

    # first two columns int, rest float
    num_float_cols = array.shape[1] - 2
    dtype = [('x', 'i4'), ('y', 'i4')] + [(f'col{i+3}', 'f4') for i in range(num_float_cols)]

    # Convert to structured array
    structured_array = np.zeros(array.shape[0], dtype=dtype)
    structured_array['x'] = array[:, 0].astype(np.int32)
    structured_array['y'] = array[:, 1].astype(np.int32)

    for i in range(num_float_cols):
        structured_array[f'col{i+3}'] = array[:, i+2]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save file
    filepath = os.path.join(output_dir, filename_base + filename_extension)
    np.save(filepath, structured_array)

# TODO : a function that will pull out pixels x radius away from a given pixel 