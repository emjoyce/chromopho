import numpy as np


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


def img_to_rgb(img):
    '''
    converts a p2p image to an rgb image
    '''
    return img.data.reshape(img.img_shape)[:,:,:3]