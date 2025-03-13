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
    return img.data.reshape(img.img_shape)[:,:,:3]