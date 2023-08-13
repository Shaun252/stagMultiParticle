import numpy as np
import itertools


########################################################################################################################################
#### Momentum dict. (irrep keys of translational group)

def mom_dictionary():
    
    momentum = {"mom000" : [0,0,0],
                "mom001" : [0,0,1],
                "mom010" : [0,1,0],
                "mom100" : [1,0,0],
                "mom110" : [1,1,0],
                "mom101" : [1,0,1],
                "mom011" : [0,1,1],
                "mom111" : [1,1,1],
                "mom012" : [0,1,2],
                "mom102" : [1,0,2],
                "mom120" : [1,2,0],
                "mom021" : [0,2,1],
                "mom210" : [2,1,0],
                "mom201" : [2,0,1],
                "mom112" : [1,1,2],
                "mom112" : [1,2,1],
                "mom112" : [2,1,1],
                "mom123" : [1,2,3],
                "mom132" : [1,3,2],
                "mom213" : [2,1,3],
                "mom312" : [3,1,2],
                "mom231" : [2,3,1],
                "mom321" : [3,2,1]}
    
    return momentum

########################################################################################################################################
#### Functions for the translation group irrep generation T^3


def generate_translation_group_elements(N, dim):
    translation_dict = {}
    for elements in itertools.product(range(N), repeat = dim):
        translation_dict[elements] = np.array(elements)
                
    return translation_dict

def mom_character(mom_three_vec, group_element, N):
    # mom_three_vec ranges from 0 to (N-2)/2 (the -2 because N is even) (can be -1 for bosons as N even equirement for fermions)
    pdotx = np.dot(mom_three_vec, group_element)
    
    char_val = np.exp(1j * 2 *np.pi * pdotx /  N)
    
    return char_val 