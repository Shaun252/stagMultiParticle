import numpy as np
import itertools
from numpy.linalg import inv, matrix_power
import os
import pickle

import stagGroup as sg
import gfunctions as gf
import cliff41Group as c41

###########################################################################################################################################
### Functions to generate faithful 0 momentum dict of gamma_4_1 rtimes W3 through isomorphism of gamma_2_2 x SW4. Also functions which relabel the elements of gamma_2_2 x SW4 to elements of gamma_4_1 rtimes W3

def generate_SW4_irrep_one_half():
    ### Generates one 3d irrep pure_rotations dictionary where the dictionary has form 
    ### key = "group element label", value = "matrix of group element"
    ### Also can generate "complete dict" which gives multiply equations for group elements
    ### Important to note negative identity is a pure rotatation here not an inversion (can only invert an odd)
    ### number of axis at time
    ### This is known as the (1/2, 1/2) representation in sharpes paper
    
    
    R_12 =  np.array([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
    R_31 =  np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
    R_23 =  np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
    
    
    R_41 = np.array([[0,0,0,-1],[0,1,0,0],[0,0,1,0],[1,0,0,0]])
    R_42 = np.array([[1,0,0,0],[0,0,0,-1],[0,0,1,0],[0,1,0,0]])
    R_43 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,-1],[0,0,1,0]])
    
    E = np.identity(4)

    dict_rot = {}

    for i in range(1,5):
    
        dict_rot["*R_12"*i] = matrix_power(R_12, i)
        dict_rot["*R__23"*i] = matrix_power(R_23, i)
        dict_rot["*R___31"*i] = matrix_power(R_31, i)
        
        
        dict_rot["*R__23*-gamma_2*gamma_3"*i] = matrix_power(R_41, i)
        dict_rot["*R___31*gamma_1*gamma_3"*i] = matrix_power(R_42, i)
        dict_rot["*R_12*-gamma_1*gamma_2"*i] = matrix_power(R_43, i)
    
    dict_copy = dict_rot.copy()
    
    for key1, val1 in dict_copy.items():
        for key2, val2 in dict_copy.items():
            dict_rot[key1+key2] = np.dot(val1 , val2)
    
    dict2_copy = dict_rot.copy()
    
    
    for key1, val1 in dict_copy.items():
        for key2, val2 in dict2_copy.items():
            dict_rot[key1+key2] = np.dot(val1 , val2)
            
            
    dict_rot["E"] = E
    
    unique_set = np.unique(list(dict_rot.values()), axis=0)

    final_dict = {}

    for mat in unique_set:
        list_keys = [key.lstrip("*") for (key, vals) in dict_rot.items() if np.all(vals ==mat)]
        label = sorted(list_keys,key=len)[0]
        final_dict[label] = mat
        
    return final_dict



def generate_clifford_2_2_faithful_irrep():
    ### Taken form the appendix of Gattringer & Lang, the chiral rep
    
    clifford_4_faithful_irrep = c41.generate_clifford_4_faithful_irrep()
    
    gamma_dict = clifford_4_faithful_irrep.copy()
    
    gamma_5 = gamma_dict["gamma_0"] @ gamma_dict["gamma_1"] @ gamma_dict["gamma_2"] @ gamma_dict["gamma_3"]
    
    #C_0 = 1j*gamma_5
    
    #gamma_123 = gamma_dict["gamma_1"] @ gamma_dict["gamma_2"] @ gamma_dict["gamma_3"]
    
    gamma_0 = gamma_dict["gamma_0"]
    
    #gamma_new = C_0 @ gamma_0 @ gamma_dict["gamma_2"]
    
    C_0 = 1j*gamma_dict["gamma_1"]
    
    gamma_123 = 1j*gamma_dict["gamma_2"]
    
    gamma_new = gamma_dict["gamma_3"]
    
    dict_rot = {}
    
    for i in range(1,4):
    
        dict_rot["*C_0"*i] = matrix_power(C_0, i)
        dict_rot["*gamma_1*gamma_2*gamma_3"*i] = matrix_power(gamma_123, i)
        dict_rot["*gamma_0"*i] = matrix_power(gamma_0, i)
        dict_rot["*C_0*gamma_0*I_S"*i] = matrix_power(gamma_new, i)
        
    
    dict_copy = dict_rot.copy()
    
    for key1, val1 in dict_copy.items():
        for key2, val2 in dict_copy.items():
            dict_rot[key1+key2] = np.dot(val1 , val2)
            
    dict2_copy = dict_rot.copy()
            
    for key1, val1 in dict2_copy.items():
        for key2, val2 in dict2_copy.items():
            dict_rot[key1+key2] = np.dot(val1 , val2)

    E = np.identity(4)
    dict_rot["E"] = E
    dict_rot["-E"] = -1*E
    
    unique_set = np.unique(list(dict_rot.values()), axis=0)

    first_draft_dict = {}

    for mat in unique_set:
        list_keys = [key.lstrip("*") for (key, vals) in dict_rot.items() if np.all(vals ==mat)]
        label = sorted(list_keys,key=len)[0]
        first_draft_dict[label] = mat
    
    first_draft_dict_copy = first_draft_dict.copy()
    
    for key, val in first_draft_dict_copy.items():
        first_draft_dict['-'+key] = -1* val
    
    unique_set = np.unique(list(first_draft_dict.values()), axis=0)
    
    final_dict = {}
    for mat in unique_set:
        list_keys = [key.lstrip("*") for (key, vals) in first_draft_dict.items() if np.all(vals ==mat)]
        label = sorted(list_keys,key=len)[0]
        final_dict[label] = mat
    
    
    return final_dict


def generate_SW4_clifford_2_2_product(cliff_2_2_faithful):
    ###Need to use 2 dim rep of Z2 as, (SW4 x Gamma_2_2) / {-1x-1} contains -I so cant use D(Z_2 )= 1,-1

    SW4 = generate_SW4_irrep_one_half()
       
    prod_dict = {}
    
    for sw4_key, sw4_val in SW4.items():
        for cliff_2_2_key, cliff_2_2_val in cliff_2_2_faithful.items():
            prod_dict[sw4_key, cliff_2_2_key] = np.kron(sw4_val, cliff_2_2_val)
    
    unique_set = np.unique(list(prod_dict.values()), axis=0)
    
    complete_dict = {}

    
    for mat in unique_set:
        list_keys = [key for (key, vals) in prod_dict.items() if np.all(vals ==mat)]
        complete_dict[np.array_str(mat)] = list_keys
        if len(list_keys[0][0]) > len(list_keys[1][0]):
            label = list_keys[0]
        else:
            label = list_keys[1]
        
        del prod_dict[label]
   
    return prod_dict


def isomorphism_direct_prod_semi_direct_prod(original_key, faithful_clifford_irrep, faithful_SW3_irrep):
    #### Takes a dict label from (SW4 x D4) / ({-1x-1}) x Z2 and converts it into a dict label for Gamma_4 \rtimes W3
    #### The original key format is in from (SW4 , D4, Z2) and final key format is (W3, Gamma_4)
    #### Can use W3["T_1"] as faithful SW3 irrep
    
    rot_base = np.identity(3)
    gamma_base = np.identity(4)
    full_rotation_list = []
    overall_minus_power=0
    for k, key_string in enumerate(original_key):
        #### Remove all - from the first / second components of the key and count them
        overall_minus_power += key_string.count("-")
        key_string = key_string.replace("-", "")
        
        
        #### split the first/second/third component strings into lists
        group_element_list = key_string.split("*")
        
        if k ==0:
            ###### For the first string / component of the of the key (SW4 element), we need to commute all gammas past the
            ###### rotation matrices into the second component to match the final key format while keeping track of the
            ###### "-" signs
            for j, group_element in enumerate(group_element_list[::-1]):
                if "gamma_" in group_element:
                    rotation_list = []
                    gamma = group_element
                    
                    if j != 0:
                        for rotation_gamma in group_element_list[-j:]:
                            if "R_" in rotation_gamma:
                                
                                rotation_list.append(rotation_gamma)
                        
                    for rotation in rotation_list:
                        gamma, minus_sign = sg.gamma_rotation_commutation_relations(gamma, rotation)
                        overall_minus_power += minus_sign
                    ###### Taking the product of the gamma matrices as we go
                    gamma_matrix = faithful_clifford_irrep[gamma]
                    gamma_base = np.dot(gamma_matrix, gamma_base)
                
                else:
                    #### Storing the rotationa matrix labels to eventually take the product
                    full_rotation_list.append(group_element)
                
            
            for rotation_label in full_rotation_list[::-1]:
                #### Computing the pure rotation component of W3 for first component of final key
                rotation_matrix = faithful_SW3_irrep[rotation_label]
                rot_base = np.dot(rot_base, rotation_matrix)

        else:
            #### Dealing with the second and third components of the original key
            if 'I_S' in group_element_list:
                #Dont need to worry about commuting IS past the gammas in the first part of the key as they 
                #appear in pairs of spatial gammas which gives a factor of 1 when you commut IS
                I_S_index = group_element_list.index('I_S')
                for gammas in group_element_list[:I_S_index]:
                    if gammas[-1] !='0':
                        overall_minus_power += 1
            

            for group_element in group_element_list:
                if "gamma_" in group_element or group_element == 'C_0':
                    gamma_matrix = faithful_clifford_irrep[group_element]
                    gamma_base = np.dot(gamma_base, gamma_matrix)
            
    gamma_label = gf.what_matrix(gamma_base, faithful_clifford_irrep)
    rot_label = gf.what_matrix(rot_base, faithful_SW3_irrep)

    if "I_S" in original_key[1]:
        #### Stripping the parity label from the third component of original key and adding it the first of (W3, Gamma4) key
        rot_label += "*I_S"
        if "E" in rot_label:
            rot_label = rot_label.replace("E*","")
    if overall_minus_power % 2 ==1:
        #### Determining the overall "-" sign and adding it to the Gamma_4 part of the key
        if "-" in gamma_label:
            gamma_label = gamma_label.replace("-" , "")
        
        else:
            gamma_label = "-" + gamma_label
        
    return (rot_label, gamma_label)


def convert_SW4_cliff_2_2_dict(SW4_cliff_2_2_dict, faithful_clifford_4_1_irrep, faithful_SW3_irrep):
    new_dict = {}
    for key, vals in SW4_cliff_2_2_dict.items():
        new_key = isomorphism_direct_prod_semi_direct_prod(key, faithful_clifford_4_1_irrep, faithful_SW3_irrep)
        new_dict[new_key] = vals
        
    return new_dict


def generate_converted_SW4_clifford_2_2_dict(W3_irrep_dict, cliff41_irrep_dict):
    
    faithful_clifford_4_1_irrep = cliff41_irrep_dict['gamma']
    
    cliff_2_2_faithful = generate_clifford_2_2_faithful_irrep()
    
    SW4_cliff_2_2_dict = generate_SW4_clifford_2_2_product(cliff_2_2_faithful)   
    
    converted_dict = convert_SW4_cliff_2_2_dict(SW4_cliff_2_2_dict, faithful_clifford_4_1_irrep, W3_irrep_dict['T_1'])
    
    
    return converted_dict



########################################################################################################################################
########################################################################################################################################
#### Function to generate the the full character set of SW4, and the direct product (SW4 x D4)/(-1 x -1)  (NEED TO CHANGE TO GAMMA_2_2

def generate_SW4_character_dict():
    ##### Taken from Mandula / Sharpe paper
    one_half_one_half = generate_SW4_irrep_one_half()
    classes = gf.conjugate_classes(one_half_one_half)
    one_half = gf.compute_characters(one_half_one_half, one_half_one_half, class_dictionary=classes)
    
    irrep_dict = {}
    irrep_dict["one_half"] = one_half
    
    link_dict = {"class1" : ["E"], 
                 "class2" : ["R_12*R_12"], 
                 "class3" : ["R_12*R_12*-gamma_1*gamma_2*R_12*-gamma_1*gamma_2*R_12",
                             "R_12*R_12*R_12*-gamma_1*gamma_2*R_12*-gamma_1*gamma_2"],
                 "class4" : ["R_12"], 
                 "class5" : ["R_12*R_12*R__23"], 
                 "class6" : ["R_12*R_12*R_12*-gamma_1*gamma_2", 'R_12*-gamma_1*gamma_2*R_12*R_12'], 
                 "class7" : ["R_12*R__23"], 
                 "class8" : ["R_12*-gamma_1*gamma_2*R_12*R_12*R__23"], 
                 "class9" : ["R_12*R__23*R_12*-gamma_1*gamma_2"], 
                 "class10" : ["R_12*R__23*R___31*gamma_1*gamma_3"],
                 "class11" : ["R_12*R_12*-gamma_1*gamma_2*R__23*R__23"], 
                 "class12" : ["R_12*-gamma_1*gamma_2*R_12*R_12*R_12"],
                 "class13" : ["R_12*R_12*-gamma_1*gamma_2"]}
    
    class_conversion_dict = {}
    for class_label, class_values in classes.items():
        for link_label, link_value in link_dict.items():
            for link_vals in link_value:
                if link_vals in class_values:
                    class_conversion_dict[link_label] = class_label
                    #class_conversion_dict[class_label] = link_label

    ########################################################################################################
    
    
    trivial_irrep = {key : 1 for key in classes.keys()}
    
    irrep_dict["trivial_irrep"] = trivial_irrep 
    
    anti_sym = trivial_irrep.copy()
    anti_sym["class4"] = -1
    anti_sym["class5"] = -1
    anti_sym["class6"] = -1
    anti_sym["class9"] = -1
    anti_sym["class10"] = -1
    
    irrep_dict["anti_sym"] = anti_sym
    
    ##########################################################################################################
    
    two_two = {key : 2 for key in classes.keys()}
    two_two["class4"] = 0
    two_two["class5"] = 0
    two_two["class6"] = 0
    two_two["class7"] = -1
    two_two["class8"] = -1
    two_two["class9"] = 0
    two_two["class10"] = 0
    
    irrep_dict["two_two"] = two_two
    
    #################################################################################################
    
    three_one = {key : 3 for key in classes.keys()}
    three_one["class4"] = 1
    three_one["class5"] = 1
    three_one["class6"] = 1
    three_one["class7"] = 0
    three_one["class8"] = 0
    three_one["class9"] = -1
    three_one["class10"] = -1
    three_one["class11"] = -1
    three_one["class12"] = -1
    three_one["class13"] = -1
    
    irrep_dict["three_one"] = three_one
    
    ###################################################################################
    
    two_one_one = {key : 3 for key in classes.keys()}
    two_one_one["class4"] = -1
    two_one_one["class5"] = -1
    two_one_one["class6"] = -1
    two_one_one["class7"] = 0
    two_one_one["class8"] = 0
    two_one_one["class9"] = 1
    two_one_one["class10"] = 1
    two_one_one["class11"] = -1
    two_one_one["class12"] = -1
    two_one_one["class13"] = -1
    
    irrep_dict["two_one_one"] = two_one_one
    
    ################################################################################################
    
    six = {key : 0 for key in classes.keys()}
    six["class1"] = 6
    six["class2"] = -2
    six["class3"] = 6
    six["class11"] = 2
    six["class12"] = -2
    six["class13"] = -2
    
    irrep_dict["six"] = six
    
    ##############################################################################################################
    
    one_zero = {key : -1 for key in classes.keys()}
    one_zero["class1"] = 3
    one_zero["class3"] = 3
    one_zero["class4"] = 1
    one_zero["class6"] = 1
    one_zero["class7"] = 0
    one_zero["class8"] = 0
    one_zero["class9"] = 1
    one_zero["class13"] = 3
    
    irrep_dict["one_zero"] = one_zero
    
    #################################################################################################
    
    zero_one = {key : -1 for key in classes.keys()}
    zero_one["class1"] = 3
    zero_one["class3"] = 3
    zero_one["class4"] = 1
    zero_one["class6"] = 1
    zero_one["class7"] = 0
    zero_one["class8"] = 0
    zero_one["class10"] = 1
    zero_one["class12"] = 3
    
    irrep_dict["zero_one"] = zero_one
    
    ###################################################################################################
    
    eight = {key : 0 for key in classes.keys()}
    eight["class1"] = 8
    eight["class2"] = 0
    eight["class3"] = -8
    eight["class7"] = -1
    eight["class8"] = 1
    
    irrep_dict["eight"] = eight
    
    ##################################################################################################
    
    
    for key1, val1 in irrep_dict.copy().items():
        for key, val in val1.copy().items():
            if key not in class_conversion_dict.values():
                irrep_dict[key1][class_conversion_dict[key]] = irrep_dict[key1].pop(key)
            
    
    one_zero_bar = {key : np.multiply(val, anti_sym[key]) for key, val in one_zero.items()}
    
    irrep_dict["one_zero_bar"] = one_zero_bar
    
    zero_one_bar = {key : np.multiply(val, anti_sym[key]) for key, val in zero_one.items()}
    
    irrep_dict["zero_one_bar"] =  zero_one_bar
    
    one_half_bar = {key : np.multiply(val, anti_sym[key]) for key, val in one_half.items()}
    
    irrep_dict["one_half_bar"] = one_half_bar
    
    
    return irrep_dict


########################################################################################################################################
#### Function to generate gamma_2_2 characters


def cliff_2_2_bosonic_character(C_0, gamma_0, gamma_123, C_0_gamma_0_I_S,  group_element):
    
    
    cliff_2_2_irrep_key = np.mod((C_0, gamma_0, gamma_123, C_0_gamma_0_I_S), 2*np.pi)
    
    cliff_2_2_irrep_irrep_vec = np.real(np.exp(np.multiply(1j , cliff_2_2_irrep_key)))
    
    character_res_arr = np.ones((len(cliff_2_2_irrep_irrep_vec)))
    for mu, Xi_mu in enumerate(group_element):
        if Xi_mu == 1:
            character_res_arr[mu] = cliff_2_2_irrep_irrep_vec[mu]
            
    char_val = np.prod(character_res_arr)  
    
    return np.real(char_val)


def generate_clifford_2_2_bosonic_irreps(clifford_2_2_faithful_dict):
    # Bosonic irreps commute so need to worry about minus sign
    irrep_labels = itertools.product((np.pi,0),repeat=4)
    
    complete_bosonic_irreps_dict = {}
    
    for irrep_label in irrep_labels:
        bosonic_irrep_dict = {}
    
        for key, vals in clifford_2_2_faithful_dict.items():
            group_element = np.zeros((4), dtype=float)
 
            if 'C_0*gamma_0*I_S' in key:
                group_element[3] = 1
                new_key=key.replace('C_0*gamma_0*I_S',"")
            else:
                new_key = key
            if 'gamma_1*gamma_2*gamma_3' in new_key:
                group_element[2] = 1
            if 'gamma_0' in new_key:
                group_element[1] = 1
            if 'C_0' in new_key:
                group_element[0] = 1
            
            bosonic_irrep_dict[key] = cliff_2_2_bosonic_character(irrep_label[0], irrep_label[1], irrep_label[2], irrep_label[3], group_element)
            
        complete_bosonic_irreps_dict[irrep_label] = bosonic_irrep_dict
        
    return complete_bosonic_irreps_dict

def generate_clifford_2_2_complete_irrep_dict():
    
    complete_irrep_dict = {}
    
    clifford_faithful_dict = generate_clifford_2_2_faithful_irrep()
    
    bosonic_irreps = generate_clifford_2_2_bosonic_irreps(clifford_faithful_dict) 

    complete_irrep_dict['gamma'] = clifford_faithful_dict
    
    complete_irrep_dict.update(bosonic_irreps)
    
    return complete_irrep_dict

########################################################################################################################################
########################################################################################################################################
#### Functions to compute SW4 x Gamma 2_2 char_dict, first in SW4 x Gamma 2_2 language then convert to Gamma_4_1 rtimes W3


def eipi(val):
    if val == np.pi:
        return -1
    else:
        return 1

def generate_SW4_gamma_2_2_direct_product_character_dict_basic():
    
    cliff_2_2_dict = generate_clifford_2_2_complete_irrep_dict()
    
    cliff_2_2_classes = gf.conjugate_classes(cliff_2_2_dict['gamma'])
    
    cliff_2_2_char_dict = gf.compute_characters_irrep_dict(irrep_dict=cliff_2_2_dict, faithful_matrix_dict=cliff_2_2_dict['gamma'], class_dictionary=cliff_2_2_classes)
    
    SW4_irrep_chars = generate_SW4_character_dict()
    
    possible_minusI_labels = ["R_12*R_12*-gamma_1*gamma_2*R_12*-gamma_1*gamma_2*R_12", 
                             "R_12*R_12*R_12*-gamma_1*gamma_2*R_12*-gamma_1*gamma_2"]
    
    minusI_label = [label for label in possible_minusI_labels if label in SW4_irrep_chars["trivial_irrep"].keys()][0]
    
    direct_product_char_dict_complete = {}
    
    for cliff_2_2_irrep_label, cliff_2_2_irrep in cliff_2_2_char_dict.items():
        for SW4_irrep_labels, SW4_irreps in SW4_irrep_chars.items():
            if cliff_2_2_irrep["-E"] == cliff_2_2_irrep["E"] and SW4_irreps[minusI_label] == SW4_irreps["E"]:
                direct_product_char_dict = {}
                
                for cliff_2_2_irrep_element_labels, cliff_2_2_irrep_char in cliff_2_2_irrep.items():
                    for SW4_irrep_element_labels, SW4_irrep_char in SW4_irreps.items():
                        direct_product_char_dict[SW4_irrep_element_labels, cliff_2_2_irrep_element_labels] = SW4_irrep_char * cliff_2_2_irrep_char
                direct_product_char_dict_complete[SW4_irrep_labels, tuple(map(eipi, cliff_2_2_irrep_label))] = direct_product_char_dict
                        
                
            elif cliff_2_2_irrep["-E"] != cliff_2_2_irrep["E"] and SW4_irreps[minusI_label] != SW4_irreps["E"]:
                direct_product_char_dict = {}
                
                for cliff_2_2_irrep_element_labels, cliff_2_2_irrep_char in cliff_2_2_irrep.items():
                    for SW4_irrep_element_labels, SW4_irrep_char in SW4_irreps.items():
                        direct_product_char_dict[SW4_irrep_element_labels, cliff_2_2_irrep_element_labels] = SW4_irrep_char * cliff_2_2_irrep_char
            
                direct_product_char_dict_complete[SW4_irrep_labels, cliff_2_2_irrep_label] = direct_product_char_dict
                        
    return direct_product_char_dict_complete
            

def remove_mom_key(key):
    return (key[1], key[2])
    
    
def mom_reduced_class_dict(class_dict):
    reduced_class_dict = {}
    for key, val in class_dict.items():
        reduced_class_dict[remove_mom_key(key)] = list(map(remove_mom_key, val))
    return reduced_class_dict
        

def generate_converted_SW4_clifford_2_2_char_dict(W3_irrep_dict, clifford_4_1_irrep_dict):
      
    char_dict = generate_SW4_gamma_2_2_direct_product_character_dict_basic()
    
    converted_char_dict = {}
    
    for irrep_SW4_clifford_2_2_label,  irrep_SW4_clifford_2_2 in char_dict.items():
    
        converted_dict = convert_SW4_cliff_2_2_dict(irrep_SW4_clifford_2_2, clifford_4_1_irrep_dict['gamma'], W3_irrep_dict['T_1'])
        
        converted_char_dict[irrep_SW4_clifford_2_2_label] = converted_dict
    
    ## Need to remove duplicate classes 17 (cliff irreps / classes x 13 ( Sw4 ireps / classes) = 221 - > 
    ## 163 (the number of irreps of the rest frame group), this because of modulo -1 x -1 in the 'direct product'
    
    essential_keys, class_dict = sg.essential_group_keys(1)
    
    # remove momentum from element label
    reduced_class_key_dict = mom_reduced_class_dict(class_dict)
    
    SW4_cliff22_converted_char_dict_class_reduced = gf.irrep_dict_restriction_class_dict(irrep_dict=converted_char_dict, subgroup=reduced_class_key_dict)
    
    return SW4_cliff22_converted_char_dict_class_reduced


########################################################################################################################################
#### Function to generate isomorphism between \Gamma_4 \rtimes W3 induced irreps and SW4 x D4 / Z2 zero mom irreps

def generate_zero_mom_irrep_comparison_dict(W3_irrep_dict, cliff41_irrep_dict):
    
    try:
        irrep_relation_dict = load_dict()
        
    except:
        print('didnt load')
        ## Note even numbers correspond to I_S = + for 000 taste and I_S = - for 00p (due to storing A_0 only once for both)
        irrep_relation_dict = {}

        bosonic_zero_mom_irreps = sg.generate_all_bosonic_momentum_specific_taste_irreps(mom_key='mom000', W3_irrep_dict=W3_irrep_dict, clifford_irrep_dict=cliff41_irrep_dict, pool_no=6)

        fermionic_zero_mom_irreps = sg.fermionic_taste_induced_representations(mom_key='mom000', W3_irrep_dict=W3_irrep_dict, clifford_irrep_dict=cliff41_irrep_dict)

        zero_mom_irreps = {**bosonic_zero_mom_irreps, **fermionic_zero_mom_irreps}

        SW4_cliff_2_2_char_dict_converted = generate_converted_SW4_clifford_2_2_char_dict(W3_irrep_dict=W3_irrep_dict, clifford_4_1_irrep_dict=cliff41_irrep_dict)

        essential_keys, class_dict = sg.essential_group_keys(1)

        # remove momentum from element label
        reduced_class_key_dict = mom_reduced_class_dict(class_dict)

        for key, vals in zero_mom_irreps.items():
            if np.array([list(vals.values())[0]]).shape ==(1,):
                dim == 1
            else:
                dim = list(vals.values())[0].shape[0]
            for key1, vals1 in SW4_cliff_2_2_char_dict_converted.items():
                if gf.check_orthogonality(vals, vals1, class_dictionary = reduced_class_key_dict) == False:

                    irrep_relation_dict[key] = [key1, dim]

        save_dict(zero_mom_group_iso_dict=irrep_relation_dict)
    return irrep_relation_dict


def save_dict(zero_mom_group_iso_dict):
    
    directory = './staggered_irreps/keys/'
    filename = 'zero_mom_group_iso_dict'
 
        
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    full_path = directory+ filename
    f = open(full_path,"wb")
    pickle.dump(zero_mom_group_iso_dict,f)
    f.close()
    
    return None


def load_dict():
    
    directory = './staggered_irreps/keys/'
    filename = 'zero_mom_group_iso_dict'
    
    full_path = directory+ filename
    
    with open(full_path, 'rb') as pfile:
    
        f = pickle.load(pfile)
        
    
    return f

########################################################################################################################################
#### Function to generate isomorphism between the mom 000 irrep of T_N \rtimes \Gamma_4 \rtimes W3 induced irreps and SW4 x D4 / Z2 zero mom irreps

def mom_reduced_irrep_dict(irrep_dict):
    reduced_irrep_dict = {}
    for key, val in irrep_dict.items():
        reduced_irrep_dict[remove_mom_key(key)] = val
    return reduced_irrep_dict

def generate_zero_mom_irrep_comparison_dict_v2(W3_irrep_dict, cliff41_irrep_dict):

    ## Note even numbers correspond to I_S = + for 000 taste and I_S = - for 00p (due to storing A_0 only once for both)
    irrep_relation_dict = {}

    zero_mom_irreps = sg.load_irrep(N=1, mom_key = 'mom000')


    #sg.momentum_induced_bosonic_representation(mom_key='mom000', N=1, dim=3, W3_irrep_dict=W3_irrep_dict, clifford_irrep_dict=cliff41_irrep_dict, converted_SW4_clifford_2_2_dict=converted_SW4_clifford_2_2_dict, pool_no=2)

    zero_mom_irreps_reduced = {}

    for irrep_label, irrep in zero_mom_irreps.items():
        zero_mom_irreps_reduced[irrep_label] = mom_reduced_irrep_dict(irrep)


    SW4_cliff_2_2_char_dict_converted = generate_converted_SW4_clifford_2_2_char_dict(W3_irrep_dict=W3_irrep_dict, clifford_4_1_irrep_dict=cliff41_irrep_dict)

    essential_keys, class_dict = sg.essential_group_keys(1)

    # remove momentum from element label
    reduced_class_key_dict = mom_reduced_class_dict(class_dict)

    for key, vals in zero_mom_irreps_reduced.items():
        for key1, vals1 in SW4_cliff_2_2_char_dict_converted.items():
            #print(gf.check_orthogonality_v2(vals, vals1, class_dictionary = reduced_class_key_dict))
            if gf.check_orthogonality(vals, vals1, class_dictionary = reduced_class_key_dict) == False:


                irrep_relation_dict[key1] = key
                
    return irrep_relation_dict


####################################################################################################################################
### Reducing SW4 to SW3

def SW4_SW3_char_dict_restriction():
    
    SW4_char_dict = rf.generate_SW4_character_dict()
    SW3_irreps = W3.generate_SW3_irrep_dict()
        
            
    
    SW4_restricted_char_dict = {}
    for irrep_label, irrep in SW4_char_dict.items():
        restricted_irrep_dict = {}
        for class_key, class_char in irrep.items():
            if 'gamma' not in class_key:
                restricted_irrep_dict[class_key] = class_char
                
        SW4_restricted_char_dict[irrep_label] = restricted_irrep_dict
        
    SW3_char_dict = {}
    for irrep_label, irrep in SW3_irreps.items():
        SW3_irrep_char_dict = {}
        for class_key in list(SW4_restricted_char_dict.values())[0]:
            try:
                SW3_irrep_char_dict[class_key] = round(np.trace(irrep[class_key]),5)
            except:
                SW3_irrep_char_dict[class_key] = irrep[class_key]
        
        SW3_char_dict[irrep_label] = SW3_irrep_char_dict
    
    class_dict = gf.conjugate_classes(SW3_irreps['T_1'])
    
    #class_dict['R___31*R_12'] = class_dict['R_12*R___31']
    #del class_dict['R_12*R___31']
    
    #print(SW4_restricted_char_dict)
    #print('')
    #print(SW3_char_dict)
    irrep_decomp_dict = {}
    for irrep_label, irrep in SW4_restricted_char_dict.items():
        
        irrep_decomp_dict[irrep_label] = []
        for irrep_label1, irrep1 in SW3_char_dict.items():
            number_times_irrep_contained = int(np.real(gf.check_orthogonality_v2(irrep, irrep1, class_dictionary=class_dict)))
        
            for number in range(number_times_irrep_contained):
                irrep_decomp_dict[irrep_label].append((irrep_label1, number))
            
         
    return irrep_decomp_dict
    