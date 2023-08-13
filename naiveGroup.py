import numpy as np
import os
import pickle
from numpy.linalg import inv, matrix_power
from functools import partial
from multiprocessing import Pool
from collections import ChainMap

import gfunctions as gf
import translationGroup as tg
import W3RotationGroup as W3


################################################################################################################################################ Generate the group elements


def group_elements(N, W3):
    translation_group = tg.generate_translation_group_elements(N=N, dim=3)
    
    group_keys = []
    for key_rot in W3.keys():
        for key_trans in translation_group.keys():
            group_keys.append((key_trans, key_rot))
   
    return group_keys

################################################################################################################################################ Group commutation relations

def translation_rotation_commutation(trans_ele, rotation, N):
    
    if rotation == 'I_S':
        vec_rot_new = -1*trans_ele
    elif  rotation == 'E':
        vec_rot_new = trans_ele
    else:
        vec_rot_new = np.copy(trans_ele)

        rot_digits = rotation[-2:]

        axis1= int(rot_digits[0])-1
        axis2= int(rot_digits[1])-1

        vec_rot_new[axis1] =  -1 *trans_ele[axis2]
        vec_rot_new[axis2] =  trans_ele[axis1]
                
    return np.mod(vec_rot_new, N)


################################################################################################################################################ function to conjugate an element of the full group by another element


def conjugate_group_element(conj_ele, group_ele, N, W3):
    #Computing (R'' t'') = (R' t') (R t) (R' t')^-1
    # = R'  t' R  t t'^-1   R'^-1
    
    conj_ele_trans = conj_ele[0]
    conj_ele_rot = conj_ele[1]
    
    conj_ele_trans_inv = np.array([N,N,N]) - conj_ele_trans
    conj_ele_rot_inv =  gf.what_matrix(inv(W3[conj_ele_rot]), W3)
    
    group_ele_trans = group_ele[0]
    group_ele_rot = group_ele[1]
    
    #Consolidate  translations and gammas that are beside eachother, trans and gammas commute!
    total_trans_group_conj_inv = group_ele_trans + conj_ele_trans_inv
    
    # Bring W3 'group' element accross conj gamma and conj translation
    conj_trans_new = np.copy(conj_ele_trans)
    
    for rot_ele in group_ele_rot.split('*'):

            conj_trans_new = translation_rotation_commutation(trans_ele=conj_trans_new, rotation=rot_ele, N=N)

    #Consolidate translations and gammas that are beside eachother again
    
    total_trans_conj_group_conj_inv = np.mod(conj_trans_new + total_trans_group_conj_inv, N)
        
    #Bring accross conj inv rotation across consolidated gammas
    new_total_trans_conj_group_conj_inv = total_trans_conj_group_conj_inv
    for rot_ele in conj_ele_rot_inv.split('*'):
    
        new_total_trans_conj_group_conj_inv = translation_rotation_commutation(trans_ele=new_total_trans_conj_group_conj_inv, rotation=rot_ele, N=N)

    
    new_total_rot_conj_group_conj_inv = gf.what_matrix(W3[conj_ele_rot] @ W3[group_ele_rot] @ W3[conj_ele_rot_inv], W3)
    
    
    return (tuple(new_total_trans_conj_group_conj_inv), new_total_rot_conj_group_conj_inv)


################################################################################################################################################ Generate the group classes

def key_len(key):
    if type(key) != type((1,1)):
        return len(key)
    else:
        basekey = ''
        for subkey in key:
            basekey+= str(subkey)
        
        return len(basekey)
    

def group_classes(N, W3):
    
    group_keys = group_elements(N=N, W3=W3)
    
    directory = './naive_irreps/keys/'
    filename_classes_temp = 'translation_' +str(N) + '_W3_classes_temp'
    filename_classes_temp_list = 'translation_' +str(N) + '_W3_classes_temp_list'

    full_path_classes = directory+ filename_classes_temp
    full_path_classes_list = directory+ filename_classes_temp_list    
    
    try:  
    
        with open(full_path_classes, 'rb') as pfile, open(full_path_classes_list, 'rb') as lfile:
    
            conjugate_dict = pickle.load(pfile)
            total_list = pickle.load(lfile)
        print('Opened dict')
        print(len(conjugate_dict))
    except:
        conjugate_dict = {}
        total_list = []
        
        if not os.path.exists(directory):
            os.makedirs(directory)
          
    i=0
    
    for key1 in group_keys:
        
        conjugate_list = []
        conjugate_string_list = []
        print(key1)
        if str(key1) not in total_list:
            print(True)
            i+=1
            conjugate_list.append(key1)
            conjugate_string_list.append(str(key1))
            for key2 in group_keys:
                conjugate_label = conjugate_group_element(conj_ele=key2, group_ele=key1, N=N, W3=W3)
                conj_string = str(conjugate_label)

                if conj_string not in conjugate_string_list:
                    #print(True)
                    conjugate_list.append(conjugate_label)
                    conjugate_string_list.append(conj_string)

            conjugate_dict[sorted(conjugate_list,key=key_len)[0]] = conjugate_list
            total_list = total_list + conjugate_string_list
            
            f = open(full_path_classes, 'wb')
            pickle.dump(conjugate_dict, f)
            f.close()
            
            f = open(full_path_classes_list, 'wb')
            pickle.dump(total_list, f)
            f.close()
            
            print(i)
        else:
            print(False)
    return conjugate_dict


################################################################################################################################################ Generate the group W3 subgroup and the coset keys of TxW3/ W3. These are used for calculating clebsch gordons

def w3_cosets(W3_irreps):
  
    subgroup_keys = []
    for key, vals in W3_irreps["T_1"].items():
        subgroup_keys.append(((0,0,0), key))
        
    return subgroup_keys

def mom_cosets(N):
    translation_group = tg.generate_translation_group_elements(N, dim=3)
    coset_keys = []
    for key in translation_group.keys():
        coset_keys.append((key, "E"))
      
    return coset_keys



################################################################################################################################################ Generate the essential keys need to take direct products and calculate cgs, they are the class represnetatives + the subgroup, cosets above

def save_keys(essential_keys, group_class_dict, N):
    
    directory = './naive_irreps/keys/'
    filename = 'translation_' +str(N) + '_W3'
    filename_classes = 'translation_' +str(N) + '_W3_classes'
 
        
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    full_path = directory+ filename
    full_path_classes = directory+ filename_classes
    f = open(full_path,"wb")
    pickle.dump(essential_keys,f)
    f.close()
    
    f = open(full_path_classes,"wb")
    pickle.dump(group_class_dict,f)
    f.close()
    
    return None


def load_keys(N):
    
    directory = './naive_irreps/keys/'
    filename = 'translation_' +str(N) + '_W3'
    filename_classes = 'translation_' +str(N) + '_W3_classes'
    
    full_path = directory+ filename
    full_path_classes = directory+ filename_classes
    
    with open(full_path, 'rb') as pfile:
    
        f = pickle.load(pfile)
        
    with open(full_path_classes, 'rb') as pfile:
    
        g = pickle.load(pfile)
    
    return f, g
    
    
def essential_group_keys(N):
    
    try:
        essential_keys, group_class_dict = load_keys(N)
        
    except:
        W3_dict = W3.generate_W3_irrep_dict()
       

        group_class_dict = group_classes(N, W3_dict['T_1'])

        w3_subgroup = w3_cosets(W3_irreps=W3_dict)
        coset2_keys = mom_cosets(N=N)
        essential_cg_keys = w3_subgroup + coset2_keys

        essential_keys = []
        for key, val in group_class_dict.items():
            class_key_known = False

            for ele in val:
                if ele in essential_cg_keys:
                    class_key_known = True
                    break

            if not class_key_known:

                essential_keys.append(key)
        essential_keys += essential_cg_keys
        essential_keys = list(set(essential_keys))
        save_keys(essential_keys, group_class_dict, N)
    
    return essential_keys, group_class_dict

def trans_essential_rot_keys(essential_keys):
    trans_essential_rot_keys = {}
    for trans_rot in essential_keys:
        if trans_rot[0] not in trans_essential_rot_keys:
            trans_essential_rot_keys[trans_rot[0]] = [trans_rot[1],]
        else:
            trans_essential_rot_keys[trans_rot[0]].append(trans_rot[1])
            
    return trans_essential_rot_keys

########################################################################################################################################
########################################################################################################################################
############################################################### BOSONS ###############################################################


################################################################################################################################################ Generate the bosonic orbits, little groups and little group irreps


def bosonic_little_groups_and_orbits(mom_vec, W3_irreps):
    ### Takes input as momentum vector [px,py,pz] ]
    ### Outputs 1. dictionary of momentum little group 2. Momentum Irrep orbit 
    ### Dictionarys have form key = "group element label", value = "matrix of group element"
    
    final_mom_dict = {}
    mom_orbit = []
    
    W3_faithful = W3_irreps["T_1"]
    #print(W3)
        
    for label, matrix in W3_faithful.items():
        
        new_mom_vec = np.dot(matrix,mom_vec)
        mom_orbit.append(new_mom_vec)
            
        if np.all(new_mom_vec == mom_vec):

            final_mom_dict[label] = matrix
                

    return final_mom_dict, np.unique(mom_orbit,axis=0)


def generate_bosonic_little_group_irreps(faithful_little_group_matrix_dict, mom_key, W3_irrep_dict):
    ###############################################################################################################
    ### Returns a dictionary keyed by irrep labels, valued by irrep dictionarys
    
    classes = gf.conjugate_classes(faithful_little_group_matrix_dict)
    number_of_irreps = len(classes)
    group_order = len(faithful_little_group_matrix_dict)
    irrep_labels = ["A_", "E_", "T_"]
    
    ############################################################################################################
    #### Add missing irreps
    
    W3_irreps = W3_irrep_dict.copy()
    

    D4h_irreps = W3.generate_D4h_irreps(taste_key='taste_00p', W3_irreps=W3_irrep_dict)
    W3_irreps.update(D4h_irreps)

    #Z2_Z2_irreps = W3.generate_Z2_Z2_irreps(taste_key='taste_00p', W3_irreps)
    #W3_irreps.update(Z2_Z2_irreps)

    
    #############################################################################################################
    ##### Get list of dimensions of available irreps to check and also dimensions of required irreps using rep theory
    #####  \sum dim(rep_i)^2 = |G|, order of group
    
    W3_irreps_dimension = [int(np.trace(matrix_dict["E"])) if np.array([matrix_dict["E"]]).shape!=(1,) else 
                           matrix_dict["E"] for matrix_dict in W3_irreps.values()]
    
    
    if "I_S" in list(faithful_little_group_matrix_dict.keys()):
        irrep_dimensions = gf.irrep_dimension_function(number_of_irreps, group_order, True)
    else:
        irrep_dimensions = gf.irrep_dimension_function(number_of_irreps, group_order)
        
    number_of_unique_dimensions = len(np.unique(W3_irreps_dimension))
    
    ##################################################################################################################
    ### One counter counts the number of irreps obtained, the other counts the available irreps
    
    
    counter1 = np.zeros(number_of_unique_dimensions, dtype = int)
    counter2 = np.zeros(number_of_unique_dimensions, dtype = int)
    
    ###################################################################################################################
    ### Everything has the trivial irrep
    
    group_irrep_dict = {}
    group_irrep_dict["A_0"] = {key : W3_irreps["A_0"][key] for (key, val) in faithful_little_group_matrix_dict.items()}
    
    counter1[0]+=1
    counter2[0]+=1

    for dim in W3_irreps_dimension:
        index = dim-1
        irrep_label = irrep_labels[index] + str(counter1[index])
        if dim in irrep_dimensions:
            try:
                irrep_dict = {key : W3_irreps[irrep_label][key] for (key, val) in faithful_little_group_matrix_dict.items()}
            except:
                counter1[index]+= 1
                continue
            
           
            
            check_irrep = gf.check_irreducibility(irrep_dict)

            orthogonality = np.all([gf.check_orthogonality(irrep_dict, irreps) 
                                                             for irreps in group_irrep_dict.values()])
            if check_irrep == 1 and orthogonality:

                group_irrep_dict[irrep_labels[index] + str(counter2[index])] = irrep_dict
                counter2[index]+= 1
                irrep_dimensions.remove(dim)

            counter1[index]+= 1
    return group_irrep_dict


################################################################################################################################################ Generate the momentum cosets under  rotation little groups

def momentum_coset_representatives(W3_irrep_dict):
    ### Generates All coset representatives that rotate each mometum into the different elements of their orbits.

    mom_dict = {"mom000" : [0,0,0],
                "mom001" : [0,0,1],
                "mom110" : [1,1,0],
                "mom111" : [1,1,1],
                "mom012" : [0,1,2],
                "mom112" : [1,1,2],
                "mom123" : [1,2,3]}
    
    
    ### Generating all the orbits
    
    coset_dict = {'mom000': ['E']}
            
    for mom_key, mom_vec in mom_dict.items():
        if mom_key not in coset_dict.keys():
            orbit_list = []
            orbit_dict = {}

            mom_group, mom_orbit = bosonic_little_groups_and_orbits(mom_vec, W3_irrep_dict)
            
            cosets_right_imp = {}

            cosets_right = gf.right_cosets(W3_irrep_dict["T_1"], mom_group)
            
            for key, val in cosets_right.items():
                cosets_right_imp[sorted(val, key=len)[0]] = val

            coset_reps = []
            coset_mom_list = [np.array([0,0,0]),]
            for key in sorted(cosets_right_imp.keys(), key=len):

                I_S = False
                coset_rep = key                
                coset_mom =  mom_vec @ W3_irrep_dict["T_1"][coset_rep]

                if not any([np.all(coset_mom == i) for i in  coset_mom_list]):
                    coset_mom_list.append(coset_mom)
                    coset_reps.append(coset_rep)
                    if 'I_S' in coset_rep:
                        I_S = True

                    neg_mom = [-i for i in coset_mom]
                    if not np.any([np.all(neg_mom == i) for i in  coset_mom_list]):
                        for key1 in sorted(cosets_right_imp.keys(), key=len):
                            if np.all(mom_vec @ W3_irrep_dict["T_1"][key1] == neg_mom):
                                for coset_rep_neg in sorted(cosets_right_imp[key1], key=len):
                                    if I_S==True:
                                        if 'I_S' not in coset_rep_neg:
                                            coset_mom_list.append(neg_mom)
                                            coset_reps.append(coset_rep_neg)
                                            break
                                    else:
                                        if 'I_S' in coset_rep_neg:
                                            coset_mom_list.append(neg_mom)
                                            coset_reps.append(coset_rep_neg)
                                            break
                                break


            coset_dict[mom_key] = coset_reps
           
    return coset_dict



########################################################################################################################################
#### Functions to generate T \rtimes W3 induced irreps

def bosonic_induced_mom_orbit_array(T_element, T_orbit_vec, N, H_element, H_little_group_keys, H_coset_representative, 
                            H_little_group_irrep, faithful_w3_irrep, LG_irrep_dim, induced_dim):
    ### H "typically" is  W3, only need to check if the W3 component of h_k h h_j^(-1) is in 
    ### the W3 subgroup of the little group

    induced_orbit_arr = np.zeros((induced_dim, induced_dim), dtype = np.complex128)
    
    h = faithful_w3_irrep[H_element]
    for k, coset_rep1 in enumerate(H_coset_representative):
        
        row_block_index_lower = LG_irrep_dim*k
        row_block_index_upper = LG_irrep_dim*(k+1)
        
        h_k = faithful_w3_irrep[coset_rep1]
        
        rotated_T_orbit_vec = T_orbit_vec @ h_k
        rotated_T_irrep_char = tg.mom_character(rotated_T_orbit_vec, T_element, N)
        res_initial = h_k @ h

            
        for j, coset_rep2 in enumerate(H_coset_representative):

            column_block_index_lower = LG_irrep_dim*j
            column_block_index_upper = LG_irrep_dim*(j+1)

            h_j = faithful_w3_irrep[coset_rep2]

            res = res_initial @ inv(h_j)

            new_mat_label = gf.what_matrix(res, faithful_w3_irrep)

            if new_mat_label in H_little_group_keys:
                induced_orbit_arr[row_block_index_lower:row_block_index_upper, column_block_index_lower:column_block_index_upper] = rotated_T_irrep_char * H_little_group_irrep[new_mat_label]

    return induced_orbit_arr

def bosonic_induced_mom_orbit_array_1D(T_element, T_orbit_vec, N, H_element, H_little_group_keys, H_coset_representative, 
                            H_little_group_irrep, faithful_w3_irrep, induced_dim):
    ### H "typically" is \Gamma_4 \rtimes W3, only need to check if the W3 component of h_k h h_j^(-1) is in 
    ### the W3 subgroup of the little group
    induced_orbit_arr = np.zeros((induced_dim, induced_dim), dtype = np.complex128)
    
    h = faithful_w3_irrep[H_element]
    for k, coset_rep1 in enumerate(H_coset_representative):
        
        h_k = faithful_w3_irrep[coset_rep1]
        
        rotated_T_orbit_vec = T_orbit_vec @ h_k
        rotated_T_irrep_char = tg.mom_character(rotated_T_orbit_vec, T_element, N)
        res_initial = h_k @ h

        for j, coset_rep2 in enumerate(H_coset_representative):

            h_j = faithful_w3_irrep[coset_rep2]

            res = res_initial @ inv(h_j)

            new_mat_label = gf.what_matrix(res, faithful_w3_irrep)
            if new_mat_label in H_little_group_keys:

                induced_orbit_arr[k, j] = rotated_T_irrep_char * H_little_group_irrep[new_mat_label]
                           
    return induced_orbit_arr


def generate_bosonic_single_array(translation_element_label, trans_essential_keys, w3_irrep, translation_group, mom_vec, N, H_little_group_keys, H_coset_representative, W3_irrep_dict, LG_irrep_dim, induced_dim, Full=False):
    
    single_irrep_dict = {}
    
    if Full == False:
        key_list = trans_essential_keys[translation_element_label]
    else:
        key_list = W3_irrep_dict["T_1"].keys()
    
    for w3_element_label in key_list:
        
        induced_mom_array = bosonic_induced_mom_orbit_array(T_element = translation_group[translation_element_label], 
                                                        T_orbit_vec=mom_vec, N=N, H_element = w3_element_label, 
                                                        H_little_group_keys = H_little_group_keys, 
                                                        H_coset_representative = H_coset_representative, 
                                                        H_little_group_irrep = w3_irrep,
                                                        faithful_w3_irrep = W3_irrep_dict["T_1"],
                                                        LG_irrep_dim=LG_irrep_dim, induced_dim=induced_dim)


        single_irrep_dict[(translation_element_label, w3_element_label)] = induced_mom_array
            
    return single_irrep_dict

def generate_bosonic_single_array_1D(translation_element_label, trans_essential_keys, w3_irrep, translation_group, mom_vec, N, H_little_group_keys, H_coset_representative, W3_irrep_dict, induced_dim, Full=False):
    
    single_irrep_dict = {}
    
    if Full == False:
        key_list = trans_essential_keys[translation_element_label]
    else:
        key_list = W3_irrep_dict["T_1"].keys()
    
    for w3_element_label in key_list:     
        
        induced_mom_array = bosonic_induced_mom_orbit_array_1D(T_element = translation_group[translation_element_label], 
                                                        T_orbit_vec=mom_vec, N=N, H_element = w3_element_label, 
                                                        H_little_group_keys = H_little_group_keys, 
                                                        H_coset_representative = H_coset_representative, 
                                                        H_little_group_irrep = w3_irrep,
                                                        faithful_w3_irrep = W3_irrep_dict["T_1"],
                                                        induced_dim=induced_dim)


        single_irrep_dict[(translation_element_label, w3_element_label)] = induced_mom_array
            
    return single_irrep_dict


def momentum_induced_bosonic_representation(mom_key, N, dim, W3_irrep_dict, pool_no, Full=False):
    
    essential_keys, class_keys = essential_group_keys(N)
    
    trans_essential_keys = trans_essential_rot_keys(essential_keys)
    
    mom_dict = tg.mom_dictionary()
    
    translation_group = tg.generate_translation_group_elements(N, dim)

    mom_vec = mom_dict[mom_key]
    
    mom_group = bosonic_little_groups_and_orbits(mom_vec, W3_irrep_dict)[0]

    mom_coset_representative = momentum_coset_representatives(W3_irrep_dict=W3_irrep_dict)[mom_key]
    
    orbit_dim = len(mom_coset_representative)
   
    mom_specific_irreps = little_group_irrep_dicts = generate_bosonic_little_group_irreps(mom_group, mom_key, W3_irrep_dict)
    induced_mom_irreps_dict = {}
    
    completed_irreps = check_irreps(mom_key=mom_key, N=N)
    
    for w3_irrep_label, w3_irrep in mom_specific_irreps.items():
        irrep_label = mom_key + "_" + w3_irrep_label
        if irrep_label not in completed_irreps:
            print(irrep_label)

            if np.array([list(w3_irrep.values())[0]]).shape !=(1,):
                LG_irrep_dim = list(w3_irrep.values())[0].shape[0]
                
                induced_dim = orbit_dim * LG_irrep_dim

                func = partial(generate_bosonic_single_array, trans_essential_keys=trans_essential_keys,
                               w3_irrep=w3_irrep, 
                            translation_group=translation_group, mom_vec=mom_vec, N=N, H_little_group_keys=list(mom_group.keys()),
                            H_coset_representative=mom_coset_representative, 
                            W3_irrep_dict=W3_irrep_dict, 
                            LG_irrep_dim=LG_irrep_dim, induced_dim=induced_dim, Full=Full)
                
            else:

                induced_dim = orbit_dim

                func = partial(generate_bosonic_single_array_1D, trans_essential_keys=trans_essential_keys,
                               w3_irrep=w3_irrep,
                                translation_group=translation_group, mom_vec=mom_vec, N=N, H_little_group_keys=list(mom_group.keys()),
                                H_coset_representative=mom_coset_representative, 
                                W3_irrep_dict=W3_irrep_dict, induced_dim=induced_dim, Full=Full)


            #p = Pool(pool_no)
            #single_irrep_list = p.map(func, translation_group.keys())
            single_irrep_list = list(map(func, translation_group.keys()))
            #p.terminate()

            single_irrep_dict = dict(ChainMap(*single_irrep_list))
            
            
            save_irrep(mom_key=mom_key, irrep_label=irrep_label, N=N, irrep=single_irrep_dict)

            del single_irrep_dict
                       
    return None

#Usage

'''
W3_irreps = W3.generate_W3_irrep_dict()
N=3
Full=False
sg.momentum_induced_bosonic_representation("mom000", N, dim=3, W3_irrep_dict=W3_irreps, pool_no=6, Full=Full)
sg.momentum_induced_bosonic_representation("mom001", N, dim=3, W3_irrep_dict=W3_irreps, pool_no=6, Full=Full)
sg.momentum_induced_bosonic_representation("mom110", N, dim=3, W3_irrep_dict=W3_irreps, pool_no=6, Full=Full)
sg.momentum_induced_bosonic_representation("mom111", N, dim=3, W3_irrep_dict=W3_irreps, pool_no=6, Full=Full)

mom000 = ng.load_irrep("mom000", N=N)
mom001 = ng.load_irrep("mom001", N=N)
mom110 = ng.load_irrep("mom110", N=N)
mom111 = ng.load_irrep("mom111", N=N)
total_mom_dict = {**mom000, **mom001, **mom110 , **mom111}

keys, class_dict = ng.essential_group_keys(N=N)
#class_dict = False

for irrep1name, irrep1 in total_mom_dict.items():
    for irrep2name, irrep2 in total_mom_dict.items():
        if gf.check_orthogonality_v2(matrix_dict1=irrep1, matrix_dict2=irrep2, class_dictionary = class_dict) == 1:
            print('same irrep')
            print(irrep1name, irrep2name)
        elif gf.check_orthogonality_v2(matrix_dict1=irrep1, matrix_dict2=irrep2, class_dictionary = class_dict) != 0:
            print('something wrong')
            print(irrep1name, irrep2name)
            print(gf.check_orthogonality_v2(matrix_dict1=irrep1, matrix_dict2=irrep2, class_dictionary = class_dict))
'''

########################################################################################################################################
########################################################################################################################################
############################################################### Fermions ###############################################################


################################################################################################################################################ Generate the fermionic orbits, little groups and little group irreps

#Copy from stag group later...

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
#### Functions to save, load and check which irreps are completeted 

def save_irrep(mom_key, irrep_label, N, irrep):
    
    directory = "./naive_irreps/N_"  + str(N) + "/" + mom_key + "/" 
    #directory = "C:\\Users\\Shaun252\\Desktop\\staggered_irreps\\N_"  + str(N) + "\\" + mom_key + "\\"  
    #directory = "staggered_irreps/"+ mom_key + "/"  + str(N) + "/"
    filename = directory + irrep_label
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.save(filename, irrep)
    print("Irrep: " + str(irrep_label) + " saved")
    
def load_irrep(mom_key, N, irrep_list=False):
    mom_irrep_dict = {}
    directory = "./naive_irreps/N_"  + str(N) + "/" + mom_key + "/" 
    #directory = "C:\\Users\\Shaun252\\Desktop\\staggered_irreps\\N_"  + str(N) + "\\" + mom_key + "\\"  
    #directory = "staggered_irreps/"+ mom_key + "/"  + str(N) + "/"
    for root, dirs, files in os.walk(directory, topdown=False):

        for name in files:
            
            if irrep_list == False:
                print(name.rstrip(".npy"))
                irrep_dict_arr = np.load(os.path.abspath(os.path.join(root, name)), allow_pickle=True)
                irrep_dict  = irrep_dict_arr[()]
                mom_irrep_dict[name.rstrip(".npy")] = irrep_dict
                
            else:
                if name.rstrip(".npy") in irrep_list:
                    print(name.rstrip(".npy"))
                    irrep_dict_arr = np.load(os.path.abspath(os.path.join(root, name)), allow_pickle=True)
                    irrep_dict  = irrep_dict_arr[()]
                    mom_irrep_dict[name.rstrip(".npy")] = irrep_dict
                    
                        
    return mom_irrep_dict

def check_irreps(mom_key, N):
    irrep_list = []
    directory = "./naive_irreps/N_"  + str(N) + "/" + mom_key + "/" 
    #directory = "C:\\Users\\Shaun252\\Desktop\\staggered_irreps\\N_"  + str(N) + "\\" + mom_key + "\\"  
    #directory = "staggered_irreps/"+ mom_key + "/"  + str(N) + "/" 
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            irrep_list.append(name.rstrip(".npy"))
            
    return irrep_list