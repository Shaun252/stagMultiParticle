import numpy as np
import os
import pickle
from numpy.linalg import inv, matrix_power
from functools import partial
from multiprocessing import Pool
from collections import ChainMap

import gfunctions as gf
import translationGroup as tg
import cliff41Group as c41
import W3RotationGroup as W3


################################################################################################################################################ Generate the staggered group elements


def stag_group_elements(N, W3, cliff_4_1):
    translation_group = tg.generate_translation_group_elements(N=N, dim=3)
    
    stag_group_keys = []
    for key_rot in W3.keys():
        for key_gamma in cliff_4_1.keys():
            for key_trans in translation_group.keys():
                stag_group_keys.append((key_trans, key_rot ,key_gamma))
   
    return stag_group_keys

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

def gamma_rotation_commutation_relations(gamma, rotation):
    #### Encodes gamma_i R_ij  = R_ij gamma_j,  
    ####         gamma_j R_ij  = -R_ij gamma_i
    ####         gamma_k R_ij  = R_ij gamma_k where k can = 0
    #### The extra 0 / 1 is if the result is positive or negative
    
    gamma_digit = gamma[-1]
    rot_digits = rotation[-2:]
    
    if gamma_digit == rot_digits[0]:

        gamma_digit = rot_digits[1]
        return "gamma_"+ gamma_digit[0], 0
        
    elif gamma_digit == rot_digits[1]:
        
        gamma_digit = rot_digits[0]
        return  "gamma_"+ gamma_digit, 1
    
    else:
        return gamma, 0
    
    

def gamma_parity_commutation_relations(gamma):
    #### Encodes gamma_i I_S  = - I_S gamma_j,  
    ####         gamma_0 I_S  = I_S gamma_0
    
    gamma_digit = gamma[-1]
    
    if gamma_digit != '0':

        return gamma, 1       
    else:
        
        return gamma, 0

        
def general_gamma_commutation(general_gamma_key, general_rotation_key, faithful_clifford_irrep):
    #### Commutes gamma_x gamma_y ... gamma_z R_ij R_kl -> R_ij R_kl gamma_x' gamma_y' ... gamma_z' 
    
    rot_base = np.identity(3)
    gamma_base = np.identity(4)
    full_rotation_list = []

    overall_minus_power = general_gamma_key.count("-")
    general_gamma_key = general_gamma_key.replace("-", "")
    
    general_gamma_list = general_gamma_key.split("*")
    general_rotation_list = general_rotation_key.split("*")
    
    for gamma in general_gamma_list[::-1]:

        if gamma != 'C_0' and gamma != 'E':
            
            for rotation in general_rotation_list:

                if 'R' in rotation:

                    gamma, minus_sign = gamma_rotation_commutation_relations(gamma, rotation)
                    overall_minus_power += minus_sign

                elif rotation == "I_S":
                    gamma, minus_sign = gamma_parity_commutation_relations(gamma)
                    overall_minus_power += minus_sign
                
        ###### Taking the product of the gamma matrices as we go
        gamma_matrix = faithful_clifford_irrep[gamma]
        gamma_base = gamma_matrix @ gamma_base

     
    gamma_label = gf.what_matrix(gamma_base, faithful_clifford_irrep)

    if overall_minus_power % 2 ==1:
        #### Determining the overall "-" sign and adding it to the Gamma_4 part of the key
        if "-" in gamma_label:
            gamma_label = gamma_label.replace("-" , "")
        
        else:
            gamma_label = "-" + gamma_label
        
    return gamma_label


################################################################################################################################################ function to conjugate an element of the full staggered group by another element


def conjugate_stag_element(conj_ele, group_ele, N, W3, clifford_group):
    #Computing (R'' gamma'' t'') = (R' gamma' t') (R gamma t) (R' gamma' t')^-1
    # = R' gamma' t' R gamma t t'^-1  gamma'^-1 R'^-1
    
    conj_ele_trans = conj_ele[0]
    conj_ele_gamma = conj_ele[2]
    conj_ele_rot = conj_ele[1]
    
    conj_ele_trans_inv = np.array([N,N,N]) - conj_ele_trans
    conj_ele_gamma_inv =  gf.what_matrix(inv(clifford_group[conj_ele_gamma]), clifford_group)
    conj_ele_rot_inv =  gf.what_matrix(inv(W3[conj_ele_rot]), W3)
    
    group_ele_trans = group_ele[0]
    group_ele_gamma = group_ele[2]
    group_ele_rot = group_ele[1]
    
    #Consolidate  translations and gammas that are beside eachother, trans and gammas commute!
    total_trans_group_conj_inv = group_ele_trans + conj_ele_trans_inv
    total_gamma_group_conj_inv = clifford_group[group_ele_gamma] @ clifford_group[conj_ele_gamma_inv]
    
    # Bring W3 'group' element accross conj gamma and conj translation
    conj_trans_new = np.copy(conj_ele_trans)
    conj_gamma_new = general_gamma_commutation(general_gamma_key=conj_ele_gamma, general_rotation_key=group_ele_rot, faithful_clifford_irrep=clifford_group)
    
    for rot_ele in group_ele_rot.split('*'):

            conj_trans_new = translation_rotation_commutation(trans_ele=conj_trans_new, rotation=rot_ele, N=N)
            
    
    #Consolidate translations and gammas that are beside eachother again
    
    total_trans_conj_group_conj_inv = np.mod(conj_trans_new + total_trans_group_conj_inv, N)
    total_gamma_conj_group_conj_inv = gf.what_matrix(clifford_group[conj_gamma_new] @ total_gamma_group_conj_inv, clifford_group)
    
    
    #Bring accross conj inv rotation across consolidated gammas
    new_total_trans_conj_group_conj_inv = total_trans_conj_group_conj_inv
    for rot_ele in conj_ele_rot_inv.split('*'):
    
        new_total_trans_conj_group_conj_inv = translation_rotation_commutation(trans_ele=new_total_trans_conj_group_conj_inv, rotation=rot_ele, N=N)

    
    new_total_gamma_conj_group_conj_inv = general_gamma_commutation(general_gamma_key=total_gamma_conj_group_conj_inv, general_rotation_key=conj_ele_rot_inv, faithful_clifford_irrep=clifford_group)

    
    new_total_rot_conj_group_conj_inv = gf.what_matrix(W3[conj_ele_rot] @ W3[group_ele_rot] @ W3[conj_ele_rot_inv], W3)
    
    
    return (tuple(new_total_trans_conj_group_conj_inv), new_total_rot_conj_group_conj_inv, new_total_gamma_conj_group_conj_inv)




################################################################################################################################################ Generate the staggered group classes

def key_len(key):
    if type(key) != type((1,1)):
        return len(key)
    else:
        basekey = ''
        for subkey in key:
            basekey+= str(subkey)
        
        return len(basekey)
    

def stagg_group_classes(N, W3, cliff_4_1):
    
    stag_keys = stag_group_elements(N=N, W3=W3, cliff_4_1=cliff_4_1)
    
    directory = './staggered_irreps/keys/'
    filename_classes_temp = 'translation_' +str(N) + '_gamma_4_1_W3_classes_temp'
    filename_classes_temp_list = 'translation_' +str(N) + '_gamma_4_1_W3_classes_temp_list'

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
    
    for key1 in stag_keys:
        
        conjugate_list = []
        conjugate_string_list = []
        print(key1)
        if str(key1) not in total_list:
            print(True)
            i+=1
            conjugate_list.append(key1)
            conjugate_string_list.append(str(key1))
            for key2 in stag_keys:
                conjugate_label = conjugate_stag_element(conj_ele=key2, group_ele=key1, N=N, W3=W3, clifford_group=cliff_4_1)
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


################################################################################################################################################ Generate the staggered group W3xGamma subgroup and the coset keys of TxW3xGamma / W3xGamma. Also subgroup W3 and coset keys for W3xGamma / W3. These are used for calculating clebsch gordons

def taste_w3_cosets(W3_irreps, clifford):
    
    coset_keys = []
    for key in clifford['gamma'].keys():
        coset_keys.append(((0,0,0), "E", key))

    subgroup_keys = []
    for key, vals in W3_irreps["T_1"].items():
        subgroup_keys.append(((0,0,0), key, "E"))
        
    return coset_keys, subgroup_keys

def staggered_mom_cosets(N):
    translation_group = tg.generate_translation_group_elements(N, dim=3)
    coset_keys = []
    for key in translation_group.keys():
        coset_keys.append((key, "E", "E"))
      
    return coset_keys



################################################################################################################################################ Generate the essential keys need to take direct products and calculate cgs, they are the class represnetatives + the subgroup, cosets above

def save_keys(essential_keys, stagg_group_class_dict, N):
    
    directory = './staggered_irreps/keys/'
    filename = 'translation_' +str(N) + '_gamma_4_1_W3'
    filename_classes = 'translation_' +str(N) + '_gamma_4_1_W3_classes'
 
        
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    full_path = directory+ filename
    full_path_classes = directory+ filename_classes
    f = open(full_path,"wb")
    pickle.dump(essential_keys,f)
    f.close()
    
    f = open(full_path_classes,"wb")
    pickle.dump(stagg_group_class_dict,f)
    f.close()
    
    return None


def load_keys(N):
    
    directory = './staggered_irreps/keys/'
    filename = 'translation_' +str(N) + '_gamma_4_1_W3'
    filename_classes = 'translation_' +str(N) + '_gamma_4_1_W3_classes'
    
    full_path = directory+ filename
    full_path_classes = directory+ filename_classes
    
    with open(full_path, 'rb') as pfile:
    
        f = pickle.load(pfile)
        
    with open(full_path_classes, 'rb') as pfile:
    
        g = pickle.load(pfile)
    
    return f, g
    
    
def essential_group_keys(N):
    
    try:
        essential_keys, stagg_group_class_dict = load_keys(N)
        
    except:
        W3_dict = W3.generate_W3_irrep_dict()

        cliff_4_1 = c41.generate_clifford_4_1_complete_irrep_dict()
       

        stagg_group_class_dict = stagg_group_classes(N, W3_dict['T_1'], cliff_4_1['gamma'])

        coset1_keys, w3_subgroup = taste_w3_cosets(W3_irreps=W3_dict, clifford=cliff_4_1)
        coset2_keys = staggered_mom_cosets(N=N)
        essential_cg_keys = coset1_keys + w3_subgroup + coset2_keys

        essential_keys = []
        for key, val in stagg_group_class_dict.items():
            class_key_known = False

            for ele in val:
                if ele in essential_cg_keys:
                    class_key_known = True
                    break

            if not class_key_known:

                essential_keys.append(key)
        essential_keys += essential_cg_keys
        essential_keys = list(set(essential_keys))
        save_keys(essential_keys, stagg_group_class_dict, N)
    
    return essential_keys, stagg_group_class_dict

def trans_essential_rot_cliff_keys(essential_keys):
    trans_essential_rot_cliff_keys = {}
    for trans_cliff_rot in essential_keys:
        if trans_cliff_rot[0] not in trans_essential_rot_cliff_keys:
            trans_essential_rot_cliff_keys[trans_cliff_rot[0]] = [(trans_cliff_rot[1], trans_cliff_rot[2]),]
        else:
            trans_essential_rot_cliff_keys[trans_cliff_rot[0]].append((trans_cliff_rot[1], trans_cliff_rot[2]))
            
    return trans_essential_rot_cliff_keys

########################################################################################################################################
########################################################################################################################################
############################################################### BOSONS ###############################################################


################################################################################################################################################ Generate the bosonic orbits, little groups and little group irreps


def bosonic_little_groups_and_orbits(mom_vec, taste_vec, W3_irreps):
    ### Takes input as momentum vector [px,py,pz] and taste vector [pi/0,pi/0,pi/0]
    ### Outputs 1. dictionary of momentum little group 2. Momentum Irrep orbit 
    ### 3. dictionary of taste little group 4. taste orbit
    ### Dictionarys have form key = "group element label", value = "matrix of group element"
    
    final_mom_dict = {}
    final_taste_dict = {}
    mom_orbit = []
    taste_orbit = []
    
    W3_faithful = W3_irreps["T_1"]
    #print(W3)
        
    for label, matrix in W3_faithful.items():
        
        new_mom_vec = np.dot(matrix,mom_vec)
        mom_orbit.append(new_mom_vec)
            
        if np.all(new_mom_vec == mom_vec):

            final_mom_dict[label] = matrix
    
            new_taste_vec = np.mod(np.dot(matrix,taste_vec), 2*np.pi)
            taste_orbit.append(new_taste_vec)
            if np.all(new_taste_vec == taste_vec):
                final_taste_dict[label] = matrix
                

    return final_mom_dict, np.unique(mom_orbit,axis=0), final_taste_dict, np.unique(taste_orbit,axis=0)


def generate_bosonic_little_group_irreps(faithful_little_group_matrix_dict, mom_key, taste_key, W3_irrep_dict):
    ###############################################################################################################
    ### Returns a dictionary keyed by irrep labels, valued by irrep dictionarys
    
    classes = gf.conjugate_classes(faithful_little_group_matrix_dict)
    number_of_irreps = len(classes)
    group_order = len(faithful_little_group_matrix_dict)
    irrep_labels = ["A_", "E_", "T_"]
    
    ############################################################################################################
    #### Add missing irreps
    
    W3_irreps = W3_irrep_dict.copy()
    
    if mom_key == 'mom000' and taste_key != 'taste_000' and taste_key != 'taste_ppp':
    
        D4h_irreps = W3.generate_D4h_irreps(taste_key=taste_key, W3_irreps=W3_irrep_dict)
        W3_irreps.update(D4h_irreps)
    
    if mom_key == 'mom001':
        
        D4h_irreps = W3.generate_D4h_irreps(taste_key='taste_00p', W3_irreps=W3_irrep_dict)
        W3_irreps.update(D4h_irreps)
    
        Z2_Z2_irreps = W3.generate_Z2_Z2_irreps(taste_key, W3_irreps)
        W3_irreps.update(Z2_Z2_irreps)

    
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


################################################################################################################################################ Generate the momentum cosets under taste x rotation little groups
#'''
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
    
    coset_dict = {'mom000': ['E']}#,
                 #'mom001' : ['E', 'R___31*R___31', 'R__23', 'R___31*R_12', 'R___31', 'R__23*R__23*R__23*R_12']}
    
            
    for mom_key, mom_vec in mom_dict.items():
        if mom_key not in coset_dict.keys():
            orbit_list = []
            orbit_dict = {}

            mom_group, mom_orbit, taste_group, taste_orbit = bosonic_little_groups_and_orbits(mom_vec, [0,0,0], W3_irrep_dict)
            
            cosets_right_imp = {}

            cosets_right = gf.right_cosets(W3_irrep_dict["T_1"], mom_group)
            #print(mom_key)
            #print(cosets_right)
            #print('')
            
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

################################################################################################################################################ Generate the bosonic taste cosets under rotation 
#'''
def bosonic_taste_coset_representatives(W3_irrep_dict):
    ### Generates All coset representatives that rotate each taste into the different elements of their orbits. I set hierarchy
    ### firstly by z taste direction, so pp0, 00p labels the triplet tastes {pp0, p0p, 0pp} & {00p, 0p0, p00}
    ### then when I have momentum with z singleout 001, 110 I set hierarchy to be 0px
    
    pi = np.pi
    mom_dict = {"mom000" : [0,0,0],
                "mom001" : [0,0,1],
                "mom110" : [1,1,0],
                "mom111" : [1,1,1],
                "mom012" : [0,1,2],
                "mom112" : [1,1,2],
                "mom123" : [1,2,3]}

    taste_dict = c41.taste_dictionary()
    
    
    
    ### Generating all the orbits
    
    coset_dict = {}
    
    
    for mom_key, mom_vec in mom_dict.items():
        orbit_list = []
        orbit_dict = {}

        if mom_key == "mom000" or mom_key == "mom111" or mom_key == "mom012" or  mom_key == "mom123":
            ### "mom012" or "mom123" don't really matter in this if statement they can be put in the next one also
            ### as they dont have any orbits > 1. These if statements are just about getting orbit representative labels
            ### correct

            for taste_key, taste_vec in taste_dict.items():

                mom_group, mom_orbit, taste_group, taste_orbit = bosonic_little_groups_and_orbits(mom_vec, taste_vec, W3_irrep_dict)



                ####Fixing labelling so we only get 00p or pp0 to represent the orbits they are in
                if np.any([np.array_equal([0,0,pi], orbital_elements) for orbital_elements in taste_orbit]):
                    label = mom_key+"_"+"taste_00p"

                elif np.any([np.array_equal([pi,pi,0], orbital_elements) for orbital_elements in taste_orbit]):
                    label = mom_key+"_"+"taste_pp0"

                else:
                    label = mom_key+"_"+taste_key


                if np.all([False if np.array_equal(taste_orbit, orbit_elements) else True for orbit_elements in orbit_list]):
                    orbit_dict[label] = taste_orbit
                    orbit_list.append(taste_orbit)

        ### Now that we have orbits we must careful to choosen the orbit represnative we want [0,0, pi] and [pi,pi,0]
        ### i.e the z component is the focus like with momentum (the groups have to coincide!)        

        for key1, val1 in orbit_dict.items():
            if np.any([np.array_equal([0,0,pi], orbital_elements) for orbital_elements in val1]):
                group = bosonic_little_groups_and_orbits(mom_vec, [0,0,pi], W3_irrep_dict)[2]

            elif np.any([np.array_equal([pi,pi,0], orbital_elements) for orbital_elements in val1]):
                group = bosonic_little_groups_and_orbits(mom_vec, [pi,pi,0], W3_irrep_dict)[2]

            else:
                group = bosonic_little_groups_and_orbits(mom_vec, val1[0], W3_irrep_dict)[2]

            
            cosets = gf.right_cosets(mom_group, group)
            ### Elements with I_S do rotate the opposite way to the normal coset elements?
            coset_reps = [sorted([coset_rep for coset_rep in val_coset if "I_S" not in coset_rep or len(val_coset) == 1], key=len)[0] 
                          for val_coset in cosets.values()]
            
            #print(mom_key, key1)
            #print(cosets)
            #print('')
            coset_dict[key1] = sorted(coset_reps, key=len)

    for mom_key, mom_vec in mom_dict.items():
        orbit_list = []
        orbit_dict = {}

        if mom_key == "mom001" or mom_key == "mom110" or mom_key == "mom112":

            for taste_key, taste_vec in taste_dict.items():

                mom_group, mom_orbit, taste_group, taste_orbit = bosonic_little_groups_and_orbits(mom_vec, taste_vec, W3_irrep_dict)


                ####Fixing labelling so we only get 0pXX to represent the orbits, third component is now a parity
                if np.any([np.array_equal([0,pi,pi], orbital_elements) for orbital_elements in taste_orbit]):
                    label = mom_key+"_"+"taste_0pp"

                elif np.any([np.array_equal([0,pi,0], orbital_elements) for orbital_elements in taste_orbit]):
                    label = mom_key+"_"+"taste_0p0"

                else:
                    label = mom_key+"_"+taste_key

                if np.all([False if np.array_equal(taste_orbit, orbit_elements) else True for orbit_elements in orbit_list]):
                    orbit_dict[label] = taste_orbit
                    orbit_list.append(taste_orbit)

        ### Now that we have orbits we must careful to choosen the orbit represnative we want [0,0, pi] and [pi,pi,0]
        ### i.e the z component is the focus like with momentum (the groups have to coincide!)        

        for key1, val1 in orbit_dict.items():
            if np.any([np.array_equal([0,pi,pi], orbital_elements) for orbital_elements in val1]):
                group = bosonic_little_groups_and_orbits(mom_vec, [0,pi,pi], W3_irrep_dict)[2]

            elif np.any([np.array_equal([0,pi,0], orbital_elements) for orbital_elements in val1]):
                group = bosonic_little_groups_and_orbits(mom_vec, [0,pi,0], W3_irrep_dict)[2]

            else:
                group = bosonic_little_groups_and_orbits(mom_vec, val1[0], W3_irrep_dict)[2]


            cosets = gf.right_cosets(mom_group, group)
            #print(mom_key, key1)
            #print(cosets)
            #print('')

            ### Elements with I_S do rotate the opposite way to the normal coset elements?
            coset_reps = [sorted([coset_rep for coset_rep in val_coset], key=len)[0] 
                          for val_coset in cosets.values()]

            coset_dict[key1] = sorted(coset_reps, key=len)
           
    return coset_dict


########################################################################################################################################
#### Functions to generate \Gamma_4 \rtimes W3 induced irreps


def bosonic_induced_taste_orbit_array(W3_group_element, LG_irrep, LG_irrep_dim, clifford_group_element, coset_representative, 
                              taste_vec, taste4, charge_conj, little_group, W3_dict, clifford_dict):
    #### See rep theory notes X^{\tilde h_k (\sigma)} = e^{i*2*\pi*(D^{3d}(\tilde h_k) p . clifford_group_element}
    #### Where D^{3d} is the faithful irrep that acts on 3 vectors
    #### Generates a single element of the induced irrep

        
    orbit_dim = len(coset_representative)
    induced_dim = orbit_dim * LG_irrep_dim
    induced_orbit_arr = np.zeros((induced_dim, induced_dim),  dtype = np.complex128)

    W3_faithful_dict = W3_dict["T_1"]

    h = W3_faithful_dict[W3_group_element]

    for k, coset_rep1 in enumerate(coset_representative):
        
        row_block_index_lower = LG_irrep_dim*k
        row_block_index_upper = LG_irrep_dim*(k+1)

        h_k = W3_faithful_dict[coset_rep1]

        taste_irrep_orbit_element = np.mod(taste_vec @ h_k, 2*np.pi)

        res_initial = h_k @ h

        taste_irrep_orbit_element = np.append(charge_conj,np.append(taste4, taste_irrep_orbit_element))

        taste_char = clifford_dict[tuple(taste_irrep_orbit_element)][clifford_group_element]

        for j, coset_rep2 in enumerate(coset_representative):
            
            column_block_index_lower = LG_irrep_dim*j
            column_block_index_upper = LG_irrep_dim*(j+1)           

            h_j = W3_faithful_dict[coset_rep2]

            res = res_initial @ inv(h_j)

            new_mat_label = gf.what_matrix(res, little_group)

            if new_mat_label != None:

                induced_orbit_arr[row_block_index_lower:row_block_index_upper, column_block_index_lower:column_block_index_upper] = taste_char * LG_irrep[new_mat_label]

    return induced_orbit_arr
   
def bosonic_induced_taste_orbit_array_1D(W3_group_element, LG_irrep, clifford_group_element, coset_representative, 
                              taste_vec, taste4, charge_conj, little_group, W3_dict, clifford_dict):
    #### See rep theory notes X^{\tilde h_k (\sigma)} = e^{i*2*\pi*(D^{3d}(\tilde h_k) p . clifford_group_element}
    #### Where D^{3d} is the faithful irrep that acts on 3 vectors
    #### Generates a single element of the induced irrep

        
    orbit_dim = len(coset_representative)
    induced_dim = orbit_dim
    induced_orbit_arr = np.zeros((induced_dim, induced_dim),  dtype = np.complex128)

    W3_faithful_dict = W3_dict["T_1"]

    h = W3_faithful_dict[W3_group_element]

    for k, coset_rep1 in enumerate(coset_representative):
        
        

        h_k = W3_faithful_dict[coset_rep1]

        taste_irrep_orbit_element = np.mod(taste_vec @ h_k, 2*np.pi)

        res_initial = h_k @ h

        taste_irrep_orbit_element = np.append(charge_conj,np.append(taste4, taste_irrep_orbit_element))

        taste_char = clifford_dict[tuple(taste_irrep_orbit_element)][clifford_group_element]

        for j, coset_rep2 in enumerate(coset_representative):
          
            h_j = W3_faithful_dict[coset_rep2]

            res = res_initial @ inv(h_j)

            new_mat_label = gf.what_matrix(res, little_group)


            if new_mat_label != None:
                induced_orbit_arr[k, j] = taste_char * LG_irrep[new_mat_label]
    
    return induced_orbit_arr


def bosonic_taste_induced_representations(taste_key, mom_key, W3_irrep_dict, clifford_irrep_dict):
    ##### Generates all irreps for a specifc momentum and taste (the orbit x the irreps of the little group, induced)
    #### Note for mom001, I really should have used 2d vectors for the taste since only the first two components decide the taste orbits
    #### so my labelling for irreps is mixed. i.e I have taste irreps labelled by taste_p0XX and taste_0pYY where if XX=YY, this is the
    #### same irrep. I never have XX=YY but I do have irreps labelled taste_p00p(=taste_0p0p) and taste_0p00(=taste_p000) etc, "maybe I 
    #### should all relabel to 0p?
    #### This is just irreps for momentum little groups
    
    
    mom_dict = tg.mom_dictionary()
    taste_dict = c41.taste_dictionary()
    
    mom_vec = mom_dict[mom_key]
    taste_vec = taste_dict[taste_key]
    
    mom_group, mom_orbit, taste_group, taste_orbit = bosonic_little_groups_and_orbits(mom_vec, taste_vec, W3_irrep_dict) 
    
    little_group_irrep_dicts = generate_bosonic_little_group_irreps(taste_group, mom_key, taste_key, W3_irrep_dict)
    
    coset_representative = bosonic_taste_coset_representatives(W3_irrep_dict)[mom_key+"_"+taste_key]
    
    induced_taste_irreps_dict = {}
    
    taste3_label = taste_key[-3:]
    
    for irrep_label, single_irrep_dict in little_group_irrep_dicts.items():
        
        for charge_conj_key, charge_conj in {"0":0,"p":np.pi}.items():
        
            for taste_4_key, taste_4 in {"0":0,"p":np.pi}.items():

                taste_label = charge_conj_key + taste_4_key + taste3_label

                induced_taste_irrep = {}

                if np.array([list(single_irrep_dict.values())[0]]).shape !=(1,):
                    LG_irrep_dim = list(single_irrep_dict.values())[0].shape[0]

                    for irrep_matrix_label, irrep_matrix in mom_group.items():

                        for taste_group_element in clifford_irrep_dict['gamma'].keys():

                            induced_element = bosonic_induced_taste_orbit_array(W3_group_element=irrep_matrix_label, 
                                                                        LG_irrep=single_irrep_dict, LG_irrep_dim=LG_irrep_dim,
                                                                        clifford_group_element=taste_group_element, 
                                                                        coset_representative=coset_representative, 
                                                                        taste_vec=taste_vec, 
                                                                        taste4=taste_4,
                                                                        charge_conj=charge_conj,
                                                                        little_group=taste_group, 
                                                                        W3_dict=W3_irrep_dict, 
                                                                        clifford_dict=clifford_irrep_dict)

                            induced_taste_irrep[(irrep_matrix_label, taste_group_element)] = induced_element

                else:
                    

                    for irrep_matrix_label, irrep_matrix in mom_group.items():

                        for taste_group_element in clifford_irrep_dict['gamma'].keys():

                            induced_element = bosonic_induced_taste_orbit_array_1D(W3_group_element=irrep_matrix_label, 
                                                                        LG_irrep=single_irrep_dict,
                                                                        clifford_group_element=taste_group_element, 
                                                                        coset_representative=coset_representative, 
                                                                        taste_vec=taste_vec, 
                                                                        taste4=taste_4,
                                                                        charge_conj=charge_conj,
                                                                        little_group=taste_group, 
                                                                        W3_dict=W3_irrep_dict, 
                                                                        clifford_dict=clifford_irrep_dict)

                            induced_taste_irrep[(irrep_matrix_label, taste_group_element)] = induced_element

                induced_taste_irreps_dict[mom_key + "_" + 'taste_' + taste_label  + "_" + irrep_label] = induced_taste_irrep

    return induced_taste_irreps_dict


def generate_all_bosonic_momentum_specific_taste_irreps(mom_key, W3_irrep_dict, clifford_irrep_dict, pool_no):
    #### Generates all irreps for a specific momentum (all taste orbits x all little group irreps, induced)

    mom_dict = tg.mom_dictionary()
    
    mom_vec = mom_dict[mom_key]

    induced_zero_mom_irreps_dict = {}
    
    taste_coset_representative = bosonic_taste_coset_representatives(W3_irrep_dict)
            
    taste_array = [coset_key.lstrip(mom_key+"_") for coset_key in taste_coset_representative.keys() if mom_key in coset_key]
    
    func = partial(bosonic_taste_induced_representations, mom_key=mom_key, W3_irrep_dict=W3_irrep_dict, clifford_irrep_dict=clifford_irrep_dict)

    #p = Pool(pool_no)
    #irrep_list = p.map(func, taste_array)
    irrep_list = list(map(func, taste_array))
    #p.terminate()
    
    #final_dict = {k: irrep_list[0][k] for k in list(irrep_list[0].keys())[:4]}
    
    final_dict = dict(ChainMap(*irrep_list))
    
    return final_dict

########################################################################################################################################
#### Functions to generate T \rtimes \Gamma_4 \rtimes W3 induced irreps

def bosonic_induced_mom_orbit_array(T_element, T_orbit_vec, N, H_element, H_little_group_keys, H_coset_representative, 
                            H_little_group_irrep, faithful_H_dict, faithful_w3_irrep, LG_irrep_dim, induced_dim):
    ### H "typically" is \Gamma_4 \rtimes W3, only need to check if the W3 component of h_k h h_j^(-1) is in 
    ### the W3 subgroup of the little group

    induced_orbit_arr = np.zeros((induced_dim, induced_dim), dtype = np.complex128)
    
    h = faithful_H_dict[H_element]
    for k, coset_rep1 in enumerate(H_coset_representative):
        
        row_block_index_lower = LG_irrep_dim*k
        row_block_index_upper = LG_irrep_dim*(k+1)
        
        h_k = faithful_H_dict[(coset_rep1, "E")]
        h_k_w3 = faithful_w3_irrep[coset_rep1]
        
        rotated_T_orbit_vec = T_orbit_vec @ h_k_w3
        rotated_T_irrep_char = tg.mom_character(rotated_T_orbit_vec, T_element, N)
        res_initial = h_k @ h

            
        for j, coset_rep2 in enumerate(H_coset_representative):

            column_block_index_lower = LG_irrep_dim*j
            column_block_index_upper = LG_irrep_dim*(j+1)

            h_j = faithful_H_dict[(coset_rep2, "E")]

            res = res_initial @ inv(h_j)

            new_mat_label = gf.what_matrix(res, faithful_H_dict)

            if new_mat_label[0] in H_little_group_keys:
                induced_orbit_arr[row_block_index_lower:row_block_index_upper, column_block_index_lower:column_block_index_upper] = rotated_T_irrep_char * H_little_group_irrep[new_mat_label]

    return induced_orbit_arr

def bosonic_induced_mom_orbit_array_1D(T_element, T_orbit_vec, N, H_element, H_little_group_keys, H_coset_representative, 
                            H_little_group_irrep, faithful_H_dict, faithful_w3_irrep, induced_dim):
    ### H "typically" is \Gamma_4 \rtimes W3, only need to check if the W3 component of h_k h h_j^(-1) is in 
    ### the W3 subgroup of the little group

    induced_orbit_arr = np.zeros((induced_dim, induced_dim), dtype = np.complex128)
    
    h = faithful_H_dict[H_element]
    
    
    
    for k, coset_rep1 in enumerate(H_coset_representative):
        
        h_k = faithful_H_dict[(coset_rep1, "E")]
        h_k_w3 = faithful_w3_irrep[coset_rep1]
        
        rotated_T_orbit_vec = T_orbit_vec @ h_k_w3
        rotated_T_irrep_char = tg.mom_character(rotated_T_orbit_vec, T_element, N)
        res_initial = h_k @ h
        
        for j, coset_rep2 in enumerate(H_coset_representative):
                
            h_j = faithful_H_dict[(coset_rep2, "E")]

            res = res_initial @ inv(h_j)

            new_mat_label = gf.what_matrix(res, faithful_H_dict)
            
            if new_mat_label[0] in H_little_group_keys:

                induced_orbit_arr[k, j] = rotated_T_irrep_char * H_little_group_irrep[new_mat_label]
                           
    return induced_orbit_arr


def generate_bosonic_single_taste_array(translation_element_label, trans_essential_keys, taste_w3_irrep, faithful_H_dict, translation_group, mom_vec, N, H_little_group_keys, H_coset_representative, W3_irrep_dict, LG_irrep_dim, induced_dim):
    
    single_irrep_dict = {}
    for taste_w3_element_label in trans_essential_keys[translation_element_label]:
        
        induced_mom_array = bosonic_induced_mom_orbit_array(T_element = translation_group[translation_element_label], 
                                                        T_orbit_vec=mom_vec, N=N, H_element = taste_w3_element_label, 
                                                        H_little_group_keys = H_little_group_keys, 
                                                        H_coset_representative = H_coset_representative, 
                                                        H_little_group_irrep = taste_w3_irrep, 
                                                        faithful_H_dict = faithful_H_dict,
                                                        faithful_w3_irrep = W3_irrep_dict["T_1"],
                                                        LG_irrep_dim=LG_irrep_dim, induced_dim=induced_dim)


        single_irrep_dict[(translation_element_label,)+ taste_w3_element_label] = induced_mom_array
            
    return single_irrep_dict

def generate_bosonic_single_taste_array_1D(translation_element_label, trans_essential_keys, taste_w3_irrep, faithful_H_dict, translation_group, mom_vec, N, H_little_group_keys ,H_coset_representative, W3_irrep_dict, induced_dim):
    
    single_irrep_dict = {}
    for taste_w3_element_label in trans_essential_keys[translation_element_label]:
        
        induced_mom_array = bosonic_induced_mom_orbit_array_1D(T_element = translation_group[translation_element_label], 
                                                        T_orbit_vec=mom_vec, N=N, H_element = taste_w3_element_label, 
                                                        H_little_group_keys = H_little_group_keys, 
                                                        H_coset_representative = H_coset_representative, 
                                                        H_little_group_irrep = taste_w3_irrep, 
                                                        faithful_H_dict = faithful_H_dict,
                                                        faithful_w3_irrep = W3_irrep_dict["T_1"],
                                                        induced_dim=induced_dim)


        single_irrep_dict[(translation_element_label,)+ taste_w3_element_label] = induced_mom_array
            
    return single_irrep_dict


def momentum_induced_bosonic_representation(mom_key, N, dim, W3_irrep_dict, clifford_irrep_dict, converted_SW4_clifford_2_2_dict, pool_no):
    
    
    essential_keys, class_keys = essential_group_keys(N)
    
    trans_essential_keys = trans_essential_rot_cliff_keys(essential_keys)
    
    mom_dict = tg.mom_dictionary()
    
    translation_group = tg.generate_translation_group_elements(N, dim)

    mom_vec = mom_dict[mom_key]
    
    mom_group = bosonic_little_groups_and_orbits(mom_vec, [0,0,0], W3_irrep_dict)[0]

    mom_coset_representative = momentum_coset_representatives(W3_irrep_dict=W3_irrep_dict)[mom_key]
    
    orbit_dim = len(mom_coset_representative)
   
    mom_specific_taste_irreps = generate_all_bosonic_momentum_specific_taste_irreps(mom_key, 
                                                                                  W3_irrep_dict, clifford_irrep_dict, pool_no)
    induced_mom_irreps_dict = {}
    
    completed_irreps = check_irreps(mom_key=mom_key, N=N)
    
    for taste_w3_irrep_label, taste_w3_irrep in mom_specific_taste_irreps.items():
        if taste_w3_irrep_label not in completed_irreps:
            print(taste_w3_irrep_label)

            if np.array([list(taste_w3_irrep.values())[0]]).shape !=(1,):
                LG_irrep_dim = list(taste_w3_irrep.values())[0].shape[0]
                
                induced_dim = orbit_dim * LG_irrep_dim

                func = partial(generate_bosonic_single_taste_array, trans_essential_keys=trans_essential_keys,
                               taste_w3_irrep=taste_w3_irrep, 
                                faithful_H_dict=converted_SW4_clifford_2_2_dict, 
                                translation_group=translation_group, mom_vec=mom_vec, N=N, H_little_group_keys=list(mom_group.keys()),
                                H_coset_representative=mom_coset_representative, 
                                W3_irrep_dict=W3_irrep_dict, 
                                LG_irrep_dim=LG_irrep_dim, induced_dim=induced_dim)
                
            else:
                LG_irrep_dim = 1

                induced_dim = orbit_dim * LG_irrep_dim

                func = partial(generate_bosonic_single_taste_array_1D, trans_essential_keys=trans_essential_keys,
                               taste_w3_irrep=taste_w3_irrep, 
                                faithful_H_dict=converted_SW4_clifford_2_2_dict, 
                                translation_group=translation_group, mom_vec=mom_vec, N=N, H_little_group_keys=list(mom_group.keys()),
                                H_coset_representative=mom_coset_representative, 
                                W3_irrep_dict=W3_irrep_dict, 
                                 induced_dim=induced_dim)


            p = Pool(pool_no)
            single_irrep_list = p.map(func, translation_group.keys())
            #single_irrep_list = list(map(func, translation_group.keys()))
            p.terminate()

            single_irrep_dict = dict(ChainMap(*single_irrep_list))
            
            save_irrep(mom_key=mom_key, irrep_label=taste_w3_irrep_label, N=N, irrep=single_irrep_dict)

            del single_irrep_dict
                       
    return None

#Usage
'''
W3_irreps = W3.generate_W3_irrep_dict()
cliff41 = c41.generate_clifford_4_1_complete_irrep_dict()
converted_SW4_clifford_2_2_dict = rf.generate_converted_SW4_clifford_2_2_dict(W3_irreps, cliff41)
N=3
Full=False
sg.momentum_induced_bosonic_representation("mom000", N, dim=3, W3_irrep_dict=W3_irreps, clifford_irrep_dict=cliff41, converted_SW4_clifford_2_2_dict=converted_SW4_clifford_2_2_dict, pool_no=6)
sg.momentum_induced_bosonic_representation("mom001", N, dim=3, W3_irrep_dict=W3_irreps, clifford_irrep_dict=cliff41, converted_SW4_clifford_2_2_dict=converted_SW4_clifford_2_2_dict, pool_no=6)
sg.momentum_induced_bosonic_representation("mom110", N, dim=3, W3_irrep_dict=W3_irreps, clifford_irrep_dict=cliff41, converted_SW4_clifford_2_2_dict=converted_SW4_clifford_2_2_dict, pool_no=6)
sg.momentum_induced_bosonic_representation("mom111", N, dim=3, W3_irrep_dict=W3_irreps, clifford_irrep_dict=cliff41, converted_SW4_clifford_2_2_dict=converted_SW4_clifford_2_2_dict, pool_no=6)

mom000 = sg.load_irrep("mom000", N=N)
mom001 = sg.load_irrep("mom001", N=N)
mom110 = sg.load_irrep("mom110", N=N)
mom111 = sg.load_irrep("mom111", N=N)
total_mom_dict = {**mom000, **mom001, **mom110 , **mom111}

keys, class_dict = sg.essential_group_keys(N=N)
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

########################################################################################################################################
#### Functions to generate central extensions of other momentum little groups little group (They are all abelian Z_n for mom != 000)


def central_extension(original_group, SW3_tilde_faithful_irrep):
    
    dict_rot =  {}
    
    for key1 in original_group.keys():
        if key1 != 'E':
            dict_rot[key1] = SW3_tilde_faithful_irrep[key1]
    
    for key1 in original_group.keys():
        for key2 in original_group.keys():
            dict_rot[key1+'*'+key2] = np.round(SW3_tilde_faithful_irrep[key1] @ SW3_tilde_faithful_irrep[key2], 10)
            dict_rot[key1+'*'+key2+'*'+key1] = np.round(SW3_tilde_faithful_irrep[key1] @ SW3_tilde_faithful_irrep[key2] @ SW3_tilde_faithful_irrep[key1], 10)
            
           
    unique_set = np.unique(list(dict_rot.values()), axis=0)

    final_dict = {}
    
    dict_rot['E'] = np.identity(4)

    for mat in unique_set:
        list_keys = [key.lstrip("*").rstrip('*') for (key, vals) in dict_rot.items() if np.all(vals ==mat)]
        label = sorted(list_keys,key=len)[0]
        final_dict[label] = mat
        
    return final_dict


def abelian_char_ele(l, n):
    
    return np.exp(2*np.pi*1j *l / n)


def faithful_Zn_reps(n, group):
    
    group_keys = list(group.keys())
    
    group_keys.remove('E')
    
    generator_ele = sorted(group_keys,key=len)[0]
    
    
    irrep_dict = {}
    D_g = {}
    
    for i in range(n):
        if i % 2==1:
            D_g['A_'+str(i)]= abelian_char_ele(i, n)
        
    for k, (irrep_label, irrep_g) in enumerate(D_g.items()):
        
        irrep_dict[irrep_label] = {}
        
        irrep_dict[irrep_label]['E'] = 1
        for i in range(n):
            element = matrix_power(group[generator_ele], i)
            element_label = gf.what_matrix(element, group)
            
            irrep_dict[irrep_label][element_label] = np.round(np.power(irrep_g, i),14)
            
            
    return irrep_dict

########################################################################################################################################
#### Function to generate all fermionic little group irreps

def generate_extended_fermionic_little_group_irreps(faithful_little_group_matrix_dict):
    ###############################################################################################################
    ### Returns a dictionary keyed by irrep labels, valued by irrep dictionarys
    
    classes = gf.conjugate_classes(faithful_little_group_matrix_dict)
    group_order = len(faithful_little_group_matrix_dict)
    
    if group_order == 48:
        irrep_dict = W3.generate_sw3_tilde_fermionic_irreps()
        
    elif group_order == 1:
        irrep_dict = {'A_1' : {'E' : 1}}
        
    else:
        irrep_dict = faithful_Zn_reps(group_order, group=faithful_little_group_matrix_dict)

    return irrep_dict



def fermionic_little_groups(mom_vec, clifford_fermionic_irrep, W3):
    
    final_mom_dict = {}
    final_taste_dict = {}
    mom_orbit = []
    
    gamma_chars = gf.compute_characters_no_class(clifford_fermionic_irrep)
        
    for label, matrix in W3.items():
        isLittleGroupEle = True
        mat_inv = gf.what_matrix(inv(matrix), W3)
        
        new_mom_vec = matrix @ mom_vec
        mom_orbit.append(new_mom_vec)
        
        if np.all(new_mom_vec == mom_vec):

            final_mom_dict[label] = matrix
            
            for clifford_group_element_label, clifford_group_element_matrix in clifford_fermionic_irrep.items():
                hnh_1 = general_gamma_commutation(clifford_group_element_label, mat_inv, faithful_clifford_irrep=clifford_fermionic_irrep)
                hnh_1_mat = clifford_fermionic_irrep[hnh_1]
                
                if np.trace(hnh_1_mat) != gamma_chars[clifford_group_element_label]:
                    isLittleGroupEle = False
                    break
                    
            if isLittleGroupEle:
                 final_taste_dict[label] = matrix
                    
    return final_mom_dict, mom_orbit, final_taste_dict


################################################################################################################################################ Generate the fermionic taste cosets under rotation 

def fermionic_taste_orbits(mom_little_group, orbit_rep_dict, taste_little_group, fermionic_taste_irreps, W3):
    
    faithful_taste_irrep = fermionic_taste_irreps['gamma']
    
    orbit_dict = {}
    
    taste_cosets = gf.right_cosets(group_dict=mom_little_group, subgroup_dict=taste_little_group)
    
    taste_coset_reps = [sorted(val_coset, key=len)[0] for val_coset in taste_cosets.values()]
          
    taste_coset_reps = sorted(taste_coset_reps, key=len)

    for coset_rep in taste_coset_reps:
        coset_rep_inv = gf.what_matrix(inv(W3[coset_rep]), W3)        
        for fermionic_irr_label, fermionic_irrep in fermionic_taste_irreps.items():
            IrrepInOrbit = True

            for taste_element in faithful_taste_irrep.keys():
                hnh_1 = general_gamma_commutation(taste_element, coset_rep_inv, faithful_clifford_irrep=faithful_taste_irrep)
                hnh_1_mat = orbit_rep_dict[hnh_1]

                
                if np.trace(hnh_1_mat) != np.trace(fermionic_irrep[taste_element]):

                    IrrepInOrbit = False
                    break
            if IrrepInOrbit:
                orbit_dict[coset_rep] = fermionic_irrep
                break

    return orbit_dict


###########################################################################################################################################
### Functions to do the first induction step to obtain fermionic irreps of gamma_4_1 rtimes W3

def induced_fermionic_taste_orbit_array(W3_group_element, little_group, extended_LG_irrep, LG_irrep_dim, clifford_group_element, clifford_irrep_orbit_dict, clifford_fermionic_irrep_dim, coset_representatives, W3_faithful_dict, projective_sw3_rep):
    #### See rep theory notes X^{\tilde h_k (\sigma)} = e^{i*2*\pi*(D^{3d}(\tilde h_k) p . clifford_group_element}
    #### Where D^{3d} is the faithful irrep that acts on 3 vectors
    #### Generates a single element of the induced irrep

        
    orbit_dim = len(coset_representatives)
    coset_block_dim = LG_irrep_dim * clifford_fermionic_irrep_dim
    induced_dim = orbit_dim * coset_block_dim
    induced_orbit_arr = np.zeros((induced_dim, induced_dim),  dtype = np.complex128)

    h = W3_faithful_dict[W3_group_element]

    for k, coset_rep1 in enumerate(coset_representatives):
        
        row_block_index_lower = coset_block_dim*k
        row_block_index_upper = coset_block_dim*(k+1)

        h_k = W3_faithful_dict[coset_rep1]
        
        taste_irrep_orbit_element_dict = clifford_irrep_orbit_dict[coset_rep1]
        
        res_initial = h_k @ h
        taste_matrix_raw = taste_irrep_orbit_element_dict[clifford_group_element]

        for j, coset_rep2 in enumerate(coset_representatives):

            column_block_index_lower = coset_block_dim*j
            column_block_index_upper = coset_block_dim*(j+1)
            
            h_j = W3_faithful_dict[coset_rep2]

            res = res_initial @ inv(h_j)
            
            new_mat_label = gf.what_matrix(res, little_group)
            
            if new_mat_label != None:
                
                induced_orbit_arr[row_block_index_lower:row_block_index_upper, column_block_index_lower:column_block_index_upper] = np.kron(taste_matrix_raw @ projective_sw3_rep[new_mat_label], extended_LG_irrep[new_mat_label])

                
    return induced_orbit_arr

def fermionic_taste_induced_representations(mom_key, W3_irrep_dict, clifford_irrep_dict):
    
    induced_taste_irreps_dict = {}
    
 
    SW3_tilde_faithful = W3.generate_SW3_irrep_threehalf()
    W3_faithful = W3_irrep_dict['T_1']
    clifford_4_1_faithful = clifford_irrep_dict['gamma']
    
    fermionic_taste_irreps_4_1 = {'gamma' : clifford_irrep_dict['gamma'], 'gamma_bar' : clifford_irrep_dict['gamma_bar']}
    
    mom_dict = tg.mom_dictionary()
    
    mom_vec = mom_dict[mom_key]
    
    mom_group, mom_orbit, taste_group = fermionic_little_groups(mom_vec=mom_vec, clifford_fermionic_irrep=clifford_4_1_faithful,  W3=W3_faithful)
    
    fermionic_taste_orbit_dict = fermionic_taste_orbits(mom_little_group = mom_group, orbit_rep_dict=clifford_4_1_faithful, taste_little_group=taste_group, fermionic_taste_irreps = fermionic_taste_irreps_4_1, W3 = W3_faithful)
    
    coset_reps = fermionic_taste_orbit_dict.keys()
    
    taste_irrep_dim = fermionic_taste_orbit_dict['E']['E'].shape[0]
    
    extended_taste_group = central_extension(original_group=taste_group, SW3_tilde_faithful_irrep=SW3_tilde_faithful)
    
    extended_little_group_irrep_dict = generate_extended_fermionic_little_group_irreps(faithful_little_group_matrix_dict=extended_taste_group)
    
    projective_sw3_rep = W3.generate_projective_sw3_rep(clifford_4_1_faithful_irrep=clifford_4_1_faithful, W3_dict=W3_faithful)
    
    for irrep_label, single_irrep_dict in extended_little_group_irrep_dict.items():
        
        #if irrep_label != 'one_half':
            #continue
        
        induced_taste_irrep = {}
        
        if type(extended_little_group_irrep_dict[irrep_label]['E']) == np.ndarray:
            lg_irrep_dim = extended_little_group_irrep_dict[irrep_label]['E'].shape[0]
        else:
            lg_irrep_dim = 1

        for irrep_matrix_label in mom_group.keys():
            
            #if irrep_matrix_label != 'E':
                #continue

            for taste_group_element in clifford_4_1_faithful.keys():
                
                #if taste_group_element != 'E':
                    #continue

                induced_element = induced_fermionic_taste_orbit_array(W3_group_element=irrep_matrix_label, little_group=taste_group, extended_LG_irrep=single_irrep_dict, LG_irrep_dim=lg_irrep_dim, clifford_group_element=taste_group_element, clifford_irrep_orbit_dict=fermionic_taste_orbit_dict, clifford_fermionic_irrep_dim=taste_irrep_dim, coset_representatives=coset_reps, W3_faithful_dict=W3_faithful, projective_sw3_rep=projective_sw3_rep)
                
                induced_taste_irrep[(irrep_matrix_label, taste_group_element)] = induced_element

        induced_taste_irreps_dict[mom_key + "_" + "gamma_4_1"  + "_" + irrep_label] = induced_taste_irrep
        
        
        
    return induced_taste_irreps_dict


    
###########################################################################################################################################
### Functions to do the second induction step to obtain fermionic irreps of T_N rtimes gamma_4_1 rtimes W3


def fermionic_induced_mom_orbit_array(T_element, T_orbit_vec, N, H_element, H_little_group_keys, H_coset_representatives, H_little_group_irrep, faithful_H_dict, faithful_W3_irrep, LG_irrep_dim, induced_dim):
    ### H "typically" is \Gamma_4 \rtimes W3, only need to check if the W3 component of h_k h h_j^(-1) is in 
    ### the W3 subgroup 

    induced_orbit_arr = np.zeros((induced_dim, induced_dim), dtype = np.complex128)
    
    h = faithful_H_dict[H_element]
    for k, coset_rep1 in enumerate(H_coset_representatives):
        
        row_block_index_lower = LG_irrep_dim*k
        row_block_index_upper = LG_irrep_dim*(k+1)
        
        h_k = faithful_H_dict[(coset_rep1, "E")]
        h_k_W3 = faithful_W3_irrep[coset_rep1]
        
        rotated_T_orbit_vec = T_orbit_vec @ h_k_W3
        rotated_T_irrep_char = tg.mom_character(rotated_T_orbit_vec, T_element, N)
        res_initial = h_k @ h
            
        for j, coset_rep2 in enumerate(H_coset_representatives):

            column_block_index_lower = LG_irrep_dim*j
            column_block_index_upper = LG_irrep_dim*(j+1)

            h_j = faithful_H_dict[(coset_rep2, "E")]

            res = res_initial @ inv(h_j)

            new_mat_label = gf.what_matrix(res, faithful_H_dict)

            if new_mat_label[0] in H_little_group_keys:
                induced_orbit_arr[row_block_index_lower:row_block_index_upper, column_block_index_lower:column_block_index_upper] = rotated_T_irrep_char * H_little_group_irrep[new_mat_label]

    return induced_orbit_arr


def generate_fermionic_single_taste_array(translation_element_label, trans_essential_keys, taste_w3_irrep, faithful_H_dict, translation_group, mom_vec, N, H_little_group_keys, H_coset_representatives, faithful_W3_irrep_dict, LG_irrep_dim, induced_dim):
    
    single_irrep_dict = {}
    for taste_w3_element_label in trans_essential_keys[translation_element_label]:
        
        taste_w3_element = faithful_H_dict[taste_w3_element_label]
        
        induced_mom_array = fermionic_induced_mom_orbit_array(T_element = translation_group[translation_element_label], 
                                                        T_orbit_vec=mom_vec, N=N, H_element = taste_w3_element_label, 
                                                        H_little_group_keys = H_little_group_keys, 
                                                        H_coset_representatives = H_coset_representatives, 
                                                        H_little_group_irrep = taste_w3_irrep, 
                                                        faithful_H_dict = faithful_H_dict,
                                                        faithful_W3_irrep = faithful_W3_irrep_dict,
                                                        LG_irrep_dim=LG_irrep_dim, induced_dim=induced_dim)


        single_irrep_dict[(translation_element_label,)+ taste_w3_element_label] = induced_mom_array
            
    return single_irrep_dict

def momentum_induced_fermionic_representation(mom_key, N, dim, W3_irrep_dict, clifford_irrep_dict, converted_SW4_clifford_2_2_dict, pool_no):
    
    essential_keys, class_keys = essential_group_keys(N)
    
    trans_essential_keys = trans_essential_rot_cliff_keys(essential_keys)
    
    mom_dict = tg.mom_dictionary()
    
    translation_group = tg.generate_translation_group_elements(N, dim)

    mom_vec = mom_dict[mom_key]
    
    mom_group = bosonic_little_groups_and_orbits(mom_vec, [0,0,0], W3_irrep_dict)[0]

    mom_coset_representatives = momentum_coset_representatives(W3_irrep_dict=W3_irrep_dict)[mom_key]
    
    orbit_dim = len(mom_coset_representatives)
   
    mom_specific_taste_irreps = fermionic_taste_induced_representations(mom_key, W3_irrep_dict, clifford_irrep_dict)
    
    induced_mom_irreps_dict = {}
    
    completed_irreps = check_irreps(mom_key=mom_key, N=N)
    
    faithful_W3_irrep_dict = W3_irrep_dict["T_1"]
    
    
    for taste_w3_irrep_label, taste_w3_irrep in mom_specific_taste_irreps.items():
        if taste_w3_irrep_label not in completed_irreps:
            print(taste_w3_irrep_label)

            LG_irrep_dim = list(taste_w3_irrep.values())[0].shape[0]

            induced_dim = orbit_dim * LG_irrep_dim

            func = partial(generate_fermionic_single_taste_array, trans_essential_keys=trans_essential_keys, taste_w3_irrep=taste_w3_irrep, 
                            faithful_H_dict=converted_SW4_clifford_2_2_dict, 
                            translation_group=translation_group, mom_vec=mom_vec, N=N, H_little_group_keys=list(mom_group.keys()),
                            H_coset_representatives=mom_coset_representatives, 
                            faithful_W3_irrep_dict=faithful_W3_irrep_dict, 
                            LG_irrep_dim=LG_irrep_dim, induced_dim=induced_dim)

            p = Pool(pool_no)
            single_irrep_list = p.map(func, translation_group.keys())
            #single_irrep_list = list(map(func, translation_group.keys()))
            p.terminate()

            single_irrep_dict = dict(ChainMap(*single_irrep_list))
            
            save_irrep(mom_key=mom_key, irrep_label=taste_w3_irrep_label, N=N, irrep=single_irrep_dict)

            del single_irrep_dict
                       
    return None

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
#### Functions to save, load and check which irreps are completeted 

def save_irrep(mom_key, irrep_label, N, irrep):
    
    directory = "./staggered_irreps/N_"  + str(N) + "/" + mom_key + "/" 
    #directory = "C:\\Users\\Shaun252\\Desktop\\staggered_irreps\\N_"  + str(N) + "\\" + mom_key + "\\"  
    #directory = "staggered_irreps/"+ mom_key + "/"  + str(N) + "/"
    filename = directory + irrep_label
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.save(filename, irrep)
    print("Irrep: " + str(irrep_label) + " saved")
    
def load_irrep(mom_key, N, irrep_list=False):
    mom_irrep_dict = {}
    directory = "./staggered_irreps/N_"  + str(N) + "/" + mom_key + "/" 
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
    directory = "./staggered_irreps/N_"  + str(N) + "/" + mom_key + "/" 
    #directory = "C:\\Users\\Shaun252\\Desktop\\staggered_irreps\\N_"  + str(N) + "\\" + mom_key + "\\"  
    #directory = "staggered_irreps/"+ mom_key + "/"  + str(N) + "/" 
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            irrep_list.append(name.rstrip(".npy"))
            
    return irrep_list



########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
#### Functions to  format the irrep string to make it latex friendly 




def tasteclean(taste):
    
    if taste == 'p':
        return ':::pi'
    else:
        return taste



def cleanMomTasteRotIrrep(irrepKey):
    
    
    zeroTasteLabelDict = {'A_0' : 'A_0^{+}',  'A_1': 'A_0^{-}',  'A_2': 'A_1^{+}',  'A_3': 'A_1^{-}',       
                          'E_0' : 'E_0^{+}',  'E_1': 'E_0^{-}',  
                          'T_0' : 'T_0^{+}',  'T_1': 'T_0^{-}',  'T_2': 'T_1^{+}',  'T_3': 'T_1^{-}'}
    
    piTasteLabelDict = {'A_0' : 'A_0^{+}',  'A_1': 'A_0^{-}',  'A_2': 'A_1^{+}',  'A_3': 'A_1^{-}', 
                        'A_4' : 'A_2^{-}',  'A_5': 'A_2^{+}',  'A_6': 'A_3^{-}',  'A_7': 'A_3^{+}',
                        'E_0' : 'E_0^{+}',  'E_1': 'E_0^{-}' }
    
    
    mom_key = irrepKey[3:6]
    C_0 = irrepKey[-9]
    taste_key = irrepKey[-8:-4]
    rotIrrep = irrepKey[-3:]
    
    newRotIrrep = rotIrrep
    if mom_key == '000':
        if taste_key[1:] in ['000', 'ppp']:
            newRotIrrep = zeroTasteLabelDict[rotIrrep]
        else:
            newRotIrrep = piTasteLabelDict[rotIrrep]
    
    newIrrepKey = str((mom_key[0], mom_key[1], mom_key[2])) + ' :::, :::rtimes :::, ' + "[" + str((tasteclean(taste_key[0]), tasteclean(taste_key[1]), tasteclean(taste_key[2]), tasteclean(taste_key[3]))) + ", " + tasteclean(C_0) + "] :::,   :::rtimes :::,   " + newRotIrrep
    
    return newIrrepKey.replace("'", "")


def cleanMomTasteRotIrrepCaption(irrepKey):
    
    
    zeroTasteLabelDict = {'A_0' : 'A_0^{+}',  'A_1': 'A_0^{-}',  'A_2': 'A_1^{+}',  'A_3': 'A_1^{-}',       
                          'E_0' : 'E_0^{+}',  'E_1': 'E_0^{-}',  
                          'T_0' : 'T_0^{+}',  'T_1': 'T_0^{-}',  'T_2': 'T_1^{+}',  'T_3': 'T_1^{-}'}
    
    piTasteLabelDict = {'A_0' : 'A_0^{+}',  'A_1': 'A_0^{-}',  'A_2': 'A_1^{+}',  'A_3': 'A_1^{-}', 
                        'A_4' : 'A_2^{-}',  'A_5': 'A_2^{+}',  'A_6': 'A_3^{-}',  'A_7': 'A_3^{+}',
                        'E_0' : 'E_0^{+}',  'E_1': 'E_0^{-}' }
    

    mom_key = irrepKey[3:6]
    C_0 = irrepKey[-9]
    taste_key = irrepKey[-8:-4]
    rotIrrep = irrepKey[-3:]
    
    newRotIrrep = rotIrrep
    if mom_key == '000':
        if taste_key[1:] in ['000', 'ppp']:
            newRotIrrep = zeroTasteLabelDict[rotIrrep]
        else:
            newRotIrrep = piTasteLabelDict[rotIrrep]
    
    newIrrepKey = str((mom_key[0], mom_key[1], mom_key[2])) + ' :::, :::rtimes :::, ' + ":::[" + str((tasteclean(taste_key[0]), tasteclean(taste_key[1]), tasteclean(taste_key[2]), tasteclean(taste_key[3]))) + ", " + tasteclean(C_0) + ":::] :::,   :::rtimes :::,   " + newRotIrrep
    
    return newIrrepKey.replace("'", "")