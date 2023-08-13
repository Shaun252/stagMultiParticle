import numpy as np
from numpy.linalg import inv, matrix_power
from math import sqrt
from scipy.linalg import expm

import gfunctions as gf
import stagGroup as sg
import cliff41Group as c41


########################################################################################################################################
########################################################################################################################################
############################################################### BOSONS ###############################################################

########################################################################################################################################
#### Functions for SW3 irrep generation.

def rotation_key_sort(key):
    
    key_list=key.split('*')
    first_len = len(key_list)
    second_len = len(key)
    
    return 10*first_len+second_len

def generate_SW3_irrep_T1():
    ### Generates one 3d irrep pure_rotations dictionary where dictionary has form 
    ### key = "group element label", value = "matrix of group element"
    ### Also prioritising the labelling on R_12 since we typically priortise the momentum which focus on the z direction
    
    R_12 =  np.array([[0,1,0],[-1,0,0],[0,0,1]])
    R_31 =  np.array([[0,0,-1],[0,1,0],[1,0,0]])
    R_23 =  np.array([[1,0,0],[0,0,1],[0,-1,0]])
    E = np.identity(3)

    dict_rot = {}

    for i in range(1,4):
    
        dict_rot["*R_12"*i] = matrix_power(R_12, i)
        dict_rot["*R__23"*i] = matrix_power(R_23, i)
        dict_rot["*R___31"*i] = matrix_power(R_31, i)
    
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
        label = sorted(list_keys,key=rotation_key_sort)[0]
        final_dict[label] = mat
        
    return final_dict


def generate_klein_subgroup_T1(SW3_irrep_T1):
    # Character matching
    KG_dict = {}

    for key, vals in gf.conjugate_classes(SW3_irrep_T1).items():
        if len(vals) == 3 or len(vals)==1:
            for val in vals:
                KG_dict[val] = SW3_irrep_T1[val]
    
    return KG_dict

def generate_s3_group_structure():
    #https://groupprops.subwiki.org/wiki/Standard_representation_of_symmetric_group:S3
    s3_perm_dict = {"perm_0" : [[1,0],[0,1]],
                    "perm_12" : [[-1,1],[0,1]],
                    "perm_23" : [[1,0],[1,-1]],
                    "perm_13" : [[0,-1],[-1,0]],
                    "perm_123" : [[0,-1],[1,-1]],
                    "perm_132" : [[-1,1],[-1,0]]}

    for label1, mat1 in s3_perm_dict.items():
        for label2, mat2 in s3_perm_dict.items():

            prod = np.dot( mat1, mat2)
            print(label1 + "*" + label2)
            print("=" + gf.what_matrix(prod , s3_perm_dict))
            print("*****************************************")


def generate_SW3_cosets_under_KG(SW3_irrep_T1_dict):
    #### Used method from https://groupprops.subwiki.org/wiki/Linear_representation_theory_of_symmetric_group:S4
    #### to generate a dictionary labeled by cosetx: [element labels in coset]
    
    klein_subgroup_T1 = generate_klein_subgroup_T1(SW3_irrep_T1_dict)
    
    KG_group = list(klein_subgroup_T1.values())
    total_list = []
    coset_dict = {}
    i=0
    for key1, vals1 in SW3_irrep_T1_dict.items():
        if np.all([np.any(vals1 != matrix) for matrix in total_list]):
            
            coset = [np.dot(vals1, kg_element) for kg_element in KG_group]
            total_list += coset           
            coset_imp = []
            for matrices in coset:
                for key2, vals2 in SW3_irrep_T1_dict.items():
                    if np.all(matrices == vals2):
                        coset_imp.append(key2)
                        
            coset_dict["coset"+str(i)] = coset_imp
            i+=1
            
    return coset_dict 


def generate_s3_subgroup_T1(SW3_irrep_T1_dict, cosets):
    #### Generates a standard matrix dictionary "label : matrix" for the S3 subgroup of SW3 through brute force comparison
    #### to S3 permutations structure also using character matching to know that R_ij = (ij) for i,j = 1,2,3
    
    s3_sg_dict = {}

    coset_dict = cosets.copy()

    R_E = SW3_irrep_T1_dict["E"]
    for key1, coset1 in coset_dict.items():
        if "R_12" not in coset1 and "R___31" not in coset1 and "R__23" not in coset1 and "E" not in coset1:
            for val1 in coset1:
                for key2, coset2 in coset_dict.items():
                    if "R_12" not in coset2 and "R___31" not in coset2 and "R__23" not in coset2 and "E" not in coset2: 
                        for val2 in coset2:
                            for i, (key3, coset3) in enumerate(coset_dict.items()):
                                if "R_12" in coset3 or "R___31" in coset3 or "R__23" in coset3 and "E" not in coset3:
                                    for val3 in coset3:
                                        for j, (key4, coset4) in enumerate(coset_dict.items()):
                                            if "R_12" in coset4 or "R___31" in coset4 or "R__23" in coset4 and "E" not in coset4:
                                                for val4 in coset4:
                                                    for k, (key5, coset5) in enumerate(coset_dict.items()):
                                                        if "R_12" in coset5 or "R___31" in coset5 or "R__23" in coset5 and "E" not in coset5:
                                                            for val5 in coset5:

                                                                R_12 = SW3_irrep_T1_dict[val3]

                                                                R_31 = SW3_irrep_T1_dict[val4]

                                                                R_23 = SW3_irrep_T1_dict[val5]

                                                                R_123 = SW3_irrep_T1_dict[val1]

                                                                R_312 = SW3_irrep_T1_dict[val2]


                                                                prod_R_12_R_123 = np.dot(R_12, R_123)
                                                                prod_R_31_R_123 = np.dot(R_31, R_123)
                                                                prod_R_23_R_123 = np.dot(R_23, R_123)
                                                                prod_R_123_R_12 = np.dot(R_123, R_12)
                                                                prod_R_123_R_31 = np.dot(R_123, R_31)
                                                                prod_R_123_R_23 = np.dot(R_123, R_23)

                                                                prod_R_12_R_312 = np.dot(R_12, R_312)
                                                                prod_R_31_R_312 = np.dot(R_31, R_312)
                                                                prod_R_23_R_312 = np.dot(R_23, R_312)
                                                                prod_R_312_R_12 = np.dot(R_312, R_12)
                                                                prod_R_312_R_31 = np.dot(R_312, R_31)
                                                                prod_R_312_R_23 = np.dot(R_312, R_23)

                                                                prod_R_123_R_312 = np.dot(R_123, R_312)
                                                                prod_R_312_R_123 = np.dot(R_312, R_123)

                                                                prod_R_123_R_123 = np.dot(R_123, R_123)
                                                                prod_R_312_R_312 = np.dot(R_312, R_312)

                                                                prod_R_12_R_12 = np.dot(R_12, R_12)
                                                                prod_R_31_R_31 = np.dot(R_31, R_31)
                                                                prod_R_23_R_23 = np.dot(R_23, R_23)

                                                                prod_R_12_R_31 = np.dot(R_12, R_31)
                                                                prod_R_12_R_23 = np.dot(R_12, R_23)
                                                                prod_R_31_R_12 = np.dot(R_31, R_12)
                                                                prod_R_31_R_23 = np.dot(R_31, R_23)
                                                                prod_R_23_R_12 = np.dot(R_23, R_12)
                                                                prod_R_23_R_31 = np.dot(R_23, R_31)




                                                                if (np.all(prod_R_12_R_123 == R_23) and 
                                                                    np.all(prod_R_31_R_123 == R_12) and 
                                                                    np.all(prod_R_23_R_123 == R_31) and 
                                                                    np.all(prod_R_123_R_12 == R_31) and
                                                                    np.all(prod_R_123_R_31 == R_23) and 
                                                                    np.all(prod_R_123_R_23 == R_12) and

                                                                    np.all(prod_R_12_R_312 == R_31) and 
                                                                    np.all(prod_R_31_R_312 == R_23) and 
                                                                    np.all(prod_R_23_R_312 == R_12) and 
                                                                    np.all(prod_R_312_R_12 == R_23) and
                                                                    np.all(prod_R_312_R_31 == R_12) and 
                                                                    np.all(prod_R_312_R_23 == R_31) and

                                                                    np.all(prod_R_123_R_123 == R_312) and 
                                                                    np.all(prod_R_312_R_312 == R_123) and
                                                                    np.all(prod_R_312_R_123 == R_E) and 
                                                                    np.all(prod_R_123_R_312 == R_E) and

                                                                    np.all(prod_R_12_R_12 == R_E) and
                                                                    np.all(prod_R_31_R_31 == R_E) and
                                                                    np.all(prod_R_23_R_23 == R_E) and

                                                                    np.all(prod_R_12_R_31 == R_312) and
                                                                    np.all(prod_R_12_R_23 == R_123) and
                                                                    np.all(prod_R_31_R_12 == R_123) and
                                                                    np.all(prod_R_31_R_23 == R_312) and
                                                                    np.all(prod_R_23_R_12 == R_312) and
                                                                    np.all(prod_R_23_R_31 == R_123) and
                                                                    np.any(R_12 != R_23) and
                                                                    np.any(R_12 != R_31) and
                                                                    np.any(R_31 != R_23) and
                                                                    np.any(R_12 != R_123) and
                                                                    np.any(R_12 != R_312) and
                                                                    np.any(R_31 != R_123) and
                                                                    np.any(R_31 != R_312) and
                                                                    np.any(R_23 != R_123) and
                                                                    np.any(R_23 != R_312) ):


                                                                        perm0 = gf.what_matrix(R_E, SW3_irrep_T1_dict)
                                                                        perm12 = gf.what_matrix(R_12, SW3_irrep_T1_dict)
                                                                        perm23 = gf.what_matrix(R_31, SW3_irrep_T1_dict)
                                                                        perm13 = gf.what_matrix(R_23, SW3_irrep_T1_dict)
                                                                        perm123 = gf.what_matrix(R_123, SW3_irrep_T1_dict)
                                                                        perm132 = gf.what_matrix(R_312, SW3_irrep_T1_dict)

                                                                        s3_sg_dict["perm_0"] = perm0
                                                                        s3_sg_dict["perm_12"] = perm12
                                                                        s3_sg_dict["perm_13"] = perm13
                                                                        s3_sg_dict["perm_23"] = perm23
                                                                        s3_sg_dict["perm_123"] = perm123
                                                                        s3_sg_dict["perm_132"] = perm132

                                                                        break

    return s3_sg_dict



def generate_SW3_irrep_E(SW3_irrep_T1_dict):
    ### Returns dictionary corresponding to the SW3 irrep E, generated from the above method
    ### Need a unitary representation
    # this corresponds to the complex rep on #https://groupprops.subwiki.org/wiki/Standard_representation_of_symmetric_group:S3
    # as opposed to rep used to generate s3 group structure which is not unitary
    
    cosets = generate_SW3_cosets_under_KG(SW3_irrep_T1_dict)
    S3_subgroup = generate_s3_subgroup_T1(SW3_irrep_T1_dict, cosets)
    eps_p = np.exp(2.0*np.pi*1j /3.0)
    eps_m = np.exp(-2.0*np.pi*1j /3.0)
    
    s3_perm_dict = {"perm_0" : np.array([[1,0],[0,1]]),
    "perm_12" : np.array([[0,1],[1,0]]),
    "perm_23" : np.array([[0, eps_m],[eps_p, 0]]),
    "perm_13" : np.array([[0, eps_p],[eps_m, 0]]),
    "perm_123" : np.array([[eps_p,0],[0, eps_m]]),
    "perm_132" : np.array([[eps_m,0],[0, eps_p]])}
    
    SW3_E_dict = {}
    
    for key1, val1 in S3_subgroup.items():
        for key2, val2 in cosets.items():
            if val1 in val2:
                for val in val2:
                    SW3_E_dict[val] = np.array(s3_perm_dict[key1])
                    
                    
    return SW3_E_dict


def generate_SW3_irrep_dict():
    
    T1 = generate_SW3_irrep_T1()
    
    E = generate_SW3_irrep_E(T1)
    
    SW3_irreps = {}
    
    trivial = {key : 1 for key in T1.keys()}
    
    trivial_asym = {key : (-val if key.count("R") % 2 == 1 else val) for key, val in trivial.items()}
    
    T2 = {key : np.multiply(val, trivial_asym[key]) for key, val in T1.items()}
    
    
    SW3_irreps["A_0"] = trivial

    SW3_irreps["A_1"] = trivial_asym
    
    SW3_irreps["T_0"] = T1
    
    SW3_irreps["T_1"] = T2
    
    SW3_irreps["E_0"] = E
    
    return SW3_irreps

########################################################################################################################################
#### Functions for W3 irrep generation.


def generate_W3_irrep_T1minus(SW3_irrep_T1_dict):
    ### Generates one 3d irrep octahedral group dictionary where dictionary has form 
    ###key = "group element label", value = "matrix of group element"
    ### Also can generate "complete dict" which gives multiply equations for group elements
    
    W3_irrep = {}
    
    for key, val in SW3_irrep_T1_dict.items():
        W3_irrep[key] = val
    
    I_S =  -1*np.identity(3)

    W3_irrep["I_S"] = I_S

    for key, vals in W3_irrep.copy().items():
    
        W3_irrep[key + "*I_S"] = np.dot(vals, I_S)
        
    del W3_irrep["E*I_S"]
    del W3_irrep["I_S*I_S"]
    
    return W3_irrep


def generate_W3_irrep_E_plus(SW3_irrep_T1_dict, W3_irrep_T1_dict):
    ### Extend the previous E irrep of SW3 to E^+ of W3 by representing all elements X*I_S of W3 by X in SW3 
    SW3_E = generate_SW3_irrep_E(SW3_irrep_T1_dict)
    
    W3_E_plus_dict = {}
    
    for key, val in W3_irrep_T1_dict.items():
        
        if key == "I_S":
            W3_E_plus_dict[key] = SW3_E["E"]
        elif "I_S" in key:
            stripped_key = key.rstrip("*I_S")
            W3_E_plus_dict[key] = SW3_E[stripped_key]
        else:
            W3_E_plus_dict[key] = SW3_E[key]
            
    return W3_E_plus_dict


def generate_W3_irrep_dict():
    
    SW3_irrep_T1_dict = generate_SW3_irrep_T1()
    
    T1_minus = generate_W3_irrep_T1minus(SW3_irrep_T1_dict)
    
    E_plus = generate_W3_irrep_E_plus(SW3_irrep_T1_dict, T1_minus)
    
    W3_irreps = {}
    
    trivial_plus = {key : 1 for key in T1_minus.keys()}
    trivial_minus = {key : (-val if "I_S" in key else val) for key, val in trivial_plus.items()}
    
    trivial_asym_plus = {key : (-val if key.count("R") % 2 == 1 else val) for key, val in trivial_plus.items()}
    trivial_asym_minus = {key : (-val if key.count("R") % 2 == 1 else val) for key, val in trivial_minus.items()}
    
    T1_plus = {key : np.multiply(val, trivial_minus[key]) for key, val in T1_minus.items()}
    
    T2_minus = {key : np.multiply(val, trivial_asym_plus[key]) for key, val in T1_minus.items()}
    T2_plus = {key : np.multiply(val, trivial_asym_minus[key]) for key, val in T1_minus.items()}
    
    E_minus = {key : np.multiply(val, trivial_minus[key]) for key, val in  E_plus.items()}
    
    W3_irreps["A_0"] = trivial_plus

    W3_irreps["A_1"] = trivial_minus
    
    W3_irreps["A_2"] = trivial_asym_plus
    W3_irreps["A_3"] = trivial_asym_minus
    
    W3_irreps["T_0"] = T1_plus
    W3_irreps["T_1"] = T1_minus
    
    W3_irreps["T_2"] = T2_plus
    W3_irreps["T_3"] = T2_minus
    
    W3_irreps["E_0"] = E_plus
    
    W3_irreps["E_1"] = E_minus
    
    return W3_irreps

########################################################################################################################################
#### Functions for W3 little group generation. Irreps of D4, D4h and Z2 X Z2 provide the remaining irreps not subduced from W3.
### We need to have these subgroups in the 3D form of T_1^(-) from W3 to compute orbits of 3 vectors.


def generate_D4_E_irrep(mom_key):
    ### Obtain by character matching to http://symmetry.jacobs-university.de/cgi-bin/group.cgi?group=304&option=4, 
    ### There is an ambiguity in choosing C_2' and C_2'' in that either set of ['R_12*R_31*R_31*I_S', 'R_31*R_31*R_12*I_S']
    ### and ['R_31*R_31*I_S', 'R_23*R_23*I_S'] satisfy both C_2' and C_2''
    ### 
    ### Different mom001 vs mom010 vs mom100 have R_12, R_23, R_31 versions of this group respectively (3 copies of the subgroup)
    ### Matching convention for last 4 of each was done by verifying the form subgroup under T1 of W3
    ### W3_irrep_dict = irrep.generate_W3_irrep_dict()
    ### D4H = irrep.little_groups_and_orbits(mom_vec=[0,0,1], taste_vec=[np.pi, np.pi, np.pi], W3_irreps=W3_irrep_dict)[2]
    ### print(D4H['R_12*R___31*R___31*I_S'])
    ### print(D4H['R___31*R___31*R_12*I_S'])
    ### print(D4H['R___31*R___31*I_S'])
    ### print(D4H['R__23*R__23*I_S'])

    D4_E_irrep12 = {"E" :           np.array([[1,0],
                                    [0,1]]),
              "R_12" :             np.array([[0,1],
                                    [-1,0]]),
              "R_12*R_12*R_12" :   np.array([[0,-1],
                                    [1,0]]),
              "R_12*R_12" :        np.array([[-1,0],
                                    [0,-1]]),
              'R__23*R__23*R_12*I_S':np.array([[0,-1],
                                    [-1,0]]),
              'R_12*R__23*R__23*I_S':np.array([[0,1],
                                    [1,0]]),
              'R___31*R___31*I_S' :    np.array([[1,0],
                                    [0,-1]]),
              'R__23*R__23*I_S' :    np.array([[-1,0],
                                    [0,1]])
             }

    
    D4_E_irrep13 = {"E" :           np.array([[1,0],
                                    [0,1]]),
              'R___31' :             np.array([[0,1],
                                    [-1,0]]),
              'R___31*R___31*R___31' :   np.array([[0,-1],
                                    [1,0]]),
              'R___31*R___31' :        np.array([[-1,0],
                                    [0,-1]]),
              'R_12*R__23*R_12*I_S':np.array([[0,-1],
                                    [-1,0]]),
              'R___31*R_12*R_12*I_S':np.array([[0,1],
                                    [1,0]]),
               'R__23*R__23*I_S' :    np.array([[1,0],
                                    [0,-1]]),
               'R_12*R_12*I_S' :    np.array([[-1,0],
                                    [0,1]])
             }

    
        
    D4_E_irrep23 = {"E" :           np.array([[1,0],
                                    [0,1]]),
              'R__23' :             np.array([[0,1],
                                    [-1,0]]),
              'R__23*R__23*R__23' :   np.array([[0,-1],
                                    [1,0]]),
              'R__23*R__23' :        np.array([[-1,0],
                                    [0,-1]]),
              'R_12*R_12*R__23*I_S':np.array([[0,-1],
                                    [-1,0]]),
              'R__23*R_12*R_12*I_S':np.array([[0,1],
                                    [1,0]]),
               'R___31*R___31*I_S':    np.array([[1,0],
                                    [0,-1]]),
              'R_12*R_12*I_S':    np.array([[-1,0],
                                    [0,1]])
             }
    
    if mom_key == 'mom001':
        return D4_E_irrep12
    elif mom_key == 'mom100':
        return D4_E_irrep23
    elif mom_key == 'mom010':
        return D4_E_irrep13


def generate_D4h_irreps(taste_key, W3_irreps):
    #### Reference : http://symmetry.jacobs-university.de/cgi-bin/group.cgi?group=604&option=4
    #### As we have 3 different possible D4h groups similar to the 3 different D4 groups for 001,100,010 momentum we generate
    #### irreps for all of them as its usefull in classiying states, there is a connection between taste 00p and mom 001 etc
    #### in terms of D4h at 000, taste 00p subduces to D4 at 001 , taste 000
    
    
    D4h_irreps = {}
    
    taste_dict1 = {'taste_00p' : 'mom001', 'taste_p00' : 'mom100', 'taste_0p0' : 'mom010'}
    taste_dict2 = {'taste_pp0' : 'mom001', 'taste_0pp' : 'mom100', 'taste_p0p' : 'mom010'}
    try:
        D4_E_irrep = generate_D4_E_irrep(taste_dict1[taste_key])
    except:
        D4_E_irrep = generate_D4_E_irrep(taste_dict2[taste_key])
    
    taste_dict = c41.taste_dictionary()
    taste_vec = taste_dict[taste_key]
    
    D4h_group_dict1 = sg.bosonic_little_groups_and_orbits([0,0,0], taste_vec, W3_irreps)[2]

    ### The trivial irrep with negative parity
    D4h_irreps["A_4"] = {key : (-1 if key not in D4_E_irrep.keys() else 1)
                        for key in D4h_group_dict1.keys()}

    
    ### E^+
    E_label = "E_2"
    D4h_irreps[E_label] = {}
    for key5 in D4h_group_dict1.keys():
        if key5 in D4_E_irrep.keys():
            D4h_irreps[E_label][key5] = D4_E_irrep[key5]

        elif key5 not in D4_E_irrep.keys():
            if "I_S" == key5:
                D4h_irreps[E_label][key5] = D4_E_irrep["E"]
            elif "I_S" in key5:
                D4h_irreps[E_label][key5] = D4_E_irrep[key5.rstrip("*I_S")]
            else:
                D4h_irreps[E_label][key5] = D4_E_irrep[key5+"*I_S"]
        else:
            D4h_irreps[E_label][key5] =D4_E_irrep["E"]


    #### All the other 1D irreps that come from a poduct with the trivial irrep with negative inversion parity
    for irrep_index in range(1,4):

        D4h_irreps["A_"+str(irrep_index+4)] = {key : np.multiply(val, W3_irreps["A_"+str(irrep_index)][key]) 
                                              for key, val in D4h_irreps["A_4"].items()}


        ### E^-
        D4h_irreps["E_3"] = {key : np.multiply(val, D4h_irreps[E_label][key]) for key, val in D4h_irreps["A_4"].items()}
    
    return D4h_irreps



def generate_Z2_Z2_irreps(taste_key, W3_irreps):
    #### The two irreps we already have are the trivial irrep and another where the elements with "I_S" are -1
    #### To satisfy orthogonality we always need two -1 hence these irreps are the only choice
    A_23 = {'E': 1, 'R_12*R_12': -1, 'R___31*R___31*I_S': -1, 'R__23*R__23*I_S': 1}
    A_31 = {'E': 1, 'R_12*R_12': -1, 'R___31*R___31*I_S': 1, 'R__23*R__23*I_S': -1}
    
    Z2_Z2_irreps = {}
    if taste_key[-2] == 'p': 
        Z2_Z2_irreps["A_8"] = {key : (A_23[key] if key in A_23.keys() else 1) for key in W3_irreps["A_0"].keys()}
    
        Z2_Z2_irreps["A_9"] = {key : (A_31[key] if key in A_31.keys() else 1) for key in W3_irreps["A_0"].keys()}
        
    elif taste_key[-2] == '0':
        Z2_Z2_irreps["A_9"] = {key : (A_23[key] if key in A_23.keys() else 1) for key in W3_irreps["A_0"].keys()}
    
        Z2_Z2_irreps["A_8"] = {key : (A_31[key] if key in A_31.keys() else 1) for key in W3_irreps["A_0"].keys()}
    
    return Z2_Z2_irreps


########################################################################################################################################
########################################################################################################################################
############################################################### Fermions ###############################################################

########################################################################################################################################
#### Functions for SW3 tilde irrep generation. (central extension of SW3)

def generate_SW3_irrep_threehalf():
    
    J_3_2_x = np.array(((0,sqrt(3),0,0),(sqrt(3),0,2,0),(0,2,0,sqrt(3)), (0,0,sqrt(3),0))) /2
    J_3_2_y = np.array(((0,-1j*sqrt(3),0,0),(1j*sqrt(3),0,-1j*2,0),(0,1j*2,0,-1j*sqrt(3)), (0,0,1j*sqrt(3),0))) /2
    J_3_2_z = np.array(((3,0,0,0),(0,1,0,0),(0,0,-1,0), (0,0,0,-3))) /2
    
    R_12 =  np.round(expm(1j*np.pi/2*J_3_2_x), 10)
    R_23 =  np.round(expm(1j*np.pi/2*J_3_2_y), 10)
    R_31 =  np.round(expm(1j*np.pi/2*J_3_2_z), 10)
    E = np.identity(4)

    dict_rot = {}

    for i in range(1,4):
    
        dict_rot["*R_12"*i] = np.round(matrix_power(R_12, i), 10)
        dict_rot["*R__23"*i] = np.round(matrix_power(R_23, i), 10)
        dict_rot["*R___31"*i] = np.round(matrix_power(R_31, i), 10)
        
    
    dict_copy = dict_rot.copy()
    
    for key1, val1 in dict_copy.items():
        for key2, val2 in dict_copy.items():
            dict_rot[key1+key2] = np.round(np.dot(val1 , val2), 10)
            
    dict2_copy = dict_rot.copy()
            
    for key1, val1 in dict2_copy.items():
        for key2, val2 in dict2_copy.items():
            dict_rot[key1+key2] = np.round(np.dot(val1 , val2), 10)


    dict_rot["E"] = E
    
    unique_set = np.unique(list(dict_rot.values()), axis=0)

    final_dict = {}

    for mat in unique_set:
        list_keys = [key.lstrip("*") for (key, vals) in dict_rot.items() if np.all(vals ==mat)]
        label = sorted(list_keys,key=len)[0]
        final_dict[label] = mat
        
    return final_dict



def generate_SW3_irrep_onehalf(sw3_tilde_faithful):
    sigma_x = np.array(((0, 1), (1, 0)))
    sigma_y = np.array(((0, -1j), (1j, 0)))
    sigma_z = np.array(((1, 0), (0, -1)))
    
    tau_x = sigma_x / 2
    tau_y = sigma_y / 2
    tau_z = sigma_z / 2
    
    
    R_12 =  np.round(expm(1j*np.pi/2*tau_x), 14)
    R_31 =  np.round(expm(1j*np.pi/2*tau_z), 14)
    R_23 =  np.round(expm(1j*np.pi/2*tau_y), 14)
    
    E = np.identity(2)

    dict_rot = {}

    for i in range(1,4):
    
        dict_rot["*R_12"*i] = np.round(matrix_power(R_12, i), 14)        
        dict_rot["*R__23"*i] = np.round(matrix_power(R_23, i), 14)
        dict_rot["*R___31"*i] = np.round(matrix_power(R_31, i), 14)
        
    
    dict_copy = dict_rot.copy()
    
    for key1, val1 in dict_copy.items():
        for key2, val2 in dict_copy.items():
            dict_rot[key1+key2] = np.round(np.dot(val1 , val2), 14)
            
    dict2_copy = dict_rot.copy()
            
    for key1, val1 in dict_copy.items():
        for key2, val2 in dict2_copy.items():
            dict_rot[key1+key2] = np.round(np.dot(val1 , val2), 14)

    dict_rot["E"] = E
    
    cleaned_dict = {key.lstrip("*") : val for (key,val) in dict_rot.items() }
    final_dict = {}
    
    for key in sw3_tilde_faithful.keys():
        final_dict[key] = cleaned_dict[key]
        
    return final_dict


def generate_sw3_tilde_fermionic_irreps():
    
    sw3_tilde_faithful = generate_SW3_irrep_threehalf()
    
    sw3_tilde_irreps = {}
    
    sw3_tilde_irreps['three_half'] = sw3_tilde_faithful
    
    sw3_tilde_irreps['one_half'] = generate_SW3_irrep_onehalf(sw3_tilde_faithful) 
    
    sw3_tilde_irreps['one_half_bar'] = {key : (-val if key.count("R") % 2 == 1 else val) for key, val in sw3_tilde_irreps['one_half'].items()}
    
    return sw3_tilde_irreps


###########################################################################################################################################
### Function to generate a projective representation of W3 which is needed because we area dealing with non abelian taste representations, see sharpe U(R) stuff with cocycles etc

def generate_projective_sw3_rep(clifford_4_1_faithful_irrep, W3_dict):
    # Need a projective irrep that acts on spinor indices (gamma matrix has spinor indices) 
    # Choose generators Juv = i/4[gamma_u, gamma_v]
    # R_ij = exp(-i/2 * theta J_ij) = exp(1/4 gamma_ij) as i!=j
    # This is projective irrep of sw3 and a normal irrep of sw3 tilde, R^4=-1 
    # This then has -1 phases which need to be cancelled by the central extentsion of the little groups
    
    projective_dict = {}
    R_12 = expm(np.pi/4*clifford_4_1_faithful_irrep['gamma_1*gamma_2'])
    R_31 = expm(np.pi/4*clifford_4_1_faithful_irrep['-gamma_1*gamma_3'])
    R_23 = expm(np.pi/4*clifford_4_1_faithful_irrep['gamma_2*gamma_3'])
    
    for key in W3_dict.keys():
        if 'I_S' not in key:
            res = np.identity(4)

            for rotation in key.split('*')[::-1]:
                if rotation == 'R_12':
                    res = R_12 @ res
                elif rotation == 'R__23':
                    res = R_23 @ res
                elif rotation == 'R___31':
                    res = R_31 @ res

            projective_dict[key] = np.round(res,14)
        
    
    return projective_dict
