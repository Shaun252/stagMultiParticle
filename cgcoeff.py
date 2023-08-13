import numpy as np
import pandas as pd
import sympy as sp
import time
import os

import gfunctions as gf
import stagGroup as sg
import translationGroup as tg
import cliff41Group as c41
import naiveGroup as ng
import operatorsTXYZ as ops

from scipy.linalg import block_diag
from multiprocessing import Pool
from functools import partial



##########################################################################################################################################
### Functions to compute subgroups and coset representatives to speed up calculating CG coefficents of the staggered group. 

def restrict_dict(irrep_keys, parent_dict):
    m_dict = {k:parent_dict[k[0]] for k in irrep_keys}
    
    return m_dict


##########################################################################################################################################
#### Function to calculate clebsch gordons using sakata method


def replace_p_pi(string):
    
    #new_string = string.replace('p', '\u03C0', 10)
    
    return string

def clean_matrix(F):
    rounded_mat = np.empty(F.shape, dtype = type(F))
    for i, column in enumerate(F):
        for j, element in enumerate(column):
            rounded_element = sp.expand(element)
            for a in sp.preorder_traversal(rounded_element):
                if isinstance(a, sp.Float):
                    rounded_element = rounded_element.subs(a, round(a, 3))
                    
            rounded_mat[i,j] = rounded_element
    
    return rounded_mat


def matrix_loop_parallel(dp_key, mat, dp_dim, total_direct_sum_dim, var_type, direct_product_dict, irreps_contained_dict_new):
    total_res=np.zeros((dp_dim, total_direct_sum_dim), dtype = var_type)

    #print(dp_key)
    #print('')

    dp_mat = direct_product_dict[dp_key]
    D_diag = list(irreps_contained_dict_new.values())[0][dp_key]
    for irrep_label, irrep_dict in irreps_contained_dict_new.items():
        if irrep_label != list(irreps_contained_dict_new.keys())[0]:

            D_diag = block_diag(D_diag, irrep_dict[dp_key])

    D_diag_H = D_diag.conj().T

    dp_mat = sp.Matrix(dp_mat)
    D_diag_H = sp.Matrix(D_diag_H)

    res = dp_mat * mat * D_diag_H
    #'''
    #print(dp_mat[1,:])
    #print('')
    #print(mat)
    #print('')
    #print(D_diag_H[:,0])
    #print('')
    
    #print(res[1,0])
    #print('')
    #print('')
    
    #print(dp_mat[15,:])
    #print('')
    #print(mat)
    #print('')
    #print(D_diag_H[:,0])
    #print('')
    
    #print(res[15,0])
    #print('')
    #print('')
    
    #print('******************************************')
    '''
    print(dp_mat[29,:])
    print('')
    #print(mat)
    print('')
    print(D_diag_H[:,0])
    print('')
    
    print(res[29,0])
    print('')
    print('')
    
    print(dp_mat[34,:])
    print('')
    #print(mat)
    print('')
    print(D_diag_H[:,0])
    print('')
    
    print(res[34,0])
    print('')
    print('')
    '''
    #print('******************************************')

    return res




def staggered_clebsch_gordons(direct_product_dict, irreps_contained_list, master_dict, W3_irrep_dict, subgroup_keys = False,
                    coset_keys1=False, coset_keys2=False):
    
    mom_dict = tg.mom_dictionary()

    taste_dict = c41.taste_dictionary()

    irreps_contained_dict = restrict_dict(irreps_contained_list, master_dict)
    
    
    dp_irrep_label = list(direct_product_dict.keys())[0]
    
    individual_prod_dim1 = list(master_dict[dp_irrep_label[0]].values())[0].shape[0]
    individual_prod_dim2 = list(master_dict[dp_irrep_label[1]].values())[0].shape[0]
    
    dp_dim = list(direct_product_dict[dp_irrep_label].values())[0].shape[0]
    
    irreps_contained_dim = [1 if np.array(list(master_dict[irreps_contained[0]].values())[0]).shape == () else 
                            list(master_dict[irreps_contained[0]].values())[0].shape[0]
                            for irreps_contained in irreps_contained_dict.keys()]
    
    total_direct_sum_dim = sum(irreps_contained_dim)
            
    
    dp_basis_functions = []
    
    irrep_label1 = dp_irrep_label[0][-3:]
    irrep_label2 = dp_irrep_label[1][-3:]
    
    
    irrep_dim_dict =  {"A" : 1, "E" : 2, "T" : 3}
    
    irrep_dim1 = irrep_dim_dict[irrep_label1[0]]
    irrep_dim2 = irrep_dim_dict[irrep_label2[0]]
    
    mom_key1 = dp_irrep_label[0][:6]
    mom_key2 = dp_irrep_label[1][:6]
    
    taste_key1 = dp_irrep_label[0][-15:-9] + dp_irrep_label[0][-7:-4]
    taste_key2 = dp_irrep_label[1][-15:-9] + dp_irrep_label[1][-7:-4]
    
    taste0_key1 = dp_irrep_label[0][-8]
    taste0_key2 = dp_irrep_label[1][-8]
    
    C_0_key1 = dp_irrep_label[0][-9]
    C_0_key2 = dp_irrep_label[1][-9]
    
    mom_coset_reps = sg.momentum_coset_representatives(W3_irrep_dict=W3_irrep_dict)
    taste_coset_reps = sg.bosonic_taste_coset_representatives(W3_irrep_dict=W3_irrep_dict)
    
    mom_basis_function_coset_rep1 = mom_coset_reps[mom_key1]
    mom_basis_function_coset_rep2 = mom_coset_reps[mom_key2]
    
    taste_basis_function_coset_rep1 = taste_coset_reps[mom_key1 + "_" + taste_key1]
    taste_basis_function_coset_rep2 = taste_coset_reps[mom_key2 + "_" + taste_key2]
    
    no_zeros1 = mom_key1.count('0')
    no_zeros2 = mom_key2.count('0')
    
    #Will need no_ones when have pmu >1
    no_ones1 = mom_key1.count('1')
    no_ones2 = mom_key2.count('1')
    
    if no_zeros1 == 1:
        non_conserved_taste_index1 = mom_key1[-3:].find('0')
    elif no_zeros1 == 2:
        non_conserved_taste_index1 = mom_key1[-3:].find('1')
    else:
        non_conserved_taste_index1 = False
        
    if no_zeros2 == 1:
        non_conserved_taste_index2 = mom_key2[-3:].find('0')
    elif no_zeros2 == 2:
        non_conserved_taste_index2 = mom_key2[-3:].find('1')
    else:
        non_conserved_taste_index2 = False
        
    
    for mom_1_coset_rep in mom_basis_function_coset_rep1:
    
        mom_basis_function_label1 = str(mom_dict[mom_key1] @ W3_irrep_dict["T_1"][mom_1_coset_rep])
    
        for taste_1_coset_rep in taste_basis_function_coset_rep1:
            taste_vec1 = np.mod(taste_dict[taste_key1] @ W3_irrep_dict["T_1"][taste_1_coset_rep] @ W3_irrep_dict["T_1"][mom_1_coset_rep],2*np.pi)
            taste_label1 = ["0" if taste_key == 0 else "p" for taste_key in taste_vec1]
            #if non_conserved_taste_index1 != False:
                #taste_label1 =taste_label1[:non_conserved_taste_index1] + taste_label1[non_conserved_taste_index1+1:]
            #taste_label1 = [taste4_key1] + taste_label1
            taste_basis_function_label1 = "".join(taste_label1)
            
            for k in range(irrep_dim1):

                for mom_2_coset_rep in mom_basis_function_coset_rep2:

                    mom_basis_function_label2 = str(mom_dict[mom_key2] @ W3_irrep_dict["T_1"][mom_2_coset_rep])

                    for taste_2_coset_rep in taste_basis_function_coset_rep2:

                            taste_vec2 = np.mod(taste_dict[taste_key2] @ W3_irrep_dict["T_1"][taste_2_coset_rep] @ W3_irrep_dict["T_1"][mom_2_coset_rep],2*np.pi)
                            taste_label2 = ["0" if taste_key == 0 else "p" for taste_key in taste_vec2]
                            #if non_conserved_taste_index2 != False:
                                #taste_label2 = taste_label2[:non_conserved_taste_index2] + taste_label2[non_conserved_taste_index2+1:]
                            #taste_label2 = [taste4_key2] + taste_label2
                            taste_basis_function_label2 = "".join(taste_label2)
                            
                            for j in range(irrep_dim2):

                                basis_function_label = str((mom_basis_function_label1, taste_basis_function_label1, irrep_label1  + "_" + str(k))) +"_X_" + str((mom_basis_function_label2, taste_basis_function_label2, irrep_label2  + "_" + str(j)))
                                dp_basis_functions.append(replace_p_pi(basis_function_label))
    
    prod_label = replace_p_pi(str(dp_irrep_label[0])) + "_X_" + replace_p_pi(str(dp_irrep_label[1]))
    
    df = pd.DataFrame({prod_label : dp_basis_functions})

    variable_list = ["a"+str(k) for k in range(dp_dim*total_direct_sum_dim)]
    variables = sp.var(','.join(variable_list))    
    U = np.array(variables).reshape(dp_dim, total_direct_sum_dim)
    var_type = type(U)
    U = sp.Matrix(U)
    
    
    if total_direct_sum_dim == 1 and type(list(irreps_contained_dict.values())[0]) == int:
        irreps_contained_dict_new = {}
        for key, vals in irreps_contained_dict.items():
            new_irrep_dict = {}
            for key1, vals1 in vals.items():
                new_irrep_dict[key1] = np.array([[vals1]])
            irreps_contained_dict_new[key] = new_irrep_dict
                
    else:
        irreps_contained_dict_new = irreps_contained_dict
    
    
    partial_mat_loop = partial(matrix_loop_parallel, dp_dim=dp_dim, var_type=var_type, total_direct_sum_dim=total_direct_sum_dim, direct_product_dict=direct_product_dict[dp_irrep_label], irreps_contained_dict_new=irreps_contained_dict_new)
    
    partial_mat_loop1 = partial(partial_mat_loop, mat = U)
    F = sum(list(map(partial_mat_loop1, subgroup_keys)), sp.zeros(U.shape[0], U.shape[1]))

    partial_mat_loop1 = partial(partial_mat_loop, mat = F)
    F = sum(list(map(partial_mat_loop1, coset_keys1)), sp.zeros(U.shape[0], U.shape[1]))

    partial_mat_loop2 = partial(partial_mat_loop, mat = F)
    F = sum(list(map(partial_mat_loop2, coset_keys2)), sp.zeros(U.shape[0], U.shape[1]))
            
    F=np.array(F)
    cleaned_mat = clean_matrix(F)
    
    k=0
    for t, (irrep_key, irrep_vals) in enumerate(irreps_contained_dict.items()):
        irrep_dim = irreps_contained_dim[t]
        if irrep_dim != 1:
            

            for  number in range(irrep_dim):
                df[(replace_p_pi(irrep_key[0]), irrep_key[1]) + (number,)] = cleaned_mat[:,k]
                k+=1

        else:
            df[(replace_p_pi(irrep_key[0]), irrep_key[1]) +  (0,)] = cleaned_mat[:,k]
            k+=1
        
    df_sorted = df.set_index(prod_label)
    
    
    del irreps_contained_dict
    del irreps_contained_dict_new
       
    return df_sorted


def generate_staggered_cg_dataframe_ST(ST1, ST2, total_mom_dict, class_dictionary, N, W3_irreps, cliff_41_irreps, irrep_restrict_key =False, irreps_contained_only = False, odd_charge_conj = True):
    
    
    STGAMMADICT = ops.generate_irrep_spin_taste_dictionary()
    
    irrep_label1 = STGAMMADICT[ST1]
    irrep_label2 = STGAMMADICT[ST2]
    
    if odd_charge_conj == True:
        if irrep_label1[13] == '0':
            irrep_label2 = irrep_label2[:13] +  'p' + irrep_label2[14:]
        elif irrep_label1[13] == 'p':
            irrep_label2 = irrep_label2[:13] +  '0' + irrep_label2[14:]
            
    elif odd_charge_conj == False:
        if irrep_label1[13] == '0':
            irrep_label2 = irrep_label2[:13] +  '0' + irrep_label2[14:]
        elif irrep_label1[13] == 'p':
            irrep_label2 = irrep_label2[:13] +  'p' + irrep_label2[14:]
    
    if irrep_restrict_key != False:
        if irrep_restrict_key[0][1] != 0:
            irrep_restrict_key_list = []
            for st in irrep_restrict_key:
                irrep_restrict_key_list.append((STGAMMADICT[st],0))
        else:
            irrep_restrict_key_list = irrep_restrict_key
    else:
        irrep_restrict_key_list = False
            
    df = generate_staggered_cg_dataframe(irrep_label1, irrep_label2, ST1, ST2, total_mom_dict, class_dictionary, N, W3_irreps, cliff_41_irreps, irrep_restrict_key =irrep_restrict_key_list, irreps_contained_only = irreps_contained_only)
    print('')
    print(df)
    print('')
    cgdfToLatex(df)
    
    return df

def generate_staggered_cg_dataframe(irrep_label1, irrep_label2, ST1, ST2, total_mom_dict, class_dictionary, N, W3_irreps, cliff_41_irreps, irrep_restrict_key =False, irreps_contained_only = False):
    
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', 500)

    coset_keys1, subgroup_keys1 = sg.taste_w3_cosets(W3_irreps=W3_irreps, clifford=cliff_41_irreps)
    coset_keys2 = sg.staggered_mom_cosets(N=N)  
    
    op1 = ops.convertOPKey(ST1)
    op2 = ops.convertOPKey(ST2)
    
    direct_product = gf.direct_product_irrep(irrep_label1, irrep_label2, total_mom_dict)
    irreps_contained_dp = gf.compute_irreps_contained(direct_product_dict=direct_product, master_dict = total_mom_dict, class_dictionary = class_dictionary)
    
    
    STGAMMADICT = ops.generate_irrep_spin_taste_dictionary()
    STGAMMADICT_reversed = {}
    for key, val in STGAMMADICT.items():
        if val not in STGAMMADICT_reversed.keys():
            STGAMMADICT_reversed[val] = [key,]
        else:
            STGAMMADICT_reversed[val].append(key)
            
         
    
    irreps_contained_dp_clean = [i[0] for i in irreps_contained_dp]
    
    irreps_contained_dp_nubmber_clean = [i[1] for i in irreps_contained_dp]
    
    irrepCompleteBreakdown = []
    for ii, irrep_no in  enumerate(irreps_contained_dp_nubmber_clean):
        irrepCompleteBreakdown.append(irreps_contained_dp_clean[ii])
        if irrep_no != 0:
            for kk in range(irrep_no):
                irrepCompleteBreakdown.append(irreps_contained_dp_clean[ii])
                    
    print(irrepCompleteBreakdown)
    print('')
    irrepDecompLatex  = " ::: :::oplus ::: ".join(list(map(sg.cleanMomTasteRotIrrep, irrepCompleteBreakdown)))
    
    
    print(op1 + ' ::: :::otimes ::: ' + op2 + " & ::: :::sim ::: " +  sg.cleanMomTasteRotIrrep(irrep_label1) + ' ::: :::otimes ::: ' + sg.cleanMomTasteRotIrrep(irrep_label2) + ":::nn ::::::[0.5em]")
    try:
        print("= ::: & " + sg.cleanMomTasteRotIrrep(irrepCompleteBreakdown[0])  + ":::quad :::sim :::quad " + ops.convertOPKey(STGAMMADICT_reversed[irrepCompleteBreakdown[0]][0]) +  " ::::::[0.5em]")
    except:
        print("= ::: & " + sg.cleanMomTasteRotIrrep(irrepCompleteBreakdown[0]) +  " ::::::[0.5em]")
    for irrep in irrepCompleteBreakdown[1:]:
        try:
            print(" :::oplus ::: &" + sg.cleanMomTasteRotIrrep(irrep)   + ":::quad :::sim :::quad " + ops.convertOPKey(STGAMMADICT_reversed[irrep][0]) + " ::::::[0.5em]")
        except:
            print(" :::oplus ::: &" + sg.cleanMomTasteRotIrrep(irrep)    + " ::::::[0.5em]")
        
    print('')
    
    if irreps_contained_only == True:
        return irreps_contained_dp
    
    if irrep_restrict_key == False:
    
        cg_dataframe = staggered_clebsch_gordons(direct_product_dict=direct_product, irreps_contained_list=irreps_contained_dp, 
                        master_dict = total_mom_dict, W3_irrep_dict = W3_irreps,
                        subgroup_keys = subgroup_keys1, coset_keys1 = coset_keys1, coset_keys2 = coset_keys2)
    else:
        cg_dataframe = staggered_clebsch_gordons(direct_product_dict=direct_product, irreps_contained_list=irrep_restrict_key, 
                        master_dict = total_mom_dict, W3_irrep_dict = W3_irreps,
                        subgroup_keys = subgroup_keys1, coset_keys1 = coset_keys1, coset_keys2 = coset_keys2)
        
    #print(cg_dataframe)
    df, elements_contained_dict, reversed_dict = relabel_dataframe(cg_dataframe)
    #print('')
    #print(elements_contained_dict)
    

    return df[(df.T != 0).any()]

###############################################################################################################################################

def naive_clebsch_gordons(direct_product_dict, irreps_contained_list, master_dict, W3_irrep_dict, subgroup_keys = False,
                    coset_keys1=False):
    
    mom_dict = tg.mom_dictionary()

    irreps_contained_dict = restrict_dict(irreps_contained_list, master_dict)
    
    
    dp_irrep_label = list(direct_product_dict.keys())[0]
    
    individual_prod_dim1 = list(master_dict[dp_irrep_label[0]].values())[0].shape[0]
    individual_prod_dim2 = list(master_dict[dp_irrep_label[1]].values())[0].shape[0]
    
    dp_dim = list(direct_product_dict[dp_irrep_label].values())[0].shape[0]
    
    irreps_contained_dim = [1 if np.array(list(master_dict[irreps_contained[0]].values())[0]).shape == () else 
                            list(master_dict[irreps_contained[0]].values())[0].shape[0]
                            for irreps_contained in irreps_contained_dict.keys()]
    
    total_direct_sum_dim = sum(irreps_contained_dim)
            
    
    dp_basis_functions = []
    
    irrep_label1 = dp_irrep_label[0][-3:]
    irrep_label2 = dp_irrep_label[1][-3:]
    
    
    irrep_dim_dict =  {"A" : 1, "E" : 2, "T" : 3}
    
    irrep_dim1 = irrep_dim_dict[irrep_label1[0]]
    irrep_dim2 = irrep_dim_dict[irrep_label2[0]]
    
    mom_key1 = dp_irrep_label[0][:6]
    mom_key2 = dp_irrep_label[1][:6]
    
    mom_coset_reps = ng.momentum_coset_representatives(W3_irrep_dict=W3_irrep_dict)
    
    mom_basis_function_coset_rep1 = mom_coset_reps[mom_key1]
    mom_basis_function_coset_rep2 = mom_coset_reps[mom_key2]

            
    for mom_1_coset_rep in mom_basis_function_coset_rep1:
    
        mom_basis_function_label1 = str(mom_dict[mom_key1] @ W3_irrep_dict["T_1"][mom_1_coset_rep])
            
        for k in range(irrep_dim1):

            for mom_2_coset_rep in mom_basis_function_coset_rep2:

                mom_basis_function_label2 = str(mom_dict[mom_key2] @ W3_irrep_dict["T_1"][mom_2_coset_rep])

                for j in range(irrep_dim2):

                    basis_function_label = str((mom_basis_function_label1, irrep_label1  + "_" + str(k))) +"_X_" + str((mom_basis_function_label2, irrep_label2  + "_" + str(j)))
                    dp_basis_functions.append(basis_function_label)
    
    prod_label = str(dp_irrep_label[0]) + "_X_" + str(dp_irrep_label[1])
    
    df = pd.DataFrame({prod_label : dp_basis_functions})

    variable_list = ["a"+str(k) for k in range(dp_dim*total_direct_sum_dim)]
    variables = sp.var(','.join(variable_list))    
    U = np.array(variables).reshape(dp_dim, total_direct_sum_dim)
    var_type = type(U)
    U = sp.Matrix(U)
    
    
    if total_direct_sum_dim == 1 and type(list(irreps_contained_dict.values())[0]) == int:
        irreps_contained_dict_new = {}
        for key, vals in irreps_contained_dict.items():
            new_irrep_dict = {}
            for key1, vals1 in vals.items():
                new_irrep_dict[key1] = np.array([[vals1]])
            irreps_contained_dict_new[key] = new_irrep_dict
                
    else:
        irreps_contained_dict_new = irreps_contained_dict
    
    
    partial_mat_loop = partial(matrix_loop_parallel, dp_dim=dp_dim, var_type=var_type, total_direct_sum_dim=total_direct_sum_dim, direct_product_dict=direct_product_dict[dp_irrep_label], irreps_contained_dict_new=irreps_contained_dict_new)
    
    partial_mat_loop1 = partial(partial_mat_loop, mat = U)
    F = sum(list(map(partial_mat_loop1, subgroup_keys)), sp.zeros(U.shape[0], U.shape[1]))
    
    if coset_keys1 != False:
        partial_mat_loop1 = partial(partial_mat_loop, mat = F)
        F = sum(list(map(partial_mat_loop1, coset_keys1)), sp.zeros(U.shape[0], U.shape[1]))
            
    F=np.array(F)
    cleaned_mat = clean_matrix(F)
    
    k=0
    for t, (irrep_key, irrep_vals) in enumerate(irreps_contained_dict.items()):
        irrep_dim = irreps_contained_dim[t]
        if irrep_dim != 1:
            

            for  number in range(irrep_dim):
                df[irrep_key + (number,)] = cleaned_mat[:,k]
                k+=1

        else:
            df[irrep_key +  (0,)] = cleaned_mat[:,k]
            k+=1
        
    df_sorted = df.set_index(prod_label)
    
    
    del irreps_contained_dict
    del irreps_contained_dict_new
       
    return df_sorted


def generate_naive_cg_dataframe(irrep_label1, irrep_label2, total_mom_dict, class_dictionary, N, W3_irreps, irrep_restrict_key =False, irreps_contained_only = False, Full=False):
    
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', 500)

    subgroup_keys1 = ng.w3_cosets(W3_irreps=W3_irreps)
    coset_keys1 = ng.mom_cosets(N=N)  
        
    
    direct_product = gf.direct_product_irrep(irrep_label1, irrep_label2, total_mom_dict)
    irreps_contained_dp = gf.compute_irreps_contained(direct_product_dict=direct_product, master_dict = total_mom_dict, class_dictionary = class_dictionary)
    print(irreps_contained_dp)
    if irreps_contained_only == True:
        return irreps_contained_dp
    
    if Full != False:
        coset_keys1 = False
        subgroup_keys1 = ng.group_elements(N=N, W3=W3_irreps['T_1'])
    
    if irrep_restrict_key == False:
        
    
        cg_dataframe = naive_clebsch_gordons(direct_product_dict=direct_product, irreps_contained_list=irreps_contained_dp, 
                        master_dict = total_mom_dict, W3_irrep_dict = W3_irreps,
                        subgroup_keys = subgroup_keys1, coset_keys1 = coset_keys1)
    else:
        cg_dataframe = naive_clebsch_gordons(direct_product_dict=direct_product, irreps_contained_list=irrep_restrict_key, 
                        master_dict = total_mom_dict, W3_irrep_dict = W3_irreps,
                        subgroup_keys = subgroup_keys1, coset_keys1 = coset_keys1)
        
    
    
    df, elements_contained_dict, reversed_dict = relabel_dataframe(cg_dataframe)
    print(df[(df.T != 0).any()])
    
    del direct_product
    del irreps_contained_dp
    del cg_dataframe
    return None


##########################################################################################################################################
### Functions to clean, sympify, save and load cg dataframes



def sympify_data_frame(data_frame):
    matrix = data_frame.to_numpy()
    dim = matrix.shape[0]
    new_array = np.zeros((dim, dim), dtype = type(matrix))
    df = data_frame.copy()
    column_headers = list(data_frame.columns.values)
    for i, row in enumerate(matrix):
        for j, element in enumerate(row):
            new_ele = sp.sympify(element)
            new_array[i,j] = new_ele
            
            df.iloc[i,j] = new_array[i,j]
            
    return df

def check_shared_variables(sympified_cg_df):
    ###checks what shared variables are in different rows of irreps, variables should be exclusive to specific irreps i.e
    ### if a0 appears in A_0 irrep it should not appear in E_0, T_0 etc

    for column in sympified_cg_df:
        for element in sympified_cg_df[column]:
            for variable in element.free_symbols:
                #print(variable)
                for column1 in sympified_cg_df:
                    if column1 != column:
                        element_list = []
                        for element1 in sympified_cg_df[column1]:
                            for variable1 in element1.free_symbols:
                                if variable1 == variable and variable not in element_list and column[:25] != column1[:25]:
                                    print(column, column1)
                                    print(variable)
                                    element_list.append(variable)
                                

def relabel_dataframe(data_frame):
    matrix = data_frame.to_numpy()
    dim = matrix.shape[0]
    variable_list = ["b"+str(k) for k in range(dim ** 2)]
    variables = sp.var(','.join(variable_list))
    
    new_array = np.zeros((dim, dim), dtype = type(matrix))
    #print(new_array.shape)
    normalised_array = np.zeros((dim, dim), dtype = type(str))
    element_dict = {}
    elements_contained_dict = {}
    k=0
    df = data_frame.copy()
    column_headers = list(data_frame.columns.values)
    for i, row in enumerate(matrix):
        for j, element in enumerate(row):
            if element != 0:
                free_vars = element.free_symbols
                
                string_split = str(sp.factor(element)).split("*")
                coeff = string_split[0]
                unimproved_var = ("*").join(string_split[1:])

                if unimproved_var not in element_dict.keys():
                    if "-" in coeff:
                        new_array[i,j] = -1*variables[k]
                    else:
                        new_array[i,j] = variables[k]
                    element_dict[unimproved_var] = variables[k]
                    elements_contained_dict[variables[k]] = element
                    k += 1
                if unimproved_var in element_dict.keys():
                    if "-" in coeff:
                        new_array[i,j] = -1*element_dict[unimproved_var]
                    else:
                        new_array[i,j] = element_dict[unimproved_var]
        
        
            df.iloc[i,j] = new_array[i,j]

            
    reversed_dict = dict([[v,k] for k,v in element_dict.items()])
    
    #for j, column in enumerate(new_array.T):
        
        #for b_var in variables[:20]:
            
            #norm = np.count_nonzero(column == b_var) + np.count_nonzero(column == -1*b_var)
            
            #for i, element in enumerate(column):
                
                #if element == b_var:
                    #df.iloc[i,j] = '1/sqrt(' + str(norm) + ')'
                #elif element == -1*b_var:
                    #df.iloc[i,j] = '-1/sqrt(' + str(norm) + ')'

    
    return df, elements_contained_dict, reversed_dict


def save_cg_dataframe(data_frame):
    dp_irrep_name = data_frame.index.name
    mom_key1 = dp_irrep_name[:6]
    mom_key2 = dp_irrep_name[-21:-15]
    
    filename = "clebsch_gordons/" + mom_key1 + "_+_"  + mom_key2 + "/" +  dp_irrep_name
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data_frame.to_csv(filename)
    print("CG_data_frame: " + str(dp_irrep_name) + " saved")
    
def load_cg_dataframe(dp_irrep_name):
    mom_key1 = dp_irrep_name[:6]
    mom_key2 = dp_irrep_name[-21:-15]
    
    filename = "clebsch_gordons/" + mom_key1 + "_+_"  + mom_key2 + "/" +  dp_irrep_name
    
    dframe = pd.read_csv(filename).set_index(dp_irrep_name)
    
    sympified_cg_df = sympify_data_frame(dframe)
                        
    return sympified_cg_df



def cleanColumnLabelCG(columnLabel):
    opKey = ops.convertOPKeyFromIrrep(columnLabel[0])
    return "$" + opKey + "$" 

def cleanIndexKey(indexKey, indexname):
    
    
    
    irrep1= indexname.split('X')[0].rstrip('_')
    irrep2= indexname.split('X')[1].lstrip('_')
    
    C_0_1 = irrep1[-9]
    taste_key1 = irrep1[-8:-4]
    
    C_0_2 = irrep2[-9]
    taste_key2 = irrep2[-8:-4]

    indexKey1, indexKey2 =  indexKey.split('X')
    momKey1, spatialTasteKey1, IrrepKey1 = indexKey1.rstrip('_').replace(".", "").replace("'", "").replace("]", "").replace("[", "").replace("(", "").replace(")", "").split(',')
    momKey2, spatialTasteKey2, IrrepKey2 = indexKey2.lstrip('_').replace(".", "").replace("'", "").replace("]", "").replace("[", "").replace("(", "").replace(")", "").split(',')

    
    momVec1 = str(tuple(map(int, ' '.join(momKey1.split()).split(" "))))
    momVec2 = str(tuple(map(int, ' '.join(momKey2.split()).split(" "))))
    
    tasteKeyFull1 = C_0_1  + taste_key1[0] + spatialTasteKey1.replace(' ', "") 
    tasteKeyFull2 = C_0_2 + taste_key2[0] + spatialTasteKey2.replace(' ', "")  
    
    rotIrrep1 = irrep1[-3:]
    rotIrrep2 = irrep2[-3:]
    
    mom_key1 = irrep1[:6]
    mom_key2 = irrep2[:6]
    
    #irrep1Row = mom_key1 + '_taste_' + tasteKeyFull1 + '_' + rotIrrep1
    #irrep2Row = mom_key2 + '_taste_' + tasteKeyFull2 + '_' + rotIrrep2
    
    
    irrepKeyFull1 = ops.convertOPKeyFromIrrep(irrep1)
    irrepKeyFull1 = ",".join(irrepKeyFull1.split(",")[:-3]+[momVec1,])
    
    try:
        irrepKeyFull2 = ops.convertOPKeyFromIrrep(irrep2)
        irrepKeyFull2 = ",".join(irrepKeyFull2.split(",")[:-3]+[momVec2,])
    except:
        irrepKeyFull2 = irrepKeyFull1
    
    return "$" + irrepKeyFull1 + " ::: :::otimes ::: " + irrepKeyFull2 + "$" 

def cleanIndexName(indexName, caption=False):
    
    irrep1= indexName.split('X')[0].rstrip('_')
    irrep2= indexName.split('X')[1].lstrip('_')
    if caption == False:
        irrep1Clean, irrep2Clean = sg.cleanMomTasteRotIrrep(irrep1), sg.cleanMomTasteRotIrrep(irrep2)
    else:
        irrep1Clean, irrep2Clean = sg.cleanMomTasteRotIrrepCaption(irrep1), sg.cleanMomTasteRotIrrepCaption(irrep2)
    
    return "$" + irrep1Clean + "::: :::otimes ::: " + irrep2Clean + "$" 

def cgdfToLatex(df):
    
    targetIrrep  = sg.cleanMomTasteRotIrrep(df.columns[0][0])
    targetIrrepCaption  = sg.cleanMomTasteRotIrrepCaption(df.columns[0][0])

    df.columns = df.columns.to_series().apply(cleanColumnLabelCG)
    IrrepDp = cleanIndexName(df.index.name)
    IrrepDpCaption = cleanIndexName(df.index.name, caption=True)
    
    cleanIndexKeyFull = partial(cleanIndexKey, indexname=df.index.name)
    df.index.name = "" 
    df.index = df.index.to_series().apply(cleanIndexKeyFull)
    
    print(df.to_latex(escape = False, caption = ("Clebsch Gordon table for "+ IrrepDp + " $:::to " + targetIrrep +"$. The irrep rows are labelled by the operators which excite them." ,"Clebsch Gordon table for "+ IrrepDpCaption + " $:::to " + targetIrrepCaption +"$.")))