import numpy as np
import pandas as pd
from numpy.linalg import matrix_power
from numpy.linalg import inv
import itertools
from sympy.utilities.iterables import multiset_permutations
from collections import defaultdict


#########################################################################################################################################
### General Useful Functions

def what_matrix(matrix, matrix_dict):
    ### Pass a matrix, return a key / label of that matrix
    for key, items in matrix_dict.items():
        if np.all(np.abs(matrix-items)<=0.001):
            return key

def group_restriction(parent_group, subgroup):
    restricted_dict = {}
    for key, vals in subgroup.items():
        restricted_dict[key] = parent_group[key]
        
    return restricted_dict

def irrep_dict_restriction(irrep_dict, subgroup):
    restricted_irrep_dict = {}
    for key, vals in irrep_dict.items():
        restricted_irrep_dict[key] = group_restriction(vals, subgroup)
        
    return restricted_irrep_dict
    

def group_restriction_list(parent_group, subgroup_list):
    restricted_dict = {}
    for key in subgroup_list:
        restricted_dict[key] = parent_group[key]
        
    return restricted_dict

def irrep_dict_restriction_list(irrep_dict, subgroup):
    restricted_irrep_dict = {}
    for key, vals in irrep_dict.items():
        restricted_irrep_dict[key] = group_restriction_list(vals, subgroup)
        
    return restricted_irrep_dict

def group_restriction_class_dict(parent_group, subgroup):
    restricted_dict = {}
    for key, vals in subgroup.items():
        for class_ele_key in vals:
            try:
                restricted_dict[key] = parent_group[class_ele_key]
                break
            except:
                continue
        
    return restricted_dict

def irrep_dict_restriction_class_dict(irrep_dict, subgroup):
    restricted_irrep_dict = {}
    for key, vals in irrep_dict.items():
        restricted_irrep_dict[key] = group_restriction_class_dict(vals, subgroup)
        
    return restricted_irrep_dict

#########################################################################################################################################
### Group Theory Functions

def key_len(key):
    if type(key) != type((1,1)):
        return len(key)
    else:
        basekey = ''
        for subkey in key:
            basekey+= str(subkey)
        
        return len(basekey)

def conjugate_classes(matrix_dict):
    ### Generated from dictionary which is "keyed" by group element labels and has "values" of corresponding matrices.
    ### Generates a dictionary "keyed" by the represenative element of each class and "valued" by all the elements 
    ### in the class.
    conjugate_dict = {}
    total_list = []
    
    inverse_dict = {key : inv(val) for (key, val) in matrix_dict.items()}
    
    #i=0
    for key1, val1 in matrix_dict.items():
        conjugate_list = []
        if key1 not in total_list:
            #i+=1
            conjugate_list.append(key1)
            for key2, val2 in matrix_dict.items():
                    
                    conjugate_label = what_matrix(val2 @ val1 @ inverse_dict[key2], matrix_dict)
                    
                    if conjugate_label not in conjugate_list:
                        conjugate_list.append(conjugate_label)
        
            conjugate_dict[sorted(conjugate_list,key=key_len)[0]] = conjugate_list
        
            total_list = total_list + conjugate_list
            #print(i)
    
    return conjugate_dict

              
def compute_characters(matrix_dict, faithful_matrix_dict, class_dictionary = False):
    ### Compute a character dictionary labelled by representative class element label and keyed by the character
    
    if class_dictionary == False:
        class_dict = conjugate_classes(faithful_matrix_dict)
    else:
        class_dict = class_dictionary
        
    class_character_dict = {}
    if np.array([list(matrix_dict.values())[0]]).shape !=(1,):
        
        for key in class_dict.keys():
        
            class_character_dict[key] = np.round(np.trace(matrix_dict[key]),4)
    else:
        for key in class_dict:
        
            class_character_dict[key] = np.round(matrix_dict[key],4)
            
        
    return class_character_dict


def compute_characters_irrep_dict(irrep_dict, faithful_matrix_dict, class_dictionary = False):
    char_irrep_dict = {}
    for key, vals in irrep_dict.items():
        char_irrep_dict[key] = compute_characters(vals, faithful_matrix_dict, class_dictionary)
        
    return char_irrep_dict


def compute_characters_no_class(matrix_dict):
    ### Compute a character dictionary labelled by representative class element label and keyed by the character
    
    matrix_char_dict = {}
    if np.array([list(matrix_dict.values())[0]]).shape !=(1,):
        
        for key, values in matrix_dict.items():
        
            matrix_char_dict[key] = np.trace(values)
    else:
        matrix_char_dict = matrix_dict.copy()
            
        
    return matrix_char_dict

def compute_characters_no_class_irrep_dict(irrep_dict):
    char_irrep_dict = {}
    for key, vals in irrep_dict.items():
        char_irrep_dict[key] = compute_characters_no_class(vals)
        
    return char_irrep_dict

def check_orthogonality(matrix_dict1, matrix_dict2, class_dictionary = False):
    
    ### Checks orthogonality of characters
    if class_dictionary == False:
        character_dict1 = compute_characters_no_class(matrix_dict1)
        character_dict2 = compute_characters_no_class(matrix_dict2)
        
        res =  sum([np.multiply(character_dict1[key], np.conj(character_dict2[key])) 
                for key in matrix_dict1.keys()])
        
        G = len(matrix_dict1)
    
    else:
        character_dict1 = compute_characters(matrix_dict1, "dummy", class_dictionary = class_dictionary)
        character_dict2 = compute_characters(matrix_dict2, "dummy", class_dictionary = class_dictionary)
        res =  sum([np.multiply(np.multiply(character_dict1[key], np.conj(character_dict2[key])), len(class_dictionary[key])) 
                for key in class_dictionary.keys()])
    
        G = sum([len(vals) for vals in class_dictionary.values()])
    
    res = np.round(res / G, 4)
    if res == 1:
        return False
        #return res
    elif res == 0:
        return True
        #return res
    else:
        return False
        #return res
        
def check_orthogonality_v2(matrix_dict1, matrix_dict2, class_dictionary = False):
    
    ### Checks orthogonality of characters (or alternatively tells you how many times irrep1 is in rep 2)
    if class_dictionary == False:
        character_dict1 = compute_characters_no_class(matrix_dict1)
        character_dict2 = compute_characters_no_class(matrix_dict2)
        
        res =  sum([np.multiply(character_dict1[key], np.conj(character_dict2[key]))
                for key in matrix_dict1.keys()])
        G = len(matrix_dict1)
    
    else:
        character_dict1 = compute_characters(matrix_dict1, "dummy", class_dictionary = class_dictionary)
        character_dict2 = compute_characters(matrix_dict2, "dummy", class_dictionary = class_dictionary)
        res =  sum([np.multiply(np.multiply(character_dict1[key], np.conj(character_dict2[key])), len(class_dictionary[key])) 
                for key in class_dictionary.keys()])
    
        G = sum([len(vals) for vals in class_dictionary.values()])
    
    res = np.round(res / G, 4)
    return res
    
        
        
def check_irreducibility(matrix_dict, class_dictionary = False):
    ### Checks irreducibility of "matrix dict" of an irrep using the class structure from a "faithful matrix dict"
    
    if class_dictionary == False:
        character_dict = compute_characters_no_class(matrix_dict)
        res =  sum([np.power(np.absolute(character_dict[key]), 2)  for key in matrix_dict.keys()])
        
        G = len(matrix_dict)
    
    else:
        character_dict = compute_characters(matrix_dict, "dummy", class_dictionary = class_dictionary)
        res =  sum([np.power(np.absolute(character_dict[key]), 2) * len(class_dictionary[key])  for key in character_dict.keys()])
    
        G = sum([len(vals) for vals in class_dictionary.values()])
    
        
    no_of_irreps =  np.round(res / G,4)
    
    return no_of_irreps



def check_complete_irrep_dict(irrep_dict, faithful_irrep):
    
    classes = conjugate_classes(faithful_irrep)
    
    number_of_reps = len(irrep_dict)
    group_order = len(list(irrep_dict.values())[0])
    
    number_of_irreps = 0
    number_of_orthogonal_irreps = 0
    
    number_of_group_vecs = 0
    number_of_orthog_group_vecs = 0
    
    irrep_dim_sum = 0
    
    group_element_orthog_dict = defaultdict(list)

    
    for irrep_label1, irrep_value1 in irrep_dict.items():
        
        if np.array([list(irrep_value1.values())[0]]).shape !=(1,):
            irrep_dim = list(irrep_value1.values())[0].shape[0]
        else:
            irrep_dim = 1
        
        irrep_dim_sum += irrep_dim ** 2
        
        character_dict1 = compute_characters(irrep_value1, "dummy", class_dictionary = classes)
        
        for group_element_label, group_element in character_dict1.items():
        
            group_element_orthog_dict[group_element_label].append(character_dict1[group_element_label])
        
        for irrep_label2, irrep_value2 in irrep_dict.items():
            irrep_orthog = check_orthogonality_v2(irrep_value1, irrep_value2, class_dictionary = classes)
            if irrep_orthog == 1.0:
                number_of_irreps += 1
            if irrep_orthog == 0.0:
                number_of_orthogonal_irreps += 1
                
    for group_element_label1, group_element_vec1 in group_element_orthog_dict.items():
        for group_element_label2, group_element_vec2 in group_element_orthog_dict.items():
            element_orthog = np.sum(np.multiply(group_element_vec1,  np.conj(group_element_vec2))) / group_order * len(classes[group_element_label1])
            if element_orthog == 1.0:
                number_of_group_vecs += 1
            if element_orthog == 0.0:
                number_of_orthog_group_vecs += 1
            
    truth_list = np.full((5), False)
    
    if number_of_irreps == number_of_reps:
        print("All reps normalised")
        truth_list[0]= True
        
    if number_of_orthogonal_irreps == number_of_reps*number_of_reps - number_of_reps:
        print("All reps orthogonal to each other")
        truth_list[1]= True
        
    if number_of_group_vecs == number_of_irreps:
        print("All element vectors normalised")
        truth_list[2]= True
        
    if number_of_orthog_group_vecs == number_of_irreps*number_of_irreps-number_of_irreps:
        print("All element vectors orthogonal to eachother")
        truth_list[3]= True
        
    if irrep_dim_sum == group_order:
        print("All reps dimensions satisfy completeness requirement")
        truth_list[4]= True
        
    if np.all(truth_list):
        print("Complete set of irreps")
        

###########################################################################################################################################
###      
        
def generate_character_table(irrep_dict, faithful_matrix_dict, class_dictionary = False):

    if class_dictionary == False:
        conjugate_classes_dict = conjugate_classes(faithful_matrix_dict)
        
    else:
        conjugate_classes_dict = class_dictionary

    class_size_dict = {key : len(value) for key, value in conjugate_classes_dict.items()}
    
    df = pd.DataFrame({'Class size' : class_size_dict})
    
    for key, val in irrep_dict.items():

        if class_dictionary == False:
            character_dict = compute_characters(val, faithful_matrix_dict)
            df[key] = pd.Series(character_dict)

        else:
            
            character_dict = compute_characters(val, faithful_matrix_dict, class_dictionary = class_dictionary)
            df[key] = pd.Series(character_dict)
            
    df_sorted = df.sort_values('Class size', ascending=True)
    
    df_sorted = df_sorted.sort_values('E', ascending=True, axis=1)
    
    df.index.name = 'Class representative'

    
    
    return df_sorted #df_sorted


def check_character_table(irrep_character_table):
    ### Check orthogonality of all characters in character table, column 0 is (representative)element label, 
    ### column 1 is class size (rows are classes, columns 2: are irreps)
    
    
 
    class_size = irrep_character_table["Class size"]
    
    G= sum(class_size)
    
    for column1 in irrep_character_table:
        if column1 != "Class size":
            for column2 in irrep_character_table:
                if column2 != "Class size":
            
                    res =  sum(np.multiply(np.multiply(irrep_character_table[column1], np.conj(irrep_character_table[column2])), class_size))

                    res = res / G

                    print(str(column1) + " - " + str(column2))
                    print("Is orthogonal?")
                    print(res)
                    print("************")

###########################################################################################################################################
###

def check_block_diagonal(matrix, dimensions, allow_permutation = True):
    ### Takes a matrix a set of block dimensions and tells you the permutation of the block dimensions i.e input would be
    ### dimension = (1,2,3) and the matrix has block dimension (3,1,2) it will verify and return (3,1,2)
    if allow_permutation == True:
        permutations = multiset_permutations(dimensions)
    else:
        permutations = [dimensions]
    val = False
    for perm in permutations:
        truth_arr = []
        for k in range(len(dimensions)-1):
            a = sum(perm[:k+1])
            b = sum(perm[:k])
            
            lower_sub_block = matrix[a:,b:a]
            upper_sub_block = matrix[b:a,a:]
            if np.all(np.absolute(upper_sub_block) == 0) and np.all(np.absolute(lower_sub_block) == 0):
                truth_arr.append(True)
                
            else:
                truth_arr.append(False)
        
        if np.all(truth_arr) == True:
            val = True
            break
                
    return val, perm

def jordan_array(matrix_dict, P):
    new_dict = {}
    try:
        inv_P = inv(P)
        for key, vals in matrix_dict.items():
            new_dict[key] = np.dot(np.dot(inv_P, vals), P)
        
        return new_dict
    except:
        return False


def attempt_block_diagonalise_rep(matrix_dict, class_dictionary = False):
    
    matrix_dim = list(matrix_dict.values())[0].shape[0]
    no_of_irreps_contained = int(gf.check_irreducibility(matrix_dict, class_dictionary = class_dictionary))
    
    possible_dimensions = compute_direct_sum_dimensions(matrix_dim=matrix_dim, 
                                                                   no_of_irreps_contained=no_of_irreps_contained)
    i=0
    for key, vals in matrix_dict.items():
        i+=1
        print(i)
        P, J = Matrix(vals).jordan_form()
        P = np.array(P).astype(dtype=float)
        new_matrix_dict = jordan_array(matrix_dict=matrix_dict, P=P)
        
        if new_matrix_dict != False:

            for dimensions in possible_dimensions:
                reduced_form_dict, dimension_irreps = check_irrep_reduced_form(matrix_dict=new_matrix_dict, 
                                                                               dimensions = dimensions)

                if reduced_form_dict != False:
                    break

                    return P, reduced_form_dict, dimension_irreps
        
    return False   


###########################################################################################################################################
###

def irrep_dimension_function(number_of_irreps, G, inversion = False):
    ### Takes number of irreps, order of the group and whether or not the inversion element is in the group and computes
    ### the "hopefully" correct number of irrep dimensions
    soln_iter = itertools.combinations_with_replacement(range(1,20), number_of_irreps)
    for soln in soln_iter:
        res = np.sum(np.power(soln, 2))
        if inversion == False:
            if res == G :
                return list(soln)
                break
                
        else:
            counts = [soln.count(unique_dim) for unique_dim in np.unique(soln)]
            
            if res == G and np.all(np.mod(counts, 2) ==0):
                return list(soln)
                break
                
    return False

def irrep_dimension_function_v2(number_of_irreps, already_known_irrep_dict, G, inversion = False):
    ### Takes number of irreps, order of the group and whether or not the inversion element is in the group and computes
    ### the "hopefully" correct number of irrep dimensions
    
    known_irrep_dims = [list(matrix_dict.values())[0].shape[0] for matrix_dict in already_known_irrep_dict.values()]
    #print(known_irrep_dims)
    guessing_dim = len(already_known_irrep_dict)
    print(guessing_dim)
    print(number_of_irreps-guessing_dim)
    soln_iter = itertools.combinations_with_replacement(range(1,20), number_of_irreps-guessing_dim)
    for soln in soln_iter:
        improved_soln = known_irrep_dims + list(soln)
        res = np.sum(np.power(improved_soln, 2))
        
        if inversion == False:
            if res == G :
                print( soln)
                
                
        else:
            counts = [soln.count(unique_dim) for unique_dim in np.unique(soln)]
            
            if res == G and np.all(np.mod(counts, 2) ==0):
                print( soln)
                
                
    return False

def compute_direct_sum_dimensions(matrix_dim, no_of_irreps_contained):
    ### Takes the dimension of a general matrix in representation 
    ### (which is more than likely a direct product representation) and also the number of irreps 
    ### contained in the representation (calculated with check_irreducibility() function) and out puts all possible
    ### sets of dimensions of length = no_of_irreps contained
    soln_list = []
    soln_iter = itertools.combinations_with_replacement(range(1,matrix_dim), no_of_irreps_contained)
    for soln in soln_iter:
        if sum(soln)==matrix_dim:
            soln_list.append(soln)
            
    return soln_list
    

def check_irrep_reduced_form(matrix_dict, dimensions):
    ### Checks if a whole irrep is in block diagonal form for a specific set of dimensions for each block (1,2,3) would be 
    ### an example for 6 dimensional matrix
    perm_list = []
    permutations = multiset_permutations(dimensions)
    for perms in permutations:
        reduced_form = True
        for key, vals in matrix_dict.items():
            check_value, perm = check_block_diagonal(vals, dimensions, allow_permutation = True)
            if check_value == False:
                reduced_form = False
                break
                
            
    return reduced_form, perm


##########################################################################################################################################
### Cosets and Coset Representatives                  
                    
                    
def right_cosets(group_dict, subgroup_dict):
    #### Generate """"""RIGHT"""""" cosets of a group under a subgroup
    
    total_list = []
    coset_dict = {}
    i=1
    base_coset = list(subgroup_dict.keys())
    coset_dict["coset0"] = list(subgroup_dict.keys())
    total_list += base_coset
    for key1, vals1 in group_dict.items():
        if np.all([np.any(key1 != label) for label in total_list]):
            #print(total_list)
            
            coset = [what_matrix(group_dict[sg_element] @ vals1, group_dict) for sg_element in subgroup_dict.keys()]
            
            if coset[0] in total_list:
                print("Problem, maybe not faithful?")
                break
            
            total_list += coset
            
            coset_dict["coset"+str(i)] = coset
            i+=1
    
    return coset_dict    


def left_cosets(group_dict, subgroup_dict):
    #### Generate """"""Left"""""" cosets of a group under a subgroup
    
    total_list = []
    coset_dict = {}
    i=1
    base_coset = list(subgroup_dict.keys())
    coset_dict["coset0"] = list(subgroup_dict.keys())
    total_list += base_coset
    for key1, vals1 in group_dict.items():
        if np.all([np.any(key1 != label) for label in total_list]):
            #print(total_list)
            
            coset = [what_matrix(vals1 @ group_dict[sg_element], group_dict) for sg_element in subgroup_dict.keys()]
            
            if coset[0] in total_list:
                print("Problem, maybe not faithful?")
                break
            
            total_list += coset
            
            coset_dict["coset"+str(i)] = coset
            i+=1
    
    return coset_dict    
                    

##########################################################################################################################################
### Direct product functions


def direct_product_irrep(irrep_key1, irrep_key2, master_dict):
    
    irrep_dict1 = master_dict[irrep_key1]
    irrep_dict2 = master_dict[irrep_key2]
    
    direct_prod_irrep_dict = {}
    direct_prod_irrep_dict_labelled = {}
    for irrep_element_label1, irrep_element1 in irrep_dict1.items():
        direct_prod_irrep_dict[irrep_element_label1] = np.kron(irrep_element1, irrep_dict2[irrep_element_label1])
    
    direct_prod_irrep_dict_labelled[irrep_key1, irrep_key2] = direct_prod_irrep_dict
        
    return direct_prod_irrep_dict_labelled


def compute_irreps_contained_v2(direct_product_dict, master_dict, class_dictionary = False):
    irrep_dict = list(direct_product_dict.values())[0]
    irreps_contained_dict = {}
    for irrep_label, irrep in master_dict.items():
        number_times_irrep_contained = int(np.real(check_orthogonality_v2(irrep_dict, irrep, 
                                                                 class_dictionary = class_dictionary)))
        
        for number in range(number_times_irrep_contained):
            irreps_contained_dict[irrep_label , number] = irrep
            
    return irreps_contained_dict

def compute_irreps_contained(direct_product_dict, master_dict, class_dictionary = False):
    irrep_dict = list(direct_product_dict.values())[0]
    irreps_contained_list = []
    for irrep_label, irrep in master_dict.items():
        number_times_irrep_contained = int(np.real(check_orthogonality_v2(irrep_dict, irrep, 
                                                                 class_dictionary = class_dictionary)))
        
        for number in range(number_times_irrep_contained):
            irreps_contained_list.append((irrep_label , number))
            
    return irreps_contained_list

