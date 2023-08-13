import numpy as np
import itertools

'''Functions for \Gamma_4_1 clifford group irrep generataion (taste group + charge conj)'''

########################################################################################################################################
### Irrep keys for spatial part of bosonic \Gamma_4_1

def taste_dictionary():
    
    pi = np.pi
    tastes = {"taste_000" : [0,0,0],
              "taste_00p" : [0,0,pi],
              "taste_pp0" : [pi,pi,0],
              "taste_0pp" : [0,pi,pi],
              "taste_ppp" : [pi,pi,pi],
              "taste_p0p" : [pi,0,pi],
              "taste_p00" : [pi,0,0],
              "taste_0p0" : [0,pi,0]}
    
    return tastes


########################################################################################################################################


def cliff_4_1_bosonic_character(charge_conj, taste_four, taste_three_vec, group_element):
    
    
    cliff_4_1_irrep_key = np.mod((charge_conj,) + (taste_four,) + taste_three_vec, 2*np.pi)
    
    cliff_4_1_irrep_irrep_vec = np.real(np.exp(np.multiply(1j , cliff_4_1_irrep_key)))
    
    character_res_arr = np.ones((len(cliff_4_1_irrep_irrep_vec)))
    for mu, Xi_mu in enumerate(group_element):
        if Xi_mu == 1:
            character_res_arr[mu] = cliff_4_1_irrep_irrep_vec[mu]
            
    char_val = np.prod(character_res_arr)  
    
    return np.real(char_val)

########################################################################################################################################

def generate_clifford_4_faithful_irrep():
    ### Taken form the appendix of Gattringer & Lang, the chiral rep
    gamma_dict = {
        
    "E" : np.identity(4),
        
    "gamma_1" : np.array([ [ 0 , 0 ,0, -1j ] ,[ 0 , 0 , -1j, 0 ], [ 0 ,1j, 0 , 0 ] ,[1j, 0, 0, 0 ]]),
    
    "gamma_2" :  np.array([ [ 0 , 0 ,0, -1 ] ,[ 0 , 0 , 1, 0 ], [ 0, 1, 0 , 0 ] ,[-1, 0, 0, 0 ]]),
    
    "gamma_3" :  np.array([ [ 0 , 0 ,-1j, 0 ] ,[ 0 , 0 , 0, 1j ], [ 1j, 0 , 0 , 0 ] ,[0, -1j, 0, 0 ]]),
    
    "gamma_0" :  np.array([ [ 0 , 0 ,1, 0 ] ,[ 0 , 0 , 0, 1 ], [1, 0 , 0 , 0 ] ,[0, 1, 0, 0 ]])
                          }
    
    
    for comb_length in range(2,5):
        combinations = itertools.combinations((0,1,2,3), comb_length)
        for combination in combinations:
            label_template = "gamma_"
            mat = gamma_dict["E"]
            
            label = ""
            for number in combination:
                
                label += label_template +str(number) +"*"
                
                gamma = gamma_dict["gamma_"+str(number)]
                mat = np.dot(mat, gamma)
                
                
            gamma_dict[label.rstrip("*")] = mat
            
            
    for gamma_key, gammas in gamma_dict.copy().items():
        gamma_dict["-"+gamma_key] = -1.0*gammas
            
    return gamma_dict

########################################################################################################################################

def generate_clifford_4_1_faithful_irreps():
    ### Taken form the appendix of Gattringer & Lang, the chiral rep
    
    clifford_faithful_irrep_4 = generate_clifford_4_faithful_irrep()
    
    gamma_dict = clifford_faithful_irrep_4.copy()
    
    gamma_5 = gamma_dict["gamma_0"] @ gamma_dict["gamma_1"] @ gamma_dict["gamma_2"] @ gamma_dict["gamma_3"]
    
    C_0 = 1j*gamma_5
    C_0_bar = -1j*gamma_5
    
    gamma = {}
    gamma_bar = {}
    
    for gamma_key, gammas in gamma_dict.items():
        
        gamma[gamma_key] = gammas
        gamma_bar[gamma_key] = gammas
        
        if '-' in gamma_key:
            gamma_key = gamma_key.replace("-", "")
            gamma["-C_0*"+gamma_key] = C_0 @ gammas
            gamma_bar["-C_0*"+gamma_key] = C_0_bar @ gammas
        else:
            gamma["C_0*"+gamma_key] = C_0 @ gammas
            gamma_bar["C_0*"+gamma_key] = C_0_bar @ gammas
    
    del gamma["C_0*"+"E"]
    del gamma["-C_0*"+"E"]
    
    del gamma_bar["C_0*"+"E"]
    del gamma_bar["-C_0*"+"E"]
    
    gamma["C_0"] = C_0
    gamma["-C_0"] = C_0_bar
    
    gamma_bar["C_0"] = C_0_bar
    gamma_bar["-C_0"] = C_0
    
    full_faithful_dict = {'gamma' : gamma, 'gamma_bar' : gamma_bar}
    
    return full_faithful_dict

########################################################################################################################################

def generate_clifford_4_1_bosonic_irreps(clifford_4_1_faithful_dict):
    # Bosonic irreps commute so need to worry about minus sign
    irrep_labels = itertools.product((np.pi,0),repeat=5)
    
    complete_bosonic_irreps_dict = {}
    
    for irrep_label in irrep_labels:
        
        bosonic_irrep_dict = {}
    
        for key in clifford_4_1_faithful_dict.keys():
            group_element = np.zeros((5), dtype=float)
            
            if 'C_0' in key:
                cliff_number_list = [0]
                new_key = key.lstrip('C_0')
                new_key = key.lstrip('-C_0')

            else:
                cliff_number_list = []
                new_key = key
                
            gamma_list = [int(i)+1 for i in new_key if i.isdigit()]
            cliff_number_list += gamma_list
            
            for numbers in cliff_number_list:
                group_element[numbers] = 1
                
            bosonic_irrep_dict[key] = cliff_4_1_bosonic_character(irrep_label[0], irrep_label[1], irrep_label[2:], group_element)
        complete_bosonic_irreps_dict[irrep_label] = bosonic_irrep_dict
        
    return complete_bosonic_irreps_dict

########################################################################################################################################

def generate_clifford_4_1_complete_irrep_dict():
    
    
    clifford_faithful_dict = generate_clifford_4_1_faithful_irreps()
    
    bosonic_irreps = generate_clifford_4_1_bosonic_irreps(clifford_4_1_faithful_dict=clifford_faithful_dict['gamma'])
    
    clifford_faithful_dict.update(bosonic_irreps)
    
    return clifford_faithful_dict