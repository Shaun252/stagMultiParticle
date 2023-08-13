import numpy as np
from itertools import permutations

import gfunctions as gf

import W3RotationGroup as W3
import restframeIsoGroup as rf
import cliff41Group as c41
import stagGroup as sg
import translationGroup as tg

import operatorsTXYZ as ops





########################################################################################################################################
#### Character dicts for continuum decomp

def generate_O4_restrricted_SW4_character_dict():
    ### Only generating bosonic low spin irreps
    
    
    SW4_char_dict = rf.generate_SW4_character_dict()
    irrep_dict = {}
    
    irrep_dict["trivial_irrep"] = SW4_char_dict["trivial_irrep"]
    irrep_dict["one_zero"] = SW4_char_dict["one_zero"]
    irrep_dict["zero_one"] = SW4_char_dict["zero_one"]
    
    #######################################################################
    two_zero = {}
    for key in SW4_char_dict["one_zero_bar"].keys():
         two_zero[key] = SW4_char_dict["one_zero_bar"][key] + SW4_char_dict["two_two"][key]
        
    irrep_dict["two_zero"] = two_zero
    
    #######################################################################
    
    zero_two = {}
    for key in SW4_char_dict["zero_one_bar"].keys():
         zero_two[key] = SW4_char_dict["zero_one_bar"][key] + SW4_char_dict["two_two"][key]
        
    irrep_dict["zero_two"] = zero_two
    
    #######################################################################
    three_zero = {}
    for key in SW4_char_dict["one_zero"].keys():
         three_zero[key] = SW4_char_dict["one_zero"][key] + SW4_char_dict["one_zero_bar"][key] + SW4_char_dict["anti_sym"][key]
        
    irrep_dict["three_zero"] = three_zero
    
    #######################################################################
    
    zero_three = {}
    for key in SW4_char_dict["zero_one"].keys():
         zero_three[key] = SW4_char_dict["zero_one"][key] + SW4_char_dict["zero_one_bar"][key] + SW4_char_dict["anti_sym"][key]
        
    irrep_dict["zero_three"] = zero_three
    
    return irrep_dict

def generate_D4_irreps_for_SW4(W3_irrep_dict):
    
    link_dict = {'E': 'E', 
                 'R_12': 'gamma_1*gamma_2*gamma_3', 
                 'R_12*R_12': '-', 
                 'R_12*R_12*R_12': '-gamma_1*gamma_2*gamma_3', 
                 'R_12*R__23*R__23*I_S': 'gamma_0*gamma_1*gamma_2*gamma_3', 
                 'R__23*R__23*R_12*I_S': 
                 '-gamma_0*gamma_1*gamma_2*gamma_3',
                 'R___31*R___31*I_S': 'gamma_0', 
                 'R__23*R__23*I_S': '-gamma_0'}
    
    
    SW4_D4_irreps = {}
    original_D4_irreps = sg.generate_bosonic_little_group_irreps(W3.generate_D4_E_irrep(mom_key = 'mom001'), mom_key='mom001', taste_key='taste_000', W3_irrep_dict=W3_irrep_dict)
    
    for key, val in original_D4_irreps.items():
        SW4_D4_irreps[key]={}
        for key1, val1 in val.items():
            
            SW4_D4_irreps[key][link_dict[key1]] = original_D4_irreps[key][key1]
            
    return SW4_D4_irreps


def bosonic_D4_cliff_2_2_conversion_dict(D4_irrep, D4_irreps):
    # This irrep conversion ignores charge conj and parity, the we just need to find the correct correspond taste four vector
    # first
    gamma_0 = D4_irreps[D4_irrep]['gamma_0']
    gamma_123 = D4_irreps[D4_irrep]['gamma_1*gamma_2*gamma_3']
    
    return (1, gamma_0, gamma_123, 1)




############################################################################################################################################### momentum decomposition 

def eipi(val):
    if val == np.pi:
        return 'p'
    else:
        return '0'

def momentum_restriction(zero_mom_irrep_key, mom_key, W3_irrep_dict):
    # This produces taste irrep rows from full taste irreps, results are not full taste irreps
    # eg. input has pp0 = {pp0, p0p, 0pp}, out will be p0 = p0, input corresponds to irrep with full orbit
    mom_dict = tg.mom_dictionary()
    mom_vec = mom_dict[mom_key]

    taste_dict = c41.taste_dictionary()
    taste_key ='taste_'+zero_mom_irrep_key[-7:-4]
    taste_vec = taste_dict[taste_key]
    
    zero_mom_irreps_conversion_dict = {}
    irrep_label = zero_mom_irrep_key[-3:]


    taste_dict_reverse = {tuple(val) : key for key, val in taste_dict.items()}  

    taste_orbit = sg.bosonic_little_groups_and_orbits([0,0,0], taste_vec,  W3_irrep_dict)[3]
    
    irrep_list = []
    
    for taste_vec in taste_orbit:
        
        little_group_zero_mom = sg.bosonic_little_groups_and_orbits([0,0,0], taste_vec,  W3_irrep_dict)[2]
        
        taste_key = taste_dict_reverse[tuple(taste_vec)]


        zero_mom_irreps = sg.generate_bosonic_little_group_irreps(faithful_little_group_matrix_dict=little_group_zero_mom, mom_key='mom000', taste_key = taste_key, W3_irrep_dict=W3_irrep_dict)

        little_group_non_zero_mom = sg.bosonic_little_groups_and_orbits(mom_vec, taste_vec,  W3_irrep_dict)[2]

    

        non_zero_mom_irreps = sg.generate_bosonic_little_group_irreps(faithful_little_group_matrix_dict=little_group_non_zero_mom, mom_key=mom_key, taste_key = taste_key, W3_irrep_dict=W3_irrep_dict)
    

        zero_mom_lg_group_restricted_to_mom_lg_dict = {}


        for key in little_group_non_zero_mom.keys():
            zero_mom_lg_group_restricted_to_mom_lg_dict[key] = zero_mom_irreps[irrep_label][key]

        
        group_restricted_to_little_group_dict_labelled = {"label" : zero_mom_lg_group_restricted_to_mom_lg_dict}
        irreps_contained = gf.compute_irreps_contained(direct_product_dict=group_restricted_to_little_group_dict_labelled, 
                                                   master_dict = non_zero_mom_irreps)

    
        taste3_key = ('').join(list(map(eipi, taste_vec)))
        
        for irrep_key in irreps_contained:
            new_irrep_label = mom_key+"_" + 'taste_'+zero_mom_irrep_key[-9:-7] + taste3_key +"_"+irrep_key[0]
            irrep_list.append(new_irrep_label)

    return irrep_list

############################################################################################################################################### Function to build momentum irreps from decomposed zero mom irrep rows

def build_irreps_from_rows(mom_key, irrep_row_list):
    # only works for p_i <=1
    no_zeros = mom_key.count('0')
    if no_zeros == 1:
        non_conserved_taste_index = mom_key[-3:].find('0')
        conserved_indices = [a for a in range(3) if a != non_conserved_taste_index]
    elif no_zeros == 2:
        non_conserved_taste_index = mom_key[-3:].find('1')
        conserved_indices = [a for a in range(3) if a != non_conserved_taste_index]
    elif no_zeros == 0:
        non_conserved_taste_index = False
    
    included_list = irrep_row_list.copy()
    included_clean_list = irrep_row_list.copy()
    non_zero_irrep_dict = {}
    non_zero_irrep_clean_dict = {}
    for item in irrep_row_list:
        if item in included_list:
            taste3_key = item[-7:-4]
            
            if non_conserved_taste_index == False:
                indepent_tastes = taste3_key
                taste_orbit = set([''.join(p) for p in permutations(indepent_tastes)])
                
                orbit_dim = len(taste_orbit)
                non_zero_irrep_dict[item, str(taste_orbit)] = [0,]*orbit_dim
                
                for j, taste in enumerate(taste_orbit):
                    irrep_row_key= item[:-7] + taste + item[-4:]
                    for item2 in irrep_row_list:
                        if item2 in included_list:
                            if item2 == irrep_row_key:
                                included_list.remove(item2)
                                non_zero_irrep_dict[item, str(taste_orbit)][j]+=1
            else:
                
                indepent_tastes = taste3_key[:non_conserved_taste_index] +taste3_key[non_conserved_taste_index + 1:]
                taste_orbit = set([''.join(p) for p in permutations(indepent_tastes)])
                
                orbit_dim = len(taste_orbit)
                non_zero_irrep_dict[item, str(taste_orbit)] = [0,]*orbit_dim
                
                for j, taste in enumerate(taste_orbit):
                    taste3 = [i for i in taste3_key]
                    taste3[conserved_indices[0]] = taste[0]
                    taste3[conserved_indices[1]] = taste[1]
                    taste3_key = ('').join(taste3)
                    irrep_row_key= item[:-7] + taste3_key + item[-4:]
                    for item2 in irrep_row_list:
                        if item2 in included_list:
                            if item2 == irrep_row_key:
                                included_list.remove(item2)
                                non_zero_irrep_dict[item, str(taste_orbit)][j]+=1 
                    
                
        if item in included_clean_list:
            taste3_key = item[-7:-4]
            if non_conserved_taste_index == False:               
                indepent_tastes = taste3_key
                clean_item = item[:-9] + indepent_tastes + item[-4:]
                taste_orbit = set([''.join(p) for p in permutations(indepent_tastes)])

                orbit_dim = len(taste_orbit)
                non_zero_irrep_clean_dict[clean_item, str(taste_orbit)] = [0,]*orbit_dim
                
                for j, taste in enumerate(taste_orbit):
                    clean_irrep_row_key = item[:-9] + taste + item[-4:]
                    for item2 in irrep_row_list:
                        if item2 in included_clean_list:

                            clean_item2 = item2[:-9] + item2[-7:-4] + item2[-4:]
                            if clean_item2 == clean_irrep_row_key:
                                non_zero_irrep_clean_dict[clean_item, str(taste_orbit)][j]+=1
                                included_clean_list.remove(item2)
            else:
                
                indepent_tastes = taste3_key[:non_conserved_taste_index] +taste3_key[non_conserved_taste_index + 1:]
                clean_item = item[:-9] + indepent_tastes + item[-4:]
                taste_orbit = set([''.join(p) for p in permutations(indepent_tastes)])

                orbit_dim = len(taste_orbit)
                non_zero_irrep_clean_dict[clean_item, str(taste_orbit)] = [0,]*orbit_dim
                
                for j, taste in enumerate(taste_orbit):
                    taste3 = [i for i in taste3_key]
                    taste3[conserved_indices[0]] = taste[0]
                    taste3[conserved_indices[1]] = taste[1]
                    taste3_key = ('').join(taste3)
                    clean_irrep_row_key = item[:-9] + taste + item[-4:]
                    for item2 in irrep_row_list:
                        
                        if item2 in included_clean_list:
                            taste3_key2 = item2[-7:-4]
                            indepent_tastes2 = taste3_key2[:non_conserved_taste_index] +taste3_key2[non_conserved_taste_index + 1:]
                            clean_item2 = item2[:-9] + indepent_tastes2 + item2[-4:]
                            if clean_item2 == clean_irrep_row_key:
                                non_zero_irrep_clean_dict[clean_item, str(taste_orbit)][j]+=1
                                included_clean_list.remove(item2)
        
    return non_zero_irrep_dict, non_zero_irrep_clean_dict

    

def continuum_state_discretised(spin, continuum_taste, continuum_parity, charge_conjugation, mom_key_list):
    ### Only bosonic and continuum_taste = 0, 15, parity/ charge c. = +1, -1 
    
    W3_irrep_dict = W3.generate_W3_irrep_dict()
    cliff41_irrep_dict = c41.generate_clifford_4_1_complete_irrep_dict()
    zero_mom_iso_dict = rf.generate_zero_mom_irrep_comparison_dict(W3_irrep_dict=W3_irrep_dict, cliff41_irrep_dict=cliff41_irrep_dict)
    
    zero_mom_iso_dict_reversed = {}
    for key, val in zero_mom_iso_dict.items():
        zero_mom_iso_dict_reversed[val[0]] = key
    
    STGAMMADICT = ops.generate_irrep_spin_taste_dictionary()
    STGAMMADICT_reversed = {}
    for key, val in STGAMMADICT.items():
        if val not in STGAMMADICT_reversed.keys():
            STGAMMADICT_reversed[val] = [key,]
        else:
            STGAMMADICT_reversed[val].append(key)
    #print(STGAMMADICT_reversed.keys())
    
    SO4_irrep_dict = generate_O4_restrricted_SW4_character_dict()
    SW4_char_dict = rf.generate_SW4_character_dict()
    D4_irreps = generate_D4_irreps_for_SW4(W3_irrep_dict)
    
    faithful_SW4_irrep = rf.generate_SW4_irrep_one_half()
    classes = gf.conjugate_classes(faithful_SW4_irrep)
    
    su2xsu4 = (spin, continuum_taste)
    if continuum_taste == 0:
        su2lxsu2sxsu2r = [(0, spin,0)]
    elif continuum_taste == 15:
        su2lxsu2sxsu2r = [(1,spin,1), (1, spin,0), (0, spin, 1)]
        
    SU2_D4_dict = {0: ["A_0"], 1 : ["A_1", "A_2", "A_3"]}
    #This doesnt work
    SU2_SU2_O4_dict = {(0,0) : "trivial_irrep", (1,0) : "one_zero", (0,1) : "zero_one", (2,0) : "two_zero", (0,2) : "zero_two"}
    
    
    SW4_D4_irrep_list = []
    for irreps in su2lxsu2sxsu2r:
        SU2_SU2_O4_dict_restricted_irrep = irreps[:2]
        SU2_D4_restricted_irrep =  irreps[2]
        
        D4_irrep_keys = SU2_D4_dict[SU2_D4_restricted_irrep]

        
        SO4_irrep_chars_dict = {}
        if SU2_SU2_O4_dict_restricted_irrep in SU2_SU2_O4_dict.keys():
            
            SO4_irrep_key = SU2_SU2_O4_dict[SU2_SU2_O4_dict_restricted_irrep]
            SO4_irrep_chars_dict[SO4_irrep_key] = SO4_irrep_dict[SO4_irrep_key]
            
            SW4_irreps = gf.compute_irreps_contained(direct_product_dict=SO4_irrep_chars_dict, master_dict=SW4_char_dict,
                                                     class_dictionary = classes)
            
            for SW4_irrep_keys in SW4_irreps:
                for D4_irrep_label in D4_irrep_keys:
                    SW4_D4_irrep_list.append((SW4_irrep_keys[0], D4_irrep_label))
            
        else:
            SU2_SU2_O4_dict_restricted_irrep1 = (SU2_SU2_O4_dict_restricted_irrep[0],0)
            SU2_SU2_O4_dict_restricted_irrep2 = (0,SU2_SU2_O4_dict_restricted_irrep[1])
            
            SO4_irrep_key1 = SU2_SU2_O4_dict[SU2_SU2_O4_dict_restricted_irrep1]
            SO4_irrep_key2 = SU2_SU2_O4_dict[SU2_SU2_O4_dict_restricted_irrep2]
            
            SO4_DP_irrep_chars = gf.direct_product_irrep(SO4_irrep_key1, SO4_irrep_key2, master_dict=SO4_irrep_dict)
            
            SW4_irreps = gf.compute_irreps_contained(direct_product_dict=SO4_DP_irrep_chars, master_dict=SW4_char_dict,
                                                    class_dictionary = classes)
            
            for SW4_irrep_keys in SW4_irreps:
                for D4_irrep_label in D4_irrep_keys:
                    SW4_D4_irrep_list.append((SW4_irrep_keys[0], D4_irrep_label))
                    
    print('')
    print(SW4_D4_irrep_list)
    print('')
    irrep_zero_mom_dict = {}
    for SW4_D4_irrep in SW4_D4_irrep_list:
        cliff_2_2_D4_conversion = bosonic_D4_cliff_2_2_conversion_dict(SW4_D4_irrep[1], D4_irreps)
        for irrep_key, irrep_value in zero_mom_iso_dict.items():
            if SW4_D4_irrep[0] == irrep_value[0][0] and cliff_2_2_D4_conversion == irrep_value[0][1]:
                taste_key = irrep_key[-8:-4]

                if taste_key.count("0") == 1 or taste_key.count("0") == 2:
                    if charge_conjugation == "+":
                        discrete_charge_conjugation = -1
                    else:
                        discrete_charge_conjugation = 1
                        
                else:
                    if charge_conjugation == "+":
                        discrete_charge_conjugation = 1
                    else:
                        discrete_charge_conjugation = -1
                       
                gamma_0 = cliff_2_2_D4_conversion[1]
                gamma_123 = cliff_2_2_D4_conversion[2]
                if continuum_parity == '-':
                    I_S = -1*gamma_0
                else:
                    I_S = gamma_0
                C_0gamma_0I_S = discrete_charge_conjugation*gamma_0*I_S
                
                cliff_2_2_irrep = (discrete_charge_conjugation, gamma_0, gamma_123, C_0gamma_0I_S)
                
                SW4G22Irrep = (irrep_value[0][0], cliff_2_2_irrep)
                
                irrep_zero_mom_dict[SW4_D4_irrep] = (SW4G22Irrep, zero_mom_iso_dict_reversed[SW4G22Irrep])
    
    
    print('Continuum Quantum Numbers:')
    print('SU(2) Spin:' + str(spin) + '    SU(4) Taste:' + str(continuum_taste) + '    Parity:' + str(continuum_parity) + '    Charge Conjugation:' + str(charge_conjugation))
    print('')
    print('Decomposes to:')
    print('Rest Frame Group: (SW4 x Γ_2_2) / Z_2  ~ T_N (p000) ◁ Γ_4_1 ◁ W3')
    print('')
    print('Γ_2_2(C_0, gamma_0, gamma_123, C_0*gamma_0*I_S)')
    print('Γ_4_1(C_0, gamma_0, gamma_1, gamma_2, gamma_3)')
    print('')
    print('')
    
    for zero_mom_trad_label, zero_mom_label in irrep_zero_mom_dict.items():
        print('\t' + str(zero_mom_label[0][0]) + ' x ' + str(zero_mom_label[0][1]).replace('-1', 'p').replace('1', '0')  + '   ~   ' + str(zero_mom_label[1]) + '  :  ' + str(sg.cleanMomTasteRotIrrep(zero_mom_label[1])))

        print('')
    
    for k, (zero_mom_trad_label, zero_mom_label) in enumerate(irrep_zero_mom_dict.items()):
        print('******************************** Staggered Irrep ' + str(k+1) + ' ********************************')
        print('')
        print('\t' + str(zero_mom_label[0][0]) + ' x ' + str(zero_mom_label[0][1]) + '   ~   ' + str(zero_mom_label[1])+ '  :  ' + str(sg.cleanMomTasteRotIrrep(zero_mom_label[1])))
        print('')
        print('Excited by operator(s): ')
        for operators in STGAMMADICT_reversed[zero_mom_label[1]]:
            
            ops.main_v2(operators[1], operators[2], operators[0])
        print('')
        if mom_key_list != None:
            
            for mom_key in mom_key_list:
            
                print('At momentum ' +  mom_key[-3:] + ' decomposes to: ')
                print('')

                non_zero_mom_irreps = momentum_restriction(zero_mom_irrep_key=zero_mom_label[1], mom_key=mom_key, W3_irrep_dict =W3_irrep_dict)

                irrep_count_dict, irrep_count_clean_dict = build_irreps_from_rows(mom_key, non_zero_mom_irreps)
                
                for j, (key, val) in enumerate(irrep_count_dict.items()):
                    print('\t ************* Staggered Irrep ' + str((k+1,j+1)) + ' *************')
                    
                    print('\t ' + 'T_N ◁ Γ_4_1 ◁ W3 Irrep: ' + sg.cleanMomTasteRotIrrep(key[0]))
                    
                print('')
                print('Operators')

                for j, (key, val) in enumerate(irrep_count_dict.items()):
                    print('\t ************* Staggered Irrep ' + str((k+1,j+1)) + ' *************')
                    dim = len(val)
                    print('')
                    print('\t ' + 'T_N ◁ Γ_4_1 ◁ W3 Irrep: ' + sg.cleanMomTasteRotIrrep(key[0]))
                    print('\t Taste orbit dimension: ' + str(dim))
                    print('')
                    print('\t Excited by operator(s): ')
                    for operators in STGAMMADICT_reversed[key[0]]:

                        ops.main_v3(operators[1], operators[2], operators[0])
                    print('')
                
    return None


#cD.continuum_state_discretised(spin=0, continuum_taste=0, continuum_parity="-", charge_conjugation="+", mom_key_list = ["mom001", "mom110", "mom111", 'mom210'])

def relabelSW4(SW4Irrep):
        return SW4Irrep.replace('trivial_irrep', ':::tiny:::yng(4)').replace('anti_sym', ':::tiny:::yng(1,1,1,1)').replace('two_two', ':::tiny:::yng(2,2)').replace('three_one', ':::tiny:::yng(3,1)').replace('two_one_one', ':::tiny:::yng(2,1,1)').replace('one_zero_bar', ':::overline...(1,0),,,').replace('zero_one_bar', ':::overline...(0,1),,,').replace('one_zero', '(1,0)').replace('zero_one', '(0,1)').replace('one_half_bar', ':::overline...:::left(:::frac...1,,,...2,,,, :::frac...1,,,...2,,,:::right)}$').replace('one_half', ':::left(:::frac...1,,,...2,,,, :::frac...1,,, ...2,,,:::right)').replace('six', ':::mathbf...6,,,').replace('eight', ':::mathbf...8,,,')
    



def continuum_state_discretisedLatex(spin, continuum_taste, continuum_parity, charge_conjugation, mom_key_list):
    ### Only bosonic and continuum_taste = 0, 15, parity/ charge c. = +1, -1 
    
    W3_irrep_dict = W3.generate_W3_irrep_dict()
    cliff41_irrep_dict = c41.generate_clifford_4_1_complete_irrep_dict()
    zero_mom_iso_dict = rf.generate_zero_mom_irrep_comparison_dict(W3_irrep_dict=W3_irrep_dict, cliff41_irrep_dict=cliff41_irrep_dict)
    
    zero_mom_iso_dict_reversed = {}
    for key, val in zero_mom_iso_dict.items():
        zero_mom_iso_dict_reversed[val[0]] = key
    
    STGAMMADICT = ops.generate_irrep_spin_taste_dictionary()
    STGAMMADICT_reversed = {}
    for key, val in STGAMMADICT.items():
        if val not in STGAMMADICT_reversed.keys():
            STGAMMADICT_reversed[val] = [key,]
        else:
            STGAMMADICT_reversed[val].append(key)
    #print(STGAMMADICT_reversed.keys())
    
    SO4_irrep_dict = generate_O4_restrricted_SW4_character_dict()
    SW4_char_dict = rf.generate_SW4_character_dict()
    D4_irreps = generate_D4_irreps_for_SW4(W3_irrep_dict)
    
    faithful_SW4_irrep = rf.generate_SW4_irrep_one_half()
    classes = gf.conjugate_classes(faithful_SW4_irrep)
    
    su2xsu4 = (spin, continuum_taste)
    if continuum_taste == 0:
        su2lxsu2sxsu2r = [(0, spin,0)]
    elif continuum_taste == 15:
        su2lxsu2sxsu2r = [(1,spin,1), (1, spin,0), (0, spin, 1)]
        
    SU2_D4_dict = {0: ["A_0"], 1 : ["A_1", "A_2", "A_3"]}
    #This doesnt work
    SU2_SU2_O4_dict = {(0,0) : "trivial_irrep", (1,0) : "one_zero", (0,1) : "zero_one", (2,0) : "two_zero", (0,2) : "zero_two"}
    
    
    su2lxsu2sxsu2rIrrepList = []
    SW4_D4_irrep_list = []
    for irreps in su2lxsu2sxsu2r:
        SU2_SU2_O4_dict_restricted_irrep = irreps[:2]
        SU2_D4_restricted_irrep =  irreps[2]
        su2lxsu2sxsu2rIrrepList.append((SU2_SU2_O4_dict_restricted_irrep, SU2_D4_restricted_irrep))
        
        D4_irrep_keys = SU2_D4_dict[SU2_D4_restricted_irrep]

        
        SO4_irrep_chars_dict = {}
        if SU2_SU2_O4_dict_restricted_irrep in SU2_SU2_O4_dict.keys():
            
            SO4_irrep_key = SU2_SU2_O4_dict[SU2_SU2_O4_dict_restricted_irrep]
            SO4_irrep_chars_dict[SO4_irrep_key] = SO4_irrep_dict[SO4_irrep_key]
            
            SW4_irreps = gf.compute_irreps_contained(direct_product_dict=SO4_irrep_chars_dict, master_dict=SW4_char_dict,
                                                     class_dictionary = classes)
            
            for SW4_irrep_keys in SW4_irreps:
                for D4_irrep_label in D4_irrep_keys:
                    SW4_D4_irrep_list.append((SW4_irrep_keys[0], D4_irrep_label))
            
        else:
            SU2_SU2_O4_dict_restricted_irrep1 = (SU2_SU2_O4_dict_restricted_irrep[0],0)
            SU2_SU2_O4_dict_restricted_irrep2 = (0,SU2_SU2_O4_dict_restricted_irrep[1])
            
            SO4_irrep_key1 = SU2_SU2_O4_dict[SU2_SU2_O4_dict_restricted_irrep1]
            SO4_irrep_key2 = SU2_SU2_O4_dict[SU2_SU2_O4_dict_restricted_irrep2]
            
            SO4_DP_irrep_chars = gf.direct_product_irrep(SO4_irrep_key1, SO4_irrep_key2, master_dict=SO4_irrep_dict)
            
            SW4_irreps = gf.compute_irreps_contained(direct_product_dict=SO4_DP_irrep_chars, master_dict=SW4_char_dict,
                                                    class_dictionary = classes)
            
            for SW4_irrep_keys in SW4_irreps:
                for D4_irrep_label in D4_irrep_keys:
                    SW4_D4_irrep_list.append((SW4_irrep_keys[0], D4_irrep_label))
    
    
    su2lxsu2sxsu2rIrrepList = list(map(str,su2lxsu2sxsu2rIrrepList))
    su2lxsu2sxsu2rIrrepList = [irrep.replace('(', "", 1).rstrip(')') for irrep in su2lxsu2sxsu2rIrrepList]
    su2lxsu2sxsu2rIrrepList = [irrep[:6] + ' ::: :::otimes ::: ' + irrep[7:] for irrep in su2lxsu2sxsu2rIrrepList]
    
    firstStepDecomp = " ::: ::: :::oplus ::: ::: ".join(map(str,su2lxsu2sxsu2rIrrepList))
    
    print('(' + str(continuum_taste) + ', ' + str(spin) + ') :::to ::: &'  + firstStepDecomp+ "::::::")
    
    D4hRelabel = {'A_0' : '(0, 0)', 'A_1' : '(:::pi, 0)', 'A_2' : '(0, :::pi)', 'A_3' : '(:::pi, :::pi)'}
    sw4D4RelabelListDict = {}
    k=0
    for SW4_D4_irrep in SW4_D4_irrep_list:
        SW4_irrep = relabelSW4(SW4_D4_irrep[0])
        D4_irrep =  D4hRelabel[SW4_D4_irrep[1]]
        
        SW4_D4_irrep_relabel = SW4_irrep + " ::: :::otimes ::: " + D4_irrep
        if SW4_irrep not in sw4D4RelabelListDict.keys():
            sw4D4RelabelListDict[SW4_irrep] = [SW4_D4_irrep_relabel,]
        else:
            sw4D4RelabelListDict[SW4_irrep].append(SW4_D4_irrep_relabel)
        
    lines = len(sw4D4RelabelListDict)
    for ii, (key, val) in enumerate(sw4D4RelabelListDict.items()):
        if ii==0 and lines !=1:
            print(':::to ::: & ' + " ::: ::: :::oplus ::: ::: ".join(val) + "::::::")
        elif ii==0 and lines ==1:
            print(':::to ::: & ' + " ::: ::: :::oplus ::: ::: ".join(val))
        elif ii != lines-1:
            print('& ' + " ::: ::: :::oplus ::: ::: ".join(val)+ "::::::")
        else:
            print('& ' + " ::: ::: :::oplus ::: ::: ".join(val))
            

    irrep_zero_mom_dict = {}
    for SW4_D4_irrep in SW4_D4_irrep_list:
        cliff_2_2_D4_conversion = bosonic_D4_cliff_2_2_conversion_dict(SW4_D4_irrep[1], D4_irreps)
        for irrep_key, irrep_value in zero_mom_iso_dict.items():
            if SW4_D4_irrep[0] == irrep_value[0][0] and cliff_2_2_D4_conversion == irrep_value[0][1]:
                taste_key = irrep_key[-8:-4]

                if taste_key.count("0") == 1 or taste_key.count("0") == 2:
                    if charge_conjugation == "+":
                        discrete_charge_conjugation = -1
                    else:
                        discrete_charge_conjugation = 1
                        
                else:
                    if charge_conjugation == "+":
                        discrete_charge_conjugation = 1
                    else:
                        discrete_charge_conjugation = -1
                       
                gamma_0 = cliff_2_2_D4_conversion[1]
                gamma_123 = cliff_2_2_D4_conversion[2]
                if continuum_parity == '-':
                    I_S = -1*gamma_0
                else:
                    I_S = gamma_0
                C_0gamma_0I_S = discrete_charge_conjugation*gamma_0*I_S
                
                cliff_2_2_irrep = (discrete_charge_conjugation, gamma_0, gamma_123, C_0gamma_0I_S)
                
                SW4G22Irrep = (irrep_value[0][0], cliff_2_2_irrep)
                
                irrep_zero_mom_dict[SW4_D4_irrep] = (SW4G22Irrep, zero_mom_iso_dict_reversed[SW4G22Irrep])

    print('')
    print('Lattice Decomp')
    print('')
    lines = len(irrep_zero_mom_dict)
    for k, (zero_mom_trad_label, zero_mom_label) in enumerate(irrep_zero_mom_dict.items()):
        
        SW4_irrep = relabelSW4(zero_mom_trad_label[0])
        D4_irrep =  D4hRelabel[zero_mom_trad_label[1]]
        
        SW4_D4_irrep_relabel = SW4_irrep + " ::: :::otimes ::: " + D4_irrep
        
        print("&" + SW4_D4_irrep_relabel + " & :::quad :::quad  &:::sim& " + sg.cleanMomTasteRotIrrep(zero_mom_label[1]) + " :::quad :::quad :::quad &:& " + ops.convertOPKeyFromIrrep(zero_mom_label[1])+ "::::::")
        
    print('')        
    print('Mom Decomp')
    print('')
    for k, (zero_mom_trad_label, zero_mom_label) in enumerate(irrep_zero_mom_dict.items()):
        subIrrepList = []
        origIrrep = sg.cleanMomTasteRotIrrep(zero_mom_label[1])

        if mom_key_list != None:
            
            for mom_key in mom_key_list:

                non_zero_mom_irreps = momentum_restriction(zero_mom_irrep_key=zero_mom_label[1], mom_key=mom_key, W3_irrep_dict =W3_irrep_dict)

                irrep_count_dict, irrep_count_clean_dict = build_irreps_from_rows(mom_key, non_zero_mom_irreps)
                

                for j, (key, val) in enumerate(irrep_count_dict.items()):
                    decompIrrep = sg.cleanMomTasteRotIrrep(key[0])
                    decompOp = ops.convertOPKeyFromIrrep(key[0])
                    subIrrepList.append(decompIrrep + " &:  " +  decompOp )
                    
                print("&" + origIrrep + "  &:::to& \quad :::begin{cases}" + " :::::: ".join(subIrrepList) + ":::end{cases}::::::")

                    
                
    return None