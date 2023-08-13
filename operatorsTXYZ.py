import numpy as np
import sympy as sp
from numpy.linalg import matrix_power
from sympy.utilities.lambdify import lambdify
import os
import pickle
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

import gfunctions as gf
import stagGroup as sg
import translationGroup as tg
import cliff41Group as c41
import W3RotationGroup as W3


################################################################################################################################################ Staggered action phases (not sure if any convention is implied at this point but I was using it with xyzt)

def bar(gamma_vec, mod = True):
    new_gamma_vec = np.zeros((len(gamma_vec)), dtype = type(gamma_vec[0]))
    for i, val1 in enumerate(gamma_vec):
        for j, val2 in enumerate(gamma_vec):
            if j != i:
                new_gamma_vec[i]+=val2
                
    if mod == True:
        return np.mod(new_gamma_vec, 2)
    else:
        return new_gamma_vec


def less_than(gamma_vec, mod = True):
    new_gamma_vec = np.zeros((len(gamma_vec)), dtype = type(gamma_vec[0]))
    for i, val1 in enumerate(gamma_vec):
        for j, val2 in enumerate(gamma_vec):
            if j < i:
                new_gamma_vec[i]+=val2
                
    if mod == True:
        return np.mod(new_gamma_vec, 2)
    else:
        return new_gamma_vec

def greater_than(gamma_vec, mod = True):
    new_gamma_vec = np.zeros((len(gamma_vec)), dtype = type(gamma_vec[0]))
    for i, val1 in enumerate(gamma_vec):
        for j, val2 in enumerate(gamma_vec):
            if j > i:
                new_gamma_vec[i]+=val2
                
    if mod == True:
        return np.mod(new_gamma_vec, 2)
    else:
        return new_gamma_vec



def epsilon_phase(n_vec):
    exponent = np.sum(n_vec)
    
    return np.power(-1, exponent)

def eta_phase(n_vec):
    exponent =  less_than(n_vec)
    
    return np.power(-1, exponent)

def zeta_phase(n_vec):
    exponent = greater_than(n_vec)
    
    return np.power(-1, exponent)


##########################################################################################################################################
#### Useful functions

def sympy_mod(expression):
    #### For mod 2 of sympy expressions eg (-1)^(3n_1 + 2n_2 + 4_n3) = (-1)^(n_1)
    new_expression = 0
    if expression ==sp.sympify(1) or expression == sp.sympify(-1):
        new_expression += sp.sympify(1)
    
    elif expression ==sp.sympify(2):
        new_expression = 0
    
    elif expression != 0:
        if len(str(expression).split(" "))==1:
            if len(expression.args) == 2: 
                if -1 in  expression.args:
                    new_expression = -1 * expression
                else:
                    for symbol in expression.free_symbols:
                        
                        coeff = expression.coeff(symbol)
                        if coeff % 2 == 1:
                            new_expression += symbol
            else:
                new_expression = expression
        else:
            for arg in expression.args:
                if arg.is_constant():
                    new_expression += arg % 2

            for symbol in expression.free_symbols:
                coeff = expression.coeff(symbol)
                if coeff % 2 == 1:
                    new_expression += symbol
    else:
        new_expression = 0
    return new_expression

def add(gamma_vec1, gamma_vec2):
    gamma_new = np.add(gamma_vec1, gamma_vec2)
    gamma_new = np.mod(gamma_new, 2)
    return gamma_new


##########################################################################################################################################
#### Converting from gamma spin taste operators to phase and shift / link operators

gamma_dict = {
    "G1":  [[0,0,0,0],1],
    "GX":  [[0,1,0,0],1],
    "GY":  [[0,0,1,0],1],
    "GZ":  [[0,0,0,1],1],
    "GT":  [[1,0,0,0],1],
    "G5":  [[1,1,1,1],1],
    "GYZ": [[0,0,1,1],1], 
    "GZX": [[0,1,0,1],-1],
    "GXY": [[0,1,1,0],1],
    "GXT": [[1,1,0,0],-1],
    "GYT": [[1,0,1,0],-1],
    "GZT": [[1,0,0,1],-1],
    "G5X": [[1,0,1,1],1],
    "G5Y": [[1,1,0,1],-1],
    "G5Z": [[1,1,1,0],1],
    "G5T": [[0,1,1,1],-1]}


def phase_shift_operator(spin, taste):
    ### convention = [x_0, x_1, x_2, x_3]
    ### See Follana HISQ paper appendix
    
    n_vec = sp.var("n_0 n_1 n_2 n_3")
    
    ### Any gamma with G"AB", i.e two Letters/numbers I don't under the definitions and the extra phases from
    ### anti commuting in the dict are not understood fully
    
    m, sign_m = gamma_dict[spin]
    s, sign_s = gamma_dict[taste]
    
    phase_exp1 = np.dot(n_vec,add(less_than(s), greater_than(m)))
    
    phase_exp2 = np.dot(m,less_than(add(s,m)))

    var_indp_phase = (-1) ** phase_exp2 * sign_m * sign_s
    
    if var_indp_phase == 1:
        total_phase_exp = phase_exp1
    else:
        total_phase_exp = phase_exp1 + sp.sympify(1)
    
    shift = add(m, s)
    
    
    return total_phase_exp, shift



##########################################################################################################################################
#### Obtaining the quantum numbers (lattice parity, charge conjugation and the four tastes) of the phase + shift/link operators. Rotation irreps has its own section next

def lattice_parity(shift, mom_vec):
    lattice_parity_phase = np.sum(shift[1:]) % 2

    
    new_mom = np.multiply(-1, mom_vec)
    new_shift = np.multiply(-1, shift)
    
    return lattice_parity_phase, new_shift,  new_mom


def phase_shift_to_lattice_parities(phase_exp, shift, mom_key):
    ### See 514 sharpe paper, this is assuming link symmetric operators
    n_vec = sp.var("n_0 n_1 n_2 n_3")
    
    mom_dict = tg.mom_dictionary()
    taste_dict = c41.taste_dictionary()
    
    phase_exponent = sum(phase_exp.free_symbols)
    
    phase_exp_func = lambdify([n_vec], phase_exponent)
    
    phase_vec_exp = list(map(phase_exp_func, [[1,0,0,0], [0,1,0,0],[0,0,1,0],[0,0,0,1]]))
    
    phase_vec = np.power(-1, phase_vec_exp)
    
    # -1 = e^{-i pi}, 1 = e^{i 0}
    
    taste_vec = np.where(np.multiply(zeta_phase(shift) , phase_vec)==-1, "p", "0")
    

    I_S = np.power(-1,lattice_parity(shift=shift, mom_vec=mom_dict[mom_key])[0])

    
    mom = mom_dict[mom_key]
    
    mom_phase = np.real(tg.mom_character(mom, shift[1:], N=32))
    
    
    lattice_charge_conjugatation = epsilon_phase(shift) * (-1) ** phase_exp_func(shift)
    
    #print(taste_vec)
    if lattice_charge_conjugatation == 1:
        taste_vec = np.insert(taste_vec, 0, '0')
    elif lattice_charge_conjugatation == -1:
        taste_vec = np.insert(taste_vec, 0, 'p')
    taste_key = "taste_" + "".join(taste_vec)
    
    return taste_key, I_S, lattice_charge_conjugatation



##########################################################################################################################################
#### Obtaining the SW3(well W3 because I don't separate the labelling with +/-) irrep of the phase + shift/link operators.


def rot_phase_exp(n_vec, axis1, axis2):
    
    first_term = sympy_mod(less_than(n_vec, mod = False)[axis1] + less_than(n_vec, mod = False)[axis2])
    second_term = sympy_mod(greater_than(n_vec, mod = False)[axis1] + greater_than(n_vec, mod = False)[axis2])
    third_term = sympy_mod(first_term + second_term)
    return [0, first_term, second_term, third_term]

def rotate_vec(vec, axis1, axis2, inverse = False):
    vec_rot_new = np.copy(vec)
    
    if len(vec) == 3:
        axis1= axis1-1
        axis2= axis2-1
        
    if inverse == False:
        vec_rot_new[axis1] =  vec[axis2]
        vec_rot_new[axis2] = -1 * vec[axis1]
        
    else:
        vec_rot_new[axis1] = -1 * vec[axis2]
        vec_rot_new[axis2] =  vec[axis1]
                
    return vec_rot_new


def operator_proper_rotation(phase_exp, shift, mom_vec, rotation):
    ### Inverse rotation
    
    n_vec = np.array(sp.var("n_0 n_1 n_2 n_3"))
    n_vec_shift = np.add(n_vec, shift)
    
    axis1 = int(rotation[-2])
    axis2 = int(rotation[-1])
    
    n_vec_rot = rotate_vec(vec=n_vec, axis1=axis1, axis2=axis2, inverse = True)
    n_vec_shift_rot = rotate_vec(vec=n_vec_shift, axis1=axis1, axis2=axis2, inverse = True)
    
    shift_rot = rotate_vec(vec=shift, axis1=axis1, axis2=axis2, inverse = True)
    mom_vec_rot = rotate_vec(vec=mom_vec, axis1=axis1, axis2=axis2, inverse = True)
    
    rot_phase_exp1 = rot_phase_exp(n_vec=n_vec_rot, axis1=axis1, axis2=axis2)
    rot_phase_exp2 = rot_phase_exp(n_vec=n_vec_shift_rot, axis1=axis1, axis2=axis2)


    sign_list = np.ones((16))
    rot_prod_exponent_list = []
    
    
    for i, ele1 in enumerate(rot_phase_exp2):
        for j, ele2 in enumerate(rot_phase_exp1):
            rot_prod_exponent_list.append(sympy_mod(ele1+ele2))
            
            if axis1 < axis2:
            
                if i == 2:
                    sign_list[4*i+j] = -1*sign_list[4*i+j]
                if j == 2:
                    sign_list[4*i+j] = -1*sign_list[4*i+j]
                    
            else:
                if i == 1:
                    sign_list[4*i+j] = -1*sign_list[4*i+j]
                if j == 1:
                    sign_list[4*i+j] = -1*sign_list[4*i+j]

    total_phase_exp = np.add(rot_prod_exponent_list, phase_exp)
    total_phase_exp = np.array(list(map(sympy_mod, total_phase_exp)))

    rot_product_pos_phase = np.power(-1,total_phase_exp)
    rot_product_corrected_phase = np.multiply(sign_list, rot_product_pos_phase)    
    rot_prod_normalised_phase = sp.simplify(1/4*np.sum(rot_product_corrected_phase))
    
    symbol_int_list = [str(symbol)[-1] for symbol in rot_prod_normalised_phase.free_symbols]
    
    if (rotation[-2] in symbol_int_list and not rotation[-1] in symbol_int_list) or (rotation[-1] in symbol_int_list and not rotation[-2] in symbol_int_list):
        
        total_phase_exp_mod = np.add(total_phase_exp, n_vec[axis1] + n_vec[axis2])
        total_phase_exp_mod = np.array(list(map(sympy_mod, total_phase_exp_mod)))
        
        rot_product_pos_phase = np.power(-1, total_phase_exp_mod)
        rot_product_corrected_phase = np.multiply(sign_list, rot_product_pos_phase)    
        rot_prod_normalised_phase = sp.simplify(1/4*np.sum(rot_product_corrected_phase))
    
    if "-1.0*(-1)" in str(rot_prod_normalised_phase) or rot_prod_normalised_phase == sp.sympify(-1):
        final_phase_exponent = sum(rot_prod_normalised_phase.free_symbols)+sp.sympify(1)
        
    else:
        final_phase_exponent = sum(rot_prod_normalised_phase.free_symbols)
        
        
    return final_phase_exponent, shift_rot, mom_vec_rot

def operator_general_rotation(phase_exp, shift, mom_vec, rotation):
    new_shift = shift
    new_phase_exp = phase_exp
    new_mom = mom_vec
    rot_list = rotation.split("*")
    # computing inverse here so not reversing list
    for rot in rot_list:
        if rot == "I_S":
            lattice_parity_phase, new_shift, new_mom = lattice_parity(shift=new_shift, mom_vec=new_mom)
            new_phase_exp = sympy_mod(new_phase_exp + lattice_parity_phase)
            
            
        else:
            new_phase_exp, new_shift, new_mom = operator_proper_rotation(phase_exp=new_phase_exp, 
                                                                        shift=new_shift, mom_vec=new_mom,
                                                                        rotation=rot)
            
    return new_phase_exp, new_shift, new_mom



def irrep_matrix(unique_basis_set, rotation):
    irrep_dim = len(unique_basis_set)
    identity = np.identity(irrep_dim)
    irrep_mat = np.empty((irrep_dim,irrep_dim))
    
    basis_relation_dict = {}
    for i, basis in enumerate(unique_basis_set):
        basis_relation_dict[str(basis)] = identity[:,i]
    
    #print("basis_relation_dict")
    #print(basis_relation_dict)
    #print('')
    for j, basis in enumerate(unique_basis_set):
        phase = basis[0]
        shift = basis[1]
        mom = basis[2]
        minus_phase = sp.sympify(phase) + sp.sympify(1)
        
        basis_relation_dict[str([str(minus_phase), shift, mom])] = -1*identity[:,j]
        
    for k, basis in enumerate(unique_basis_set):
        
        phase = sp.sympify(basis[0])
        shift = np.array(basis[1])
        mom = basis[2]
        new_phase, new_shift, new_mom = operator_general_rotation(phase_exp=phase, shift=shift, 
                                                                  mom_vec=mom, rotation=rotation)
        
        irrep_mat[:,k] = basis_relation_dict[str([str(new_phase), np.abs(new_shift), new_mom])]
        
    return irrep_mat
        
def oneD_irrep_char(unique_basis_set, rotation):
    basis = unique_basis_set[0]

    basis_relation_dict = {}
    basis_relation_dict[str(basis)] = 1
    
    phase = sp.sympify(basis[0])
    shift = basis[1]
    mom = basis[2]
    minus_phase = phase + sp.sympify(1)
        
    basis_relation_dict[str([str(minus_phase), shift, mom])] = -1
    
    new_phase, new_shift, new_mom = operator_general_rotation(phase_exp=phase, shift=shift, 
                                                                  mom_vec=mom, rotation=rotation)
    
    irrep_char = basis_relation_dict[str([str(new_phase), np.abs(new_shift), new_mom])]
        
    return irrep_char
    
        

def identify_rotation_irrep(phase_exp, shift, mom_key, taste_key):
    ### Given an operator, we create a basis, then from the basis we determine the irrep it belongs to
    W3_irrep_dict = W3.generate_W3_irrep_dict()
    
     
    
    mom_dict = tg.mom_dictionary()
    taste_dict = c41.taste_dictionary()
    
    mom_vec = mom_dict[mom_key]
    
    spatial_taste_key = "taste_" + taste_key[-3:]
    
    little_group = sg.bosonic_little_groups_and_orbits(mom_vec = mom_vec, 
                                                  taste_vec = taste_dict[spatial_taste_key], W3_irreps=W3_irrep_dict)[2]
    
    
    #print(spatial_taste_key)
    #print(len(little_group))
    #print(little_group)
    
    ################################################################################################################
    ### This unique set ignores negative shifts and also ignores over all phases of 1 or -1 of operators, as its not
    ### important for determining basis vectors, it will be important for determining irrep matrices though!
    phase_exponent = sum(phase_exp.free_symbols)
    unique_phase_shift_set = [[str(phase_exponent),np.array(shift), np.array(mom_vec)],]
    
    displayOp([phase_exponent,shift,mom_vec])
    
    for key in little_group.keys():
        not_known = True
        
        if key != "E":
            
            
            
            new_phase, new_shift, new_mom = operator_general_rotation(phase_exp=phase_exponent, shift=shift,
                                                                  mom_vec = mom_vec, rotation=key)
            
            #print("New Operator from Rotation: " + key)
            #displayOp([new_phase, new_shift, new_mom])
            #print('')
            
            for phase_shift in unique_phase_shift_set:
                if phase_shift[0] == str(sum(sp.sympify(new_phase).free_symbols)) and np.all(phase_shift[1] == np.abs(new_shift)) and np.all(phase_shift[2] == new_mom):
                    not_known = False
                
                
            if not_known:
                unique_phase_shift_set.append([str(sum(sp.sympify(new_phase).free_symbols)), np.abs(new_shift), np.array(new_mom)])
                
                
            
    SW3_irrep_dim = len(unique_phase_shift_set)
    ############################################################################################
    #### Matching to irreps

    little_group_irreps = sg.generate_bosonic_little_group_irreps(faithful_little_group_matrix_dict=little_group, mom_key=mom_key, taste_key=spatial_taste_key, W3_irrep_dict=W3_irrep_dict)
    
    
    character_dict = gf.generate_character_table(little_group_irreps, little_group)
    
    #print(character_dict)
    
    possible_irreps = character_dict.columns.tolist()[1:]
    for index, row in character_dict[possible_irreps].iterrows():
        for key, val in row.iteritems():
            if index == "E":
            
                if val != SW3_irrep_dim:
                    possible_irreps.remove(key)
                    
    for index, row in character_dict[possible_irreps].iterrows():
        if index != "E":
            if SW3_irrep_dim == 1:
                #print(index)
                irrep_char = oneD_irrep_char(unique_basis_set=unique_phase_shift_set, rotation=index)
                #print(irrep_char)
                for key, val in row.iteritems():
                    if val != irrep_char:
                        try:
                            possible_irreps.remove(key)
                        except:
                            continue
            else:
                #print("")
                #print(index)
                irrep_mat = irrep_matrix(unique_basis_set=unique_phase_shift_set, rotation=index)
                #print(irrep_mat)
                #print(np.trace(irrep_mat))
                
                for key, val in row.iteritems():
                    if val != np.trace(irrep_mat):
                        try:
                            possible_irreps.remove(key)
                        except:
                            continue

                    
                    
    return possible_irreps, unique_phase_shift_set


def identify_complete_irrep_phase_shift(phase_exp, shift, mom_key):
    #### Full of identification of staggered group irrep from phase + shift/link operator

    taste_key, I_S, lattice_charge_conjugatation = phase_shift_to_lattice_parities(phase_exp=phase_exp, 
                                                                                   shift=shift, mom_key=mom_key)
    
    possible_irreps, unique_phase_shift_set = identify_rotation_irrep(phase_exp=phase_exp, shift=shift, mom_key=mom_key, taste_key=taste_key)
    
    return mom_key, taste_key, possible_irreps, I_S, lattice_charge_conjugatation, unique_phase_shift_set


##########################################################################################################################################
#### Continuum quantum numbers parity and charge conjugation

def continuum_parity_charge_conj(taste_key, I_S, discrete_charge_conjugation):
    taste = taste_key[-4:]
    if taste.count("0") == 1 or taste.count("0") == 2:
        charge_conjugation = discrete_charge_conjugation * -1

    else:
        charge_conjugation = discrete_charge_conjugation
    
    if type(I_S) == type(" "):
        parity = "Parity is not a quantum number"
        
    else:
        
        if taste[0] == "p":
            parity = -1 * I_S
        else:
            parity =  I_S
    

    return parity, charge_conjugation



##########################################################################################################################################
#### Input spin taste and momentum, get phase-shift/link operator & staggered irrep

def main(spin, taste, momentum):
    #### Must use Z component as I don't have the little group irreps for [0,1,0], [1,0,0]
    #### ***I do now so use 001 for momentum and any taste is fine, beware of taste orbits!
    #### It is the same group but different rotations for each element. Also note taste_4 is not definite.
    phase_exp, shift = phase_shift_operator(spin=spin, taste=taste)
    
    mom_key, taste_key, possible_irreps, I_S, C_0, ops = identify_complete_irrep_phase_shift(phase_exp=phase_exp,
                                                                                            shift=shift, 
                                                                                            mom_key=momentum)
    

    parity, charge_conj = continuum_parity_charge_conj(taste_key=taste_key, I_S=I_S, discrete_charge_conjugation=C_0)
               
                

    print("Spin: " + str(spin) + " , " + "taste: " + str(taste) + " , " + "momentum: " + momentum)
    print("")
    print("W3 Operator Basis: ")
    for operator in ops:
        print("Phase: " + operator[0] + " , " + "Shift" + str(operator[1]) +  " , " + "Momentum" + 
              str(operator[2]) + " , " + "taste_key: " + str(phase_shift_to_lattice_parities(phase_exp=sp.sympify(operator[0]), shift=operator[1], mom_key=momentum)[0]))
    print("")
    print("Lattice irrep:")
    print(mom_key + "_" +  taste_key + "_" + possible_irreps[0])
    print("Lattice parity: " + str(I_S) + "  ,  " + " Lattice charge conjugation: " + str(C_0))
    print("")
    print("Parity: " + str(parity) + " , " + "Charge conjugation: " + str(charge_conj)) 
    print('Charge conjugation only valid for flavour singlets')
    #print(possible_irreps)

def main_v2(spin, taste, momentum):
    #### Must use Z component as I don't have the little group irreps for [0,1,0], [1,0,0]
    #### ***I do now so use 001 for momentum and any taste is fine, beware of taste orbits!
    #### It is the same group but different rotations for each element. Also note taste_4 is not definite.
    phase_exp, shift = phase_shift_operator(spin=spin, taste=taste)
    
    mom_key, taste_key, possible_irreps, I_S, C_0, ops = identify_complete_irrep_phase_shift(phase_exp=phase_exp,
                                                                                            shift=shift, 
                                                                                            mom_key=momentum)
    

    parity, charge_conj = continuum_parity_charge_conj(taste_key=taste_key, I_S=I_S, discrete_charge_conjugation=C_0)
               
                
    print('\t ---------------------------------------------')
    print("\t Spin: " + str(spin) + " , " + "taste: " + str(taste) + " , " + "momentum: " + momentum)
    print("")
    print("\t W3 Operator Basis: ")
    for operator in ops:
        print("\t Phase: " + operator[0] + " , " + "Shift" + str(operator[1]))
    #print(possible_irreps)
    
def main_v3(spin, taste, momentum):
    #### Must use Z component as I don't have the little group irreps for [0,1,0], [1,0,0]
    #### ***I do now so use 001 for momentum and any taste is fine, beware of taste orbits!
    #### It is the same group but different rotations for each element. Also note taste_4 is not definite.
    phase_exp, shift = phase_shift_operator(spin=spin, taste=taste)
    
    mom_key, taste_key, possible_irreps, I_S, C_0, ops = identify_complete_irrep_phase_shift(phase_exp=phase_exp,
                                                                                            shift=shift, 
                                                                                            mom_key=momentum)
    

    parity, charge_conj = continuum_parity_charge_conj(taste_key=taste_key, I_S=I_S, discrete_charge_conjugation=C_0)
               
                
    print('\t \t ---------------------------------------------')
    print("\t \t Spin: " + str(spin) + " , " + "taste: " + str(taste) + " , " + "momentum: " + momentum)
    print("")
    print("\t \t W3 Operator Basis: ")
    for operator in ops:
        print("\t \t Phase: " + operator[0] + " , " + "Shift" + str(operator[1]))
    #print(possible_irreps)

##########################################################################################################################################
#### Functions to create a dictionary, identify spin taste with irreps

def spin_taste_irrep_isomorphism(spin, taste, momentum):
    ####### Labelling taste irreps by consisten orbit representatives
    phase_exp, shift = phase_shift_operator(spin=spin, taste=taste)
    
    mom_key, taste_key, possible_irreps, I_S, C_0, ops = identify_complete_irrep_phase_shift(phase_exp=phase_exp,
                                                                                            shift=shift, 
                                                                                            mom_key=momentum)
    

    return lattice_irrep_taste_key, possible_irreps, C_0, shift

def generate_irrep_spin_taste_dictionary():
    
    try:
        irrep_spin_taste_dict = load_dict()
        return irrep_spin_taste_dict
    except:
        pass
        
    
    mom_dict = tg.mom_dictionary()   
    
    irrep_spin_taste_dict = {'a' : 'b'}
    k=0
    
    for momentum in mom_dict.keys():    
        for spin in gamma_dict.keys():
            for taste in gamma_dict.keys():
                
                print(momentum, spin, taste)
                
                
                phase_exp, shift = phase_shift_operator(spin=spin, taste=taste)
    
                mom_key, taste_key, possible_irreps, I_S, C_0, ops = identify_complete_irrep_phase_shift(phase_exp=phase_exp,
                                                                                            shift=shift, 
                                                                                            mom_key=momentum)
                
                
                

                try:
                    w3_irrep_label = str(possible_irreps[0])
                except:
                    w3_irrep_label = "no_irr"
                    
                full_irrep_label = mom_key + "_" +  taste_key + "_" + w3_irrep_label
                print(full_irrep_label)
                print('')
                irrep_spin_taste_dict[(momentum, spin, taste)] = full_irrep_label

    
    save_dict(irrep_spin_taste_dict)
    
    return irrep_spin_taste_dict

def save_dict(STDict):
    
    directory = './staggered_irreps/keys/'
    filename = 'ST_IRREP_DICT'
 
        
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    full_path = directory+ filename
    f = open(full_path,"wb")
    pickle.dump(STDict,f)
    f.close()
    
    return None


def load_dict():
    
    directory = './staggered_irreps/keys/'
    filename = 'ST_IRREP_DICT'
    
    full_path = directory+ filename
    
    with open(full_path, 'rb') as pfile:
    
        f = pickle.load(pfile)
        
    
    return f

####
#XYZT

def gamma_dict_xyzt():
    gamma_dict = {
    "G1":  [[0,0,0,0],1],
    "GX":  [[1,0,0,0],1],
    "GY":  [[0,1,0,0],1],
    "GZ":  [[0,0,1,0],1],
    "GT":  [[0,0,0,1],1],
    "G5":  [[1,1,1,1],1],
    "GYZ": [[0,1,1,0],1], 
    "GZX": [[1,0,1,0],-1],
    "GXY": [[1,1,0,0],1],
    "GXT": [[1,0,0,1],1],
    "GYT": [[0,1,0,1],1],
    "GZT": [[0,0,1,1],1],
    "G5X": [[0,1,1,1],-1],
    "G5Y": [[1,0,1,1],1],
    "G5Z": [[1,1,0,1],-1],
    "G5T": [[1,1,1,0],1],
    "GT5": [[1,1,1,0],-1]}
    return gamma_dict


############################################################################################################################################### ST display funcs


def displayOp(product_op):
    
    phase1, shift1, mom1 = product_op[0], product_op[1], product_op[2]
    
    fig, ax = plt.subplots(figsize=(12,1))
    fig.patch.set_visible(False)
    ax.axis('off')
    
    PSTemplate = r'$\sum_n (-1)^{{{}}} e^{{i n \cdot {}}} \bar \chi(n) \chi(n + {})$'.format(phase1, mom1, shift1)
    
    
    ax.text(0.0, 0.0,'$%s$'%PSTemplate, fontsize=16)
    plt.show()
    
    return PSTemplate


def convertOPKeyFromIrrep(IrrepKey):
    STGAMMADICT = generate_irrep_spin_taste_dictionary()
    STGAMMADICT_reversed = {}
    for key, val in STGAMMADICT.items():
        if val not in STGAMMADICT_reversed.keys():
            STGAMMADICT_reversed[val] = [key,]
        else:
            STGAMMADICT_reversed[val].append(key)

    opKey = STGAMMADICT_reversed[IrrepKey][0]
      
    return convertOPKey(opKey)

#for latex
def convertOPKey(opKey):
    
    gammaConvertDict = {'T' : "0", 'X' : "1", 'Y' : "2", 'Z' : "3", '5' : "5"}
    
    momKey = opKey[0]
    spinKey = opKey[1]
    tasteKey = opKey[2]
    
    momTuple = str((momKey[-3], momKey[-2], momKey[-1])).replace("'", "")
    
    if spinKey == 'G1':
        gammaSpinKey = "1"
    elif len(spinKey[1:]) == 1:
        gammaSpinKey = ":::gamma_" + gammaConvertDict[spinKey[-1]]
    else:
        gammaSpinKey = ":::gamma_" + gammaConvertDict[spinKey[-2]] +  ":::gamma_" + gammaConvertDict[spinKey[-1]]
        
    
    if tasteKey == 'G1':
        gammaTasteKey = "1"
    elif len(tasteKey[1:]) == 1:
        gammaTasteKey = ":::gamma_" + gammaConvertDict[tasteKey[-1]]
    else:
        gammaTasteKey = ":::gamma_" + gammaConvertDict[tasteKey[-2]] +  ":::gamma_" + gammaConvertDict[tasteKey[-1]]
        
    return ":::mathcal{O}^{ " + gammaSpinKey + " :::, :::otimes :::, " +  gammaTasteKey  + "}:::," + momTuple