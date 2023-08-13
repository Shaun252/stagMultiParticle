import numpy as np
import sympy as sp
from sympy.combinatorics import Permutation
import itertools as it
import weakref
import random
from collections import deque
from itertools import product
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


#####################################################################################################################################
#### Classes for qcd field theory obects

class dirac_field:
    lorentz_index_counter = 0
    
    def __init__(self, flavour, position_var, time_var):
        dirac_field.lorentz_index_counter += 1
        
        self.lorentz_index = str(dirac_field.lorentz_index_counter)
        self.flavour = flavour
        self.position = position_var
        self.time = time_var
        
         
class psi(dirac_field):
    def __init__(self, flavour, position_var, time_var):
        dirac_field.__init__(self, flavour, position_var, time_var)
        
class psi_bar(dirac_field):
    def __init__(self, flavour, position_var, time_var):
        dirac_field.__init__(self, flavour, position_var, time_var)
        
class gamma:
    instances = []
    def __init__(self, gamma_label, lorentz_index_left, lorentz_index_right):
        self.label = gamma_label
        self.lorentz_index_left = lorentz_index_left
        self.lorentz_index_right = lorentz_index_right
        self.__class__.instances.append(weakref.proxy(self))
        
    def display_form(self):
        gamma_string = self.label
        
        return gamma_string
    
    def display_form_mom(self):
        gamma_string = self.label
        
        return gamma_string
        
        

class operator:
    def __init__(self, operator_specification):
        
        self.operator_string = operator_specification[0]
        self.spacetime_coord_index = operator_specification[1]
        
        self.position = "x_"+str(self.spacetime_coord_index)
        self.time = "t_"+str(self.spacetime_coord_index)
        
        struct =  self.operator_string.split("*")

        self.op_psi_bar = psi_bar(struct[0][0], position_var=self.position, time_var=self.time)
               
        self.op_psi = psi(struct[2], position_var=self.position, time_var=self.time)
                    
        self.op_gamma =  gamma(gamma_label=struct[1], 
                                lorentz_index_left=self.op_psi_bar.lorentz_index, 
                                lorentz_index_right=self.op_psi.lorentz_index)
        
        
class multi_operator:
    
    def __init__(self, multi_operator_specification):
        
        operator_dict = {"pi+": "d_bar*gamma_5*u",
                     "pi-": "u_bar*gamma_5*d",
                      "pi0": "u_bar*gamma_5*u",
                     "rho0_u": "u_bar*gamma_i*u",
                     "rho0X_u": "u_bar*gamma_x*u",
                     "rho0_d": "d_bar*gamma_i*d",
                     "rho0X_d": "d_bar*gamma_x*d",    
                     "B+": "b_bar*gamma_5*u",
                     "W+": "u_bar*gamma_i*b"
                    }
        operands = ["+", "-"]
        self.multi_operator = multi_operator_specification
        self.operator_list = []
        for item in self.multi_operator:
            if item not in operands:
                operator_specification = [operator_dict[item[0]],item[1]]
                op = operator(operator_specification)
                self.operator_list.append(op)
                
            else:
                self.operator_list.append(item)

class propagator:
    def __init__(self, flavour, lorentz_index_left, lorentz_index_right, position_left, position_right, time_left, time_right):
        self.flavour = flavour
        self.lorentz_index_left = lorentz_index_left
        self.lorentz_index_right = lorentz_index_right
        self.position_left =  position_left
        self.position_right = position_right
        self.time_left = time_left
        self.time_right = time_right
        
    def display_form(self):
        propagator_string = "D_{}({},{} | {},{})".format(self.flavour, self.position_left, self.time_left, self.position_right, self.time_right)
            
        return propagator_string
    
    def display_form_mom(self):
        propagator_string = "D_{}({})".format(self.flavour, self.position_left[2:], self.time_left, self.position_right, self.time_right)
        
        return propagator_string

    
#####################################################################################################################################
#### Functions to generate the combinatorics for wick contractions


def generate_possible_contractions(n_point):
    ### Generate all possible wick contractions for an n-point function not taking into account flavour
    new_list = []
    nrange = range(n_point)

    for item in list(it.combinations(list(it.product(nrange,nrange)), n_point)):
        left_list, right_list = zip(*item)
        good_contraction = all([left_list.count(k) == 1 and right_list.count(k) == 1 for k in nrange])

        if good_contraction:
            new_list.append(item)
            
    return new_list


def wick_contraction(corr_func):
    ### Generate all wick contractions based off "generate_possible_contractions(n_point)" i.e 
    ### including flavour consideration
    n_point = len(corr_func)
    
    possible_contractions = generate_possible_contractions(n_point)
    
    valid_contraction_list = []
    for possible_contraction in possible_contractions:

        valid_contraction = all([corr_func[j].op_psi_bar.flavour == corr_func[i].op_psi.flavour for (i,j) in possible_contraction])
        
        if valid_contraction:
            valid_contraction_list.append(possible_contraction)
            
    return valid_contraction_list
    
def contraction_permutation_factor(contraction):
    ### Compute the sign assoaciated with a wick contraction, note this assume corr function in form 
    ### <psi_1 \bar psi_1 ... psi_n \bar psi_n > i.e you have already picked up an overall (-1)^n from swapping
    ### psi with psi bar
    
    psi_index, psi_bar_index = zip(*contraction)
    
    psi_index_iso = [2*i for i in psi_index]
    psi_bar_index_iso = [2*i+1 for i in psi_bar_index]
    
    contraction_iso = zip(psi_index_iso,  psi_bar_index_iso)
    
    perm_list = list(it.chain.from_iterable(contraction_iso))
    wick_permutation = Permutation(perm_list)
    if wick_permutation.is_even:
        return 1
    else:
        return -1
    
    
#####################################################################################################################################
#### Function which forms the trace correctly, checking for disconnected parts
    
def is_subset(subset, full_set):
    ### "in" doesnt work with lists of objects
    for elements in subset:
        truth_arr = []
        for full_elements in full_set:
            truth_arr.append(elements == full_elements)
            
        if not any(truth_arr):
            return False
        
    return True

def check_disconnected(struct, final_list):
    #### Examine a list of propagators and gammas in object form and check if it is a closed loop in terms of indices
    left_truth = False
    right_truth = False
    for objects in final_list:
        if struct.lorentz_index_left == objects.lorentz_index_right:
            left_truth = True
        if struct.lorentz_index_right == objects.lorentz_index_left:
            right_truth = True
    
    disconnected_truth = left_truth and right_truth
    return disconnected_truth
    
    
def sublist(parent_list, remove_list):
    new_list = []
    
    for elements in parent_list:
        remove_element = False
        for elements2 in remove_list:
            if elements == elements2:
                remove_element = True
                
        if not remove_element:
            new_list.append(elements)
            
    return new_list

def find_disconnected_propagator(gamma_instance, prop_list):
    ### Only care about loops with prop and one gamma here for some reason....
    for prop in prop_list:
        if gamma_instance.lorentz_index_left == prop.lorentz_index_right and gamma_instance.lorentz_index_right == prop.lorentz_index_left:
            return prop
        else:
            return None
    
def find_gamma(prop_instance, prop_list, loop_list, check_list, final_list):
    ### Bounce between this function and find prop to build the final contraction based on spinor indices
    if prop_instance.lorentz_index_right not in list(map(lambda obj: obj.lorentz_index_left, check_list)):
        for gamma_inst in gamma.instances:
            if prop_instance.lorentz_index_right == gamma_inst.lorentz_index_left:
                loop_list.append(gamma_inst)
                check_list.append(gamma_inst)
                find_prop(gamma_inst, prop_list, loop_list, check_list, final_list)
    else:
        return None

def find_prop(gamma_instance, prop_list, loop_list, check_list, final_list):
    
    if check_disconnected(gamma_instance, loop_list) and  not is_subset(subset=prop_list, full_set=check_list):
        final_list.append(loop_list)
        loop_list = []
        disconnected_prop = find_disconnected_propagator(gamma_instance=gamma_instance, prop_list=prop_list)
        unfound_props = sublist(parent_list=prop_list, remove_list=[disconnected_prop]+check_list)
        new_prop = random.choice(unfound_props)
        loop_list.append(new_prop)
        check_list.append(new_prop)
        find_gamma(new_prop, prop_list, loop_list, check_list, final_list)
    
    elif gamma_instance.lorentz_index_right not in list(map(lambda obj: obj.lorentz_index_left, check_list)):
        for prop_inst in prop_list:
            if gamma_instance.lorentz_index_right == prop_inst.lorentz_index_left:
                loop_list.append(prop_inst)
                check_list.append(prop_inst)
                find_gamma(prop_inst, prop_list, loop_list, check_list, final_list)
    else:
        final_list.append(loop_list)
        return None


    
#####################################################################################################################################
#### Main function to generate the wick contractions from a correlation function

def correlation_function(corr_specification, mom, norm, disconnected=True, isospin=False):
    multi_operator_dict = {"A_pi+pi-" : [["pi+", "t"],["pi-","t+1"],"-",["pi-","t"],["pi+", "t+1"]],
                           "Adag_pi+pi-" : [["pi+", "1"],["pi-","0"],"-",["pi-","1"],["pi+", "0"]],
                           
                           "A_pi+pi-pi0" : [["pi+", "t"],["pi-","t"],["pi0","t"],"-",["pi-","t"],["pi+", "t"],["pi0", "t"]],
                           "Adag_pi+pi-pi0" : [["pi+", "0"],["pi-","0"],["pi0","0"],"-",["pi-","0"],["pi+", "0"],["pi0", "0"]],
                           
                           
                           "rho0" : [["rho0_u","t"],"-",["rho0_d","t"]],
                           "rho0_dag" : [["rho0_u","0"],"-",["rho0_d","0"]],
                           "pi+": [["pi+", "0",]],
                           "pi-": [["pi-", "1",]],
                           "pi0": [["pi0", "0"]],
                           
        
    }
    
    total_norm = np.prod(norm)

    ####################################################################################################
    ### This whole section is just parsing the operators into classes
    operands = {"+" : 1, "-" : -1}
    corr_op_list = []
    for item in corr_specification:
        multi_operator_spefication = multi_operator_dict[item]
        multi_op = multi_operator(multi_operator_spefication)
        sign = 1
        sub_op_list = []
        op_list = []
        for i, struct in enumerate(multi_op.operator_list):
            if struct not in operands.keys():
                sub_op_list.append(struct)
            else:
                op_list.append([sub_op_list,sign])
                sub_op_list = []
                sign = operands[struct]
                
            if i==len(multi_op.operator_list)-1:
                op_list.append([sub_op_list,sign])
        corr_op_list.append(op_list)
                
                
    ####################################################################################################
    ### This whole section is just doing  multiplication with operators / objects
    corr_funcs = []
    for corr_func in list(it.product(*corr_op_list)):
        overall_sign = 1
        corr_func_refactor = []
        for ops in corr_func:
            overall_sign *= ops[-1]
            for sub_ops in ops[:-1]:
                for sub_sub_ops in sub_ops:
                    corr_func_refactor.append(sub_sub_ops)
                
        
        corr_funcs.append([corr_func_refactor, overall_sign])
    
    #################################################################################################
    ### Wick contration
    
    final_complete_wick_contraction_list = list()
    for correlators in corr_funcs:
        
        ops = correlators[0]
        contractions = wick_contraction(ops)
        
        
        n_point = len(ops)
            
        temp_list = [str(["*".join(["bar " + operators.op_psi_bar.flavour, operators.op_gamma.label, operators.op_psi.flavour]) +"("+ operators.op_psi.position + "," + operators.op_psi.time+")"]) for operators in ops]
            
        if correlators[1] == 1:
            sign = ""
        else:
            sign = "-"
            
        print("*******************************************************************")
        print("")
        display_temp_list(temp_list, sign, mom, total_norm)
        operator_psi_exchange_phase = np.power(-1, n_point)
        for contraction in contractions:
            prop_display_list = []
            overall_phase = contraction_permutation_factor(contraction) * operator_psi_exchange_phase * correlators[1]
            if overall_phase == 1:
                sign = ""
            else:
                sign = "-"
            
            prop_list = []
            for (i,j) in contraction:
                psi = ops[i].op_psi
                psi_bar = ops[j].op_psi_bar
                prop = propagator(psi.flavour, psi.lorentz_index, psi_bar.lorentz_index, psi.position, psi_bar.position, 
                                  psi.time, psi_bar.time)
                
                prop_list.append(prop)
        
            final_list = []
            loop_list = [prop_list[0]]
            check_list = [prop_list[0]]
            
            find_gamma(prop_list[0],prop_list, loop_list, check_list, final_list)
            
            
            if disconnected == False:
                if len(final_list) != 1:
                    continue

            display_list = ["*".join(list(map(lambda obj: obj.display_form(), sub_list))) for sub_list in final_list]
            
            #print(display_list)
            
            final_complete_wick_contraction_list.append((display_list, sign))
            #print(display_list)
            print(sign + str(total_norm)+ wick_string_display(display_list))
            
            
    #print(final_complete_wick_contraction_list)
                
                
            
    if isospin == True:
        print("Isospin Limit:")
        print("")
        isospin_limit(final_complete_wick_contraction_list, total_norm)
            
            
    ###Clean up weak gamma refences
    gamma.instances = []
    
    return None
        

def display_temp_list(temp_list, sign, mom, total_norm):
    
    full_corr = ''
    fig, ax = plt.subplots(figsize=(0.3,0.3), dpi=100)
    fig.patch.set_visible(False)
    ax.axis('off')
    
    for i, op in enumerate(temp_list):
        
        op_split = op.split('*')
        bar_q = op_split[0][-1]
        q = op_split[2][0]
        gamma = op_split[1]
        coords = op_split[2][2:-3]
        momstr = mom[i]
    
        full_corr+= r'\sum_{{\vec {}}}\, e^{{i {} \cdot \vec {}}}\, \bar {}({})\, \{} \, {}({})'.format(coords[:3], coords[:3], momstr, bar_q, coords, gamma, q, coords)
    
    full_corr = '$'+str(sign) + str(total_norm)+full_corr+'$'
    
    ax.text(0.0, 0.99,'$%s$'%full_corr, fontsize=16)
    plt.show()    

    return None

def wick_string_display(wick_string):
    
    display_string = 'tr[' + '] x tr['.join(wick_string) + ']'
    
    return display_string
    
    
 
            
def isospin_limit(wick_string_list, total_norm):
    
    unique_dict = dict()
    unique_count = dict()
    
    for wick_string in wick_string_list:
        if wick_string[1] == "-":
            add = -1
        else:
            add = 1
            
        loop_list = list(map(lambda obj: obj.replace("u", "l").replace("d", "l"), wick_string[0]))
        actual_label = wick_string_display(loop_list)
        known = False    
        for known_wick_contractions in unique_dict.keys():
            
            if actual_label in unique_dict[known_wick_contractions]:
                unique_count[known_wick_contractions] += add
                known = True
                
        if known == False:

            all_permutations_list = list()

            unique_dict[actual_label] = set()
            unique_dict[actual_label].add(actual_label)
            unique_count[actual_label] = add
            
            no_sup_perms = len(loop_list)
            
            sup_perm_list = deque(loop_list)
            
            for i in range(no_sup_perms):
                all_permutations_list = list()
                
                for quark_loop in sup_perm_list:

                    loop_perm_list = list()
                    quark_loop_list = quark_loop.split("*")
                    perm_list  = deque(quark_loop_list)
                    for cyclic_perm_no in range(int(len(perm_list) /2)):
                        perm_list.rotate(2)
                        loop_perm_list.append("*".join(list(perm_list)))

                    all_permutations_list.append(loop_perm_list)
                    
                

                for permutation in product(*all_permutations_list):
                    permutated_wick_string = wick_string_display(permutation)
                    unique_dict[actual_label].add(permutated_wick_string)  

                
                sup_perm_list.rotate(1)
    
    for wick_contraction in unique_dict.keys():
        if unique_count[wick_contraction] != 0:
            print(str(unique_count[wick_contraction]*total_norm) + "*" + wick_contraction)
              
              
    return None
              
              
              
              
              
              
              
              
              