from collections import defaultdict, Counter
import os
import pickle
import re
import sys
import ast
sys.path.append('..')

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils import *
from units import *
from post_processing_2 import *


#using minimum cut-off for temperature

prop_names = ['Density', 'Glass transition temperature', 'Refractive index', 'Abbe value', "Young's modulus", 'Shear modulus', 'Vickers hardness', 'Poisson ratio', 'Fracture toughness', 'Crystallization temp', 'Melting temp', 'Electric conductivity', 'Dielectric constant', 'Softening Point (Temperature)', 'Annealing Point (Temperature)', 'Thermal expansion coefficient', 'Liquidus temperature', 'Bulk modulus', 'Activation energy']


def eval_(node):
    if isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.BinOp):
        return operators[type(node.op)](eval_(node.left), eval_(node.right))
    else:
        raise TypeError(node)
        
def eval_expr(expr):
    try:
        return eval_(ast.parse(expr, mode='eval').body)
    except ZeroDivisionError:
        print(f"Warning: Division by zero encountered in expression '{expr}'.")
        return 0
    except SyntaxError:
        print(f"SyntaxError: Invalid expression '{expr}'.")
        return 0


def temp_cut_off(f_pred):    
    
#     remove_pred_list = []
    indices_to_remove = set()  # Using set for O(1) lookup
    
    for idx, tup in enumerate(f_pred):
        if any(tup):
            
            if type(tup[2]) == str:
                tup[2] = eval_expr(tup[2])

            if not isinstance(tup, tuple):
#                 print(tup)
                print()
    
            if tup[1] in ['Density', 'Glass transition temperature', 'Refractive index', "Young's modulus", 'Shear modulus', 'Vickers hardness', 'Crystallization temp', 'Melting temp', 'Electric conductivity', 'Dielectric constant', 'Softening Point (Temperature)', 'Annealing Point (Temperature)', 'Liquidus temperature', 'Bulk modulus', 'Activation energy']:
                if tup[2]<=0:
                    indices_to_remove.add(idx)
                    continue
               
            if  tup[1] in ['Glass transition temperature', 'Crystallization temp', 'Softening Point (Temperature)', 'Annealing Point (Temperature)', 'Softening Point (Viscosity)']:
                if tup[2]<50 or tup[2]>2500:
                    indices_to_remove.add(idx)
            elif  tup[1] in ['Melting temp', 'Liquidus temperature']:
                if tup[2]<50 or tup[2]>4000:
                    indices_to_remove.add(idx)
            elif  tup[1] in ["Young's modulus", 'Shear modulus', 'Bulk modulus']:
                if tup[2]<1 or tup[2] > 2*10**12: #and tup[3] == 'GPa':
                    indices_to_remove.add(idx)
            elif  tup[1] in ['Refractive index']:
                if tup[2]<=1 or tup[2]>=7:
                    indices_to_remove.add(idx)
            elif  tup[1] in ['Density']:
                if 'cm' in tup[-1] and tup[2]>25:
                    indices_to_remove.add(idx)
#                 elif 'kg' in tup[-1] and tup[2]<75:
#                     indices_to_remove.add(idx)
                elif tup[2]<=0 or tup[2]>25000:
                    indices_to_remove.add(idx)
            elif  tup[1] in ['Poisson ratio']:
                if tup[2]<=-1 or tup[2]>=0.5:
                    indices_to_remove.add(idx)
            elif  tup[1] in ['Abbe value']:
                if tup[2]<=5 or tup[2]>=1000:
                    indices_to_remove.add(idx)
            elif tup[1] in ['Activation energy']:
                temp_unit = tup[-1].lower().replace(' ', '') 
                if tup[2]>1500 and temp_unit not in ['j/mol', 'mev']:
                    indices_to_remove.add(idx)
            
                    

                
#     upd_f_pred = [x for x in f_pred if x not in remove_pred_list]
    upd_f_pred = [tup for idx, tup in enumerate(f_pred) if idx not in indices_to_remove]
    
    return upd_f_pred

                
#using undesired units to detect noisy tuples
def remove_tuples_on_units(f_pred):
    
    indices_to_remove = set()
    
    for idx, tup in enumerate(f_pred):
        if any(tup):
            if  tup[1] in ['Density'] and tup[3] in ['atoms/A-3', 'A', 'Å', 'O-O']:
                indices_to_remove.add(idx)
            elif tup[1] in ['Glass transition temperature', 'Crystallization temp', 'Melting temp', 'Softening Point (Temperature)', 'Annealing Point (Temperature)', 'Liquidus temperature','Softening Point (Viscosity)'] and tup[3] in ['GPa', 'K/m', 'J/gK', 'nm', 'MPa', 'min', 'minute', 's', 'sec', 'second', 'hr', 'hour', 'ms']:
                indices_to_remove.add(idx)
            elif tup[1] in ['Fracture toughness'] and tup[3] in ['mPa -1', 'mPa-1']:
                indices_to_remove.add(idx)
            elif tup[1] in ['Electric conductivity'] and tup[3] in ['m2 s-1', 'm2s-1']:
                indices_to_remove.add(idx)
            elif tup[1] in ['Activation energy'] and tup[3] in ['kV/cm']:
                indices_to_remove.add(idx)
            elif tup[1] in ['Vickers hardness'] and tup[3] in ['degC', 'K', 'g/cm3', 'kg/cm3']:
                indices_to_remove.add(idx)
            elif tup[1] in ["Young's modulus", 'Shear modulus', 'Bulk modulus'] and tup[3] in ['K/m', 'nm', 'min', 'minute', 's', 'sec', 'second', 'hr', 'hour', 'ms', 'degree', 'degrees', 'Degree']:
                indices_to_remove.add(idx)
            
        
    upd_f_pred = [tup for idx, tup in enumerate(f_pred) if idx not in indices_to_remove]
    
    return upd_f_pred

        
#checking whether the property predicted falls within its specified range with the help of its median        
den_v, temp1_v, temp2_v, temp3_v, ri_v, m_v, ab_v, pr_v, dc_v, tec_v, ea_v = set(),set(), set(), set(), set(), set(), set(), set(), set(), set(), set()
def check_whether_in_limit(index, vals, prop_code):
    
    global den_v, temp1_v, temp2_v, temp3_v, ri_v, m_v, ab_v, pr_v, dc_v, tec_v, ea_v, prop_names
    
    prop_dict = {i+2: prop_names[i] for i in range(len(prop_names))}
    prop_name = prop_dict[prop_code]
    
    values = [find_num(v) for v in vals if find_num(v)!=None]
    numeric_values = [float(value) for value in values]
    median = np.median(numeric_values)
    
    if prop_name == 'Density' and not (0<=median<=25000):
        den_v.add(index)
        return False
    elif prop_name in ['Glass transition temperature', 'Annealing Point (Temperature)'] and not 50<=median<=2000:
        temp1_v.add(index)
        return False
    elif prop_name in ['Softening Point (Temperature)', 'Softening Point (Viscosity)', 'Crystallization temp'] and not 50<=median<=2500:
        temp3_v.add(index)
        return False
    elif prop_name in ['Melting temp', 'Liquidus temperature'] and not 50<=median<=4000:
        temp2_v.add(index)
        return False
    elif prop_name in ['Refractive index'] and not 1<=median<=7:
        ri_v.add(index)
        return False
    elif prop_name in ['Abbe value'] and not 10<=median<=1000:
        ab_v.add(index)
        return False
    elif prop_name in ['Poisson ratio'] and not -1<=median<=0.5:
        pr_v.add(index)
        return False
    elif prop_name in ['Dielectric constant'] and not  1<=median<=4:
        dc_v.add(index)
        return False
    elif prop_name in ['Thermal expansion coefficient'] and not median<=250:
        tec_v.add(index)
        return False
    elif prop_name in ['Activation energy'] and not 0<=median<=3000:
        ea_v.add(index)
        return False
    
    return True
        

        
#analyse property based on heading
comp_v = set()
m_gpa, y_gpa, vh_gpa, t_exo, t_endo, sint_temp, tm_liq = set(), set(), set(), set(), set(), set(), set()

def check_heading(pii, t_idx, index, vals, caption, pred_label):
    
    if pred_label == 0:
        return True
    
    heading = vals[0]
    num_val = vals[1:]
    heading_non_unit = re.sub(r"\s*\(.*?\)|\s*\[.*?\]|\s*\{.*?\}|\s*<.*?>", "", heading).strip()
    
    
    #check if composition is mistaken as prop
    global comp_v, m_gpa, y_gpa, vh_gpa, t_exo, t_endo, sint_temp, tm_liq
    comp_check = True
    comp_unit_list = ['mol%', 'mol %', 'wt%', 'wt %', 'at%']
    if any([elem in heading.lower() for elem in comp_unit_list]) and pred_label>=2:
        comp_v.add(index)
#         comp_check = False
        return False

    if any(term in heading.lower() for term in ['calcination temp', 'boiling ', 'colour temp', 'fracture energy', 'compressive modulus', 'shear strength']) or heading in ['CCT (K)', 'CCT (degC)', 'CCT', 'CCT (C)', 'CCT(K)', 'CCT(degC)']:
        return False
    
    if pred_label == 13:
        if any(term in heading.lower() for term in ['thermal', 'photo', 'radio', 'hydraulic', 'mass', 'acoustic', 'spin', 'magnon', 'diffusive', 'permeability', 'phonon', 'chemical', 'magnet']):
            return False
        
        
    if pred_label == 18:
        if 'log' in heading.lower() and 'liq' not in heading.lower():
            return False
        
    if pred_label == 10:
        if any(term in heading.lower() for term in ['strength', 'strain', 'impact', 'breaking', 'ultimate', 'yield', 'flexural', 'tensile', 'compressive', 'rupture', 'resilience', 'hardness', 'elongation']):
            return False
        
    if pred_label == 20:
        if 'activation' not in heading.lower() and any(term in heading.lower() for term in ['band', 'gap', 'binding', 'ionization', 'formation', 'dissociation', 'stokes', 'potential', 'kinetic', 'fermi', 'transition', 'optical', 'electron', 'hole', 'valence', 'conduction', 'photon']):
            return False
        
    if pred_label == 8:
        if ('hardness' not in heading.lower()) and any(term in heading.lower() for term in ['strength', 'modulus', 'toughness', 'stiffness', 'ductility', 'elasticity', 'plasticity', 'fracture', 'fatigue', 'resilience']):
            return False
        
    #for RI - if n is not Aravami and not RI
    cc1 = ['refractiv', 'optical prop']
    cc2 = ['parameter','Avrami','exponent', 'ion density']
    if heading_non_unit.lower() == 'n' and pred_label == 4:
        if caption!=None:
            caption_check1 = any([element in caption for element in cc1])
            caption_check2 = any([element in caption for element in cc2])
            if caption_check2 and not caption_check1:
                return False
            
    #check if sintering temp is mistaken as annealing temp by model
    if 'sintering temp' in heading.lower() and pred_label == 16:
        sint_temp.add(index)
        return False
    
    #improve precision of TC  
#     if any(term in heading for term in ['Tc', 'T c', 'T  c', 'T C']) and pred_label == 11:
    if heading_non_unit in ['Tc', 'T c', 'T  c', 'T C', 'TC'] and pred_label == 11:
        cap_ch = 'Curie temperature'
        cap2_ch = 'Temperatures of Curie'
        if cap_ch.lower() in caption.lower() or cap2_ch.lower() in caption.lower():
            return False
        
#     if any(term in heading for term in ['Ts', 'TS', 'T s', 'Ts.']) and pred_label == 15:
    if heading_non_unit in ['Ts', 'T s', 'Ts.', 'TS'] and pred_label == 15:
        if 'substrate' in caption.lower() and 'soft' not in caption.lower():
            return False
        
        
    if heading_non_unit in ['Eg'] and pred_label == 3:
        return False
    
        
    #improve precision of TM
    if heading_non_unit.lower() in ['tm', 'tm1', 'tm2', 'tm3', 't m', 't m1', 't m2', 't m3'] and pred_label == 12:
        cap_ch = 'melting '
        cap_ch1 = 'maximum '
        cap_ch2 = 'minimum '
        
        
        if cap_ch not in caption.lower() and (cap_ch1 in caption.lower() or cap_ch2 in caption.lower()):
            if check_paper_for_prop(pii, t_idx, 'Melting temp'):
                return True
            else:
                return False
        
    #improve precision in PR
    if 'swel' in heading.lower() and pred_label == 9:
        return False
    
    if heading_non_unit in ['s', 'n'] and pred_label == 9:
        #check no conductivity in caption but Poisson in caption for s
        #check no refractive in caption but Poisson in caption for n
        #range of value should be btwn [0.1, 0.5]
        cap_ch = 'poisson'
        cap_ch1 = 'conductivity'
        cap_ch2 = 'refractive'
        cap_check = False
        
        if heading_non_unit == 's':
            if cap_ch in caption.lower() and cap_ch1 not in caption.lower():
                cap_check = True
        elif heading_non_unit == 'n':
            if cap_ch in caption.lower() and cap_ch2 not in caption.lower():
                cap_check = True
        
        all_elem_in_range = all((-1 <= float(find_num(elem)) <= 0.5) if find_num(elem) is not None else True for elem in num_val)
        if not (cap_check or all_elem_in_range): #changed to 'or' -- any one is enough -- in labelling it is 'and'
            return False
        

        
    
    
    return True


def direct_matching(element):
    ## modify it accordingly
    # element = heading
    
    
    if 'transition temperature' in element.lower():
        return 3

    elif 'refractive index' in element.lower() and 'non' not in element.lower():
        return 4

    elif 'abbe number' in element.lower() or 'abbe value' in element.lower():
        return 5

    elif 'young' in element.lower() and 'modulus' in element.lower():
        return 6

    elif 'shear modulus' in element.lower():
        return 7

    elif 'hardness' in element.lower():
        return 8

    elif 'poisson' in element.lower() and 'ratio' in element.lower():
        return 9

    elif 'temperature' in element.lower() and 'crystallization' in element.lower():
        return 11

    elif 'dc conductiv' in element.lower() or 'electrical conductiv' in element.lower():
        return 13

    elif 'dielectric constant' in element.lower():
        return 14

    elif 'annealing temperature' in element.lower() or 'annealing point' in element.lower():
        return 16

    elif 'thermal expansion coefficient' in element.lower() or 'coefficient of thermal expansion' in element.lower():
        return 17

    elif 'liquidus temp' in element.lower():
#         print(element)
        return 18

    elif 'melting temp' in element.lower():
#         print(element)
        return 12

    elif 'softening point' in element.lower() or 'softening temp' in element.lower():
        return 15
#                 
    elif 'bulk modulus' in element.lower():
        return 19

    elif 'activation energy' in element.lower():
        return 20

    else: 
        return 0
                  