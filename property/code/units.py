import os
import re
import numpy as np
import pandas as pd
import sys
from units_2 import *
from utils import *


def extra_unit_check(unit):
    
    if len(unit)==0 or not unit[0].isalpha() or len(unit)>10 : return False
    
    out_words = ['nominal', 'ordinal', 'experimental']
    if unit.lower() in out_words:
        return False
    
    un_list = [int(s) for s in re.findall(r'\d+', unit)]
    if len(un_list)>0:
        for uni in un_list:
            #print(uni)
            if uni>10:
                return False
    return True

def set_units(spt_c, prop, pii, t_idx):
    unit = ''
    ind_c = 0
    ind_c_val = 0
    num_count = 0
    c = len(spt_c)
    prop_names = ['Density', 'Glass transition temperature', 'Refractive index', 'Abbe value', "Young's modulus", 'Shear modulus', 'Vickers hardness', 'Poisson ratio', 'Fracture toughness', 'Crystallization temp', 'Melting temp', 'Electric conductivity', 'Dielectric constant', 'Softening Point (Temperature)', 'Annealing Point (Temperature)', 'Thermal expansion coefficient', 'Liquidus temperature', 'Bulk modulus', 'Activation energy']

    prop_dict = {i+2: prop_names[i] for i in range(len(prop_names))}
    prop_name = prop_dict[prop]
    
    for indd in range(0, c):
        ex_strr = spt_c[indd]
        if ex_strr.replace('.', '').isnumeric():
            ind_c = indd
            #indc_val = float(spt_c[ind_c])
            break
            
    #assert ind_c!=0, 'No numeric value found in col'
    if ind_c == 0 :
        ind_c = min(int(c/2), 3)
    
    for indd in range(0, ind_c):
        ex_strr = spt_c[indd]
        if re.search('[\(|\[]\s?\+\-\s?[0-9]+\s?\.?[0-9]*\s?[a-zA-Z]+\s?[\)|\]]', ex_strr) is not None:
            ex_strr = re.sub('\s?\+\-\s?[0-9]+\s?\.?[0-9]*\s?', '', ex_strr)
        ex_str = re.search('[\(|\[]\s?[a-zA-Z|\-|\s|0-9|\\|\/\|\°]+[\)|\]]', ex_strr)
        
        if ex_str is not None:
            unit = ex_str.group()[1:-1].strip()
            unit = re.sub('\s+', ' ', unit)
            #unit = re.sub('\+\-\s?[0-9]+\s?', '', unit)
                        
            ##Only property specific constraint in the program
            if unit is not None and unit != '':
                if prop_name in ['Glass transition temperature', 'Crystallization temp', 'Melting temp', 'Softening Point (Temperature)', 'Annealing Point (Temperature)', 'Liquidus temperature','Softening Point (Viscosity)']:
                    if unit[len(unit)-1].upper() == 'C':
                        unit = 'degC'
                    elif unit[len(unit)-1].upper() == 'K':
                        unit = 'K'
                 
                                
                if prop_name in ['Thermal expansion coefficient'] and 'C-1' in unit.upper():
                    unit = 'degC-1'
                elif  prop_name in ['Thermal expansion coefficient'] and 'K-1' in unit.upper():
                    unit = 'K-1'
                        
            if extra_unit_check(unit):
                break
            else:
                unit = ''
                
    if unit=='':
        if prop_name in ['Refractive index', 'Abbe value', 'Poisson ratio','Dielectric constant']:
            unit = None
           
    
    #finding mean of the values present for the property
    c_val = -50 #impossible default value
    for indd in range(ind_c, c):
        ex_strr = spt_c[indd]
        if '+-' in ex_strr:
            ex_strr = re.split('\\+\\-', ex_strr)[0].strip()
        if ex_strr.replace('.', '').isnumeric():
            #print(f'ex_strr = {ex_strr}  {type(ex_strr)}')
            try:
                ind_c_val += float(ex_strr)
                num_count += 1
            except ValueError:
                if ex_strr[-1] == '.':
#                     ex_strr = ex_strr[:-1]
                    ex_strr = ex_strr.rstrip('.')
                    ind_c_val += float(ex_strr)
                    num_count += 1

            #ind_c_val += ex_strr
                
            
    if(num_count == 0):
        num_count = 1
        #print(f'num count 0 in {pii}__{t_idx}')
    c_val = ind_c_val / num_count

            
    if unit=='' and prop_name in ['Shear modulus', 'Bulk modulus']:
        if c_val>=10 and c_val<=150: unit = 'GPa'
        elif c_val>=10000 and c_val<=150000: unit = 'MPa'

        
    nor_unit = norm_unit(unit, prop_name)
    
    if prop_name not in ['Refractive index', 'Abbe value', 'Poisson ratio', 'Dielectric constant']:
    
#         if invalid_unit(nor_unit):
        
        if nor_unit not in ['g/cm3', 'mg/m3', 'kg/m3', 'g/cm', 'lb/in3', 'degC', 'K', 'GPa', 'MPa', 'Pa', 'psi', 'dyn/cm2', 'kb', 'Kg/mm2', 'GPa', 'MPa', 'Pa', 'psi', 'MPam1/2', 'O-1 cm-1', 'O-1 m-1', '(MO m)-1', 'degC-1', 'K-1', 'kJ/mol', 'eV/at', 'kcal/mol', 'J/mol', 'eV']:
            
            # Extract unit from the first part
            extracted_unit = find_unit_further(nor_unit, prop_name, spt_c[0])

            if extracted_unit:
                nor_unit = normalize_unit(extracted_unit, prop_name)

            # Try to extract from the caption if the first attempt failed
            if not extracted_unit:
                caption = comp_data_dict.get((pii, t_idx), {}).get('caption', '')
                if caption:  # Ensure caption is not an empty string
                    extracted_unit = find_unit_further(nor_unit, prop_name, caption)
                    if extracted_unit:  # Only normalize if extraction was successful
                        nor_unit = normalize_unit(extracted_unit, prop_name)
                
    
    return nor_unit


    
def norm_unit(unit, prop_name):
    
    #value to be normalized to:
    
    den_unit = {'g/cm3': {'gm/cm3', 'g cm-3', 'g/cm-3', 'g/cm3', 'g/cm 3', 'gcm -3', 'gcm-3', 'g/cc', 'gm/cc', 'gw/cm3', 'gm cc-1', 'g/ cm3', 'g/mL', 'd gcm-3'}, 'mg/m3' : {'Mg/m3', 'mgm-3'}, 'kg/m3' : {'kgm -3', 'kgm-3', 'kg/m3', 'kg m-3', 'g/l'}, 'g/cm' : {'g/cm -1', 'gcm -1', 'gcm-1', 'g cm-1', 'g/cm'}, 'lb/in3': {'lb/in3', 'lb/ in3', 'lb in-3', 'lbin -3'}}

    tg_unit = {'degC' : {'degC', 'degC/min', 'C'}, 'K': {'K', 'K min-1', 'T/K'}}

    ri_unit = {None : {None}}

    ym_unit = {'GPa': {'GPa', 'G P a'}, 'MPa':{'Mpa', 'M P a'}, 'Pa':{'Pa', 'P a'}, 'psi':{'psi'}, 'dyn/cm2': {'dyn/cm2'}, 'kb' : {'kb'}}


    vh_unit = {'kg/mm2': {'Kg/mm2', 'kg/mm2', 'kgmm -2', 'kgmm-2', 'kg mm-2'}, 'kgf/mm2': {'Kgf/mm 2', 'kgfmm-2', 'kgf mm-2', 'Kgf/mm2', 'kgf/mm2', 'kgf/mm 2'},  'GPa': {'GPa'}, 'MPa':{'Mpa'}, 'Pa':{'Pa'}, 'psi':{'psi'}}

    ft_unit = {'MPam1/2': {'MPa m', 'MPa m1/2', 'MPam1/2', 'MPa/sqrt(m)', 'MPa(m)1/2', 'MPa·m1/2', 'm'}}

    ec_unit = {'O-1 cm-1': {'O-1 cm- 1', 'O-1 cm-1', 'ohm-1 cm-1', 'Scm-1', 'S cm-1', 'S/cm', '(O-cm)-1', 'S cm', 'Scm', 'o-1 cm-1', 'O-1cm-1'}, 'O-1 m-1': {'O-1 m- 1', 'O-1 m-1', 'ohm-1 m-1', '(Om)-1', '(O m)-1', 'Sm-1', 'S m-1', 'S/m', ' S m', 'Sm', 'sm-1', 's m-1', 's/m'}, '(MO m)-1': {'(MO m)-1', '(Mohm m)-1', '(mohm m)-1', '(MOHM M)-1'}}

    exc_unit = {'degC-1': {'degC-1' ,'degC', 'C-1', 'x10-7/degC', '/degC', 'C -1', '/C', '/ C'}, 'K-1':{'K-1', 'x10 K-1', '/K', '/ K', 'K -1'}}

    ea_unit = {'kJ/mol': {'kJmol-1', 'kJ mol-1','kJ/mol', 'kJ / mol', 'kJ/molK', 'kJmol -1'}, 'eV/at':{'eV/at'}, 'kcal/mol':{'kcal/mol', 'kcalmol-1', 'kcal mol-1'}, 'J/mol': {'Jmol-1', 'J mol-1','J/mol', 'J / mol'}, 'eV':{'eV'}}



    if prop_name == 'Density':
        for key, value in den_unit.items():
            if unit in value:
                return key
        if 'cm' in unit: return 'g/cm3'
        elif 'kg' in unit: return 'kg/m3'
        return unit
    
    elif prop_name in ['Glass transition temperature', 'Crystallization temp', 'Melting temp', 'Softening Point (Temperature)', 'Annealing Point (Temperature)', 'Liquidus temperature','Softening Point (Viscosity)']:
        for key, value in tg_unit.items():
            if unit in value:
                return key
        return unit
    
    elif prop_name in ['Refractive index', 'Abbe value', 'Poisson ratio','Dielectric constant']:
        return None
    
    elif prop_name in ["Young's modulus", 'Shear modulus', 'Bulk modulus']:
        for key, value in ym_unit.items():
            if unit in value:
                return key
        return unit
        
    elif prop_name == 'Vickers hardness':
        for key, value in vh_unit.items():
            if unit in value:
                return key
        return unit
    
    elif prop_name == 'Fracture toughness':
        for key, value in ft_unit.items():
            if unit in value:
                return key
        return unit
    
    elif prop_name == 'Electric conductivity':
        for key, value in ec_unit.items():
            if unit in value:
                return key
        return unit
    
    elif prop_name == 'Thermal expansion coefficient':
        for key, value in exc_unit.items():
            if unit in value:
                return key
        return unit
    
    elif prop_name == 'Activation energy':
        for key, value in ea_unit.items():
            if unit in value:
                return key
        return unit
    
    else:
        raise RuntimeError('Uncovered property')