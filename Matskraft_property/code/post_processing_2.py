import pickle
import os
import pandas as pd
import sys
import copy
import re
from utils import *
from units import *


# paper_data = pickle.load(open('../data/inference_paper_text.pkl', 'rb'))
paper_data = pickle.load(open('../data/train_val_test_paper_data_test.pkl', 'rb'))



# property_names = ['Density', 'Glass transition temperature', 'Refractive index', 'Abbe value', "Young's modulus", 'Shear modulus', 'Vickers hardness', 'Poisson ratio', 'Fracture toughness', 'Crystallization temp', 'Melting temp', 'Electric conductivity', 'Dielectric constant', 'Softening Point (Temperature)', 'Annealing Point (Temperature)', 'Thermal expansion coefficient', 'Liquidus temperature', 'Bulk modulus', 'Activation energy']




def check_non_controv_unit(prop_name, unit):
    
    den_unit = {'g/cm3', 'mg/m3','kg/m3', 'g/cm', 'lb/in3'}
    tg_unit = {'degC', 'K'}
    ri_unit = {None}
    ym_unit = {'GPa', 'MPa', 'Pa', 'psi', 'dyn/cm2', 'kb'}
#     vh_unit = {'Kg/mm2', 'GPa', 'MPa', 'Pa', 'psi'} #
    vh_unit = {'HV', 'HK', 'HRA', 'HRB', 'HRC', 'HB', 'HR', 'Shore A', 'Shore D', 'Mohs', 'GPa', 'MPa', 'kgf/mm²', 'N/mm²', 'kg/mm2', 'Pa', 'psi'}
    ft_unit = {'MPam1/2'}
    ec_unit = {'O-1 cm-1', 'O-1 m-1', 'mO-1 cm-1', 'mO-1 m-1', '(MO m)-1'}
    exc_unit = {'degC-1', 'K-1'}
    ea_unit = {'kJ/mol', 'eV/at', 'kcal/mol', 'J/mol', 'eV'}
    
    if prop_name == 'Density' and unit in den_unit:
        return True
    
    elif prop_name in ['Glass transition temperature', 'Crystallization temp', 'Melting temp', 'Softening Point (Temperature)', 'Annealing Point (Temperature)', 'Liquidus temperature','Softening Point (Viscosity)'] and unit in tg_unit:
        return True
    
    elif prop_name in ['Refractive index', 'Abbe value', 'Poisson ratio','Dielectric constant'] and unit in ri_unit:
        return True
    
    elif prop_name in ["Young's modulus", 'Shear modulus', 'Bulk modulus'] and unit in ym_unit:
        return True
        
    elif prop_name == 'Vickers hardness' and unit in vh_unit:
        return True
    
    elif prop_name == 'Fracture toughness' and unit in ft_unit:
        return True
    
    elif prop_name == 'Electric conductivity' and unit in ec_unit:
        return True
    
    elif prop_name == 'Thermal expansion coefficient' and unit in exc_unit:
        return True
    
    elif prop_name == 'Activation energy' and unit in ea_unit:
        return True
    
    return False
    
    
def get_paper_text(pii, t_idx):
    
    try:
        paper_text = paper_data[pii]
        paper_text['table_caption'] = comp_data_dict[(pii, t_idx)]['caption']
    except KeyError:
        # print(f"KeyError: The key '{pii}' was not found in paper_data.")
        return ''

    des_key = ['experiment', 'result', 'discussion', 'method']
    ptext_list = []  # List to accumulate ptext values

    
    for key in paper_text.keys():
        # Check if the key matches any descriptor or is the 'table_caption'
        if any(elem in key.lower() for elem in des_key) or key == 'table_caption':
            ptext_list.append(paper_text[key].lower())

    # Join the accumulated ptext values with '---' as separator
    combined_ptext = '---'.join(ptext_list)
    
    return combined_ptext if combined_ptext else ''

    
    
def check_paper_for_prop(pii, t_idx, prop_name):
    
#     paper_text = paper_data[pii]
    try:
        paper_text = paper_data[pii]
        paper_text['table_caption'] = comp_data_dict[(pii, t_idx)]['caption']
    except KeyError:
#         print(f"KeyError: The key '{pii}' was not found in paper_data.")
        return False

    present = False
    for key in paper_text.keys():
        des_key = ['experiment', 'result', 'discussion', 'method']
        if (any([elem in key.lower() for elem in des_key])) or (key == 'table_caption'):
            ptext = paper_text[key].lower()
            
            if prop_name == 'Density':
                if 'densit' in ptext:
                    return True

            elif prop_name in ['Glass transition temperature', 'Crystallization temp', 'Softening Point (Temperature)', 'Annealing Point (Temperature)', 'Liquidus temperature','Softening Point (Viscosity)']:
                if 'temp' in ptext:
                    return True
                
            elif prop_name in ['Melting temp']:
                if 'melting tem' in ptext:
                    return True

            elif prop_name in ['Refractive index', 'Abbe value', 'Poisson ratio', 'Dielectric constant']:
                return True

            elif prop_name in ["Young's modulus"]:
                if 'modul' in ptext or 'elastic cons' in ptext:
                    return True

            elif prop_name in ['Shear modulus', 'Bulk modulus']:
                if 'modul' in ptext:
                    return True

            elif prop_name == 'Vickers hardness':
                if 'hardness' in ptext or 'hv' in ptext: 
                    return True

            elif prop_name == 'Fracture toughness':
                if 'fracture tough' in ptext:
                    return True

            elif prop_name == 'Electric conductivity':
                if ('conduct' in ptext) and (' ion' in ptext or ' electric' in ptext):
                    return True

            elif prop_name == 'Thermal expansion coefficient':
                if ('thermal' in ptext and ('expansion' in ptext  or 'coef' in ptext)) or (' tec ' in ptext):
                    return True

            elif prop_name == 'Activation energy':
                if 'activation energ' in ptext:
                    return True
                
    return False


def check_table_for_prop(pii, t_idx, prop_name):
    # Collect all elements from the first two rows and first two columns
    search_text = []
    table = comp_data_dict[(pii, t_idx)]['act_table']

    # Add all elements in the first two rows
    for row in table[:2]:
        search_text.extend([str(cell).lower() for cell in row])

    # Add all elements in the first two columns (up to the number of rows in the table)
    for row in table:
        search_text.extend([str(cell).lower() for cell in row[:2]])

    # Combine all collected text for easier searching
    combined_text = ' '.join(search_text)

    # Property checking logic similar to check_paper_for_prop
    if prop_name == 'Density':
        if 'densit' in combined_text:
            return True

    elif prop_name in ['Crystallization temp', 'Softening Point (Temperature)', 'Annealing Point (Temperature)', 'Liquidus temperature', 'Softening Point (Viscosity)']:
        if 'temp' in combined_text:
            return True
        
    elif prop_name in ['Glass transition temperature']:
        if 'temp' in combined_text or 'tg' in combined_text or 't g' in combined_text:
            return True

    elif prop_name in ['Melting temp']:
        if 'melting tem' in combined_text or 'tm' in combined_text or 't m' in combined_text:
            return True

    elif prop_name in ['Refractive index', 'Abbe value', 'Poisson ratio', 'Dielectric constant']:
        return True
    
    elif prop_name in ["Young's modulus"]:
        if 'modul' in combined_text or 'elastic cons' in combined_text or 'c11' in combined_text or 'c 11' in combined_text or 'c33' in combined_text or 'c 33' in combined_text or 'c44' in combined_text or ' e ' in combined_text:
            return True

    elif prop_name in ['Shear modulus', 'Bulk modulus']:
        if 'modul' in combined_text:
            return True

    elif prop_name == 'Vickers hardness':
        if 'hardness' in combined_text or 'hv' in combined_text or 'h v' in combined_text:
            return True

    elif prop_name == 'Fracture toughness':
        if 'fracture tough' in combined_text:
            return True

    elif prop_name == 'Electric conductivity':
        if ('conduct' in combined_text) and ('ion' in combined_text or 'electric' in combined_text or 'dc' in combined_text):
            return True

    elif prop_name == 'Thermal expansion coefficient':
        if ('thermal' in combined_text and ('expansion' in combined_text  or 'coef' in combined_text)) or ('tec' in combined_text):
            return True

    elif prop_name == 'Activation energy':
        if 'activation energ' in combined_text:
            return True

    return False


def final_checker_on_units(pred_tuples):
    
    indices_to_remove = set()
#     indices_to_remove.add(ind)
    
    for ind, tup in enumerate(pred_tuples):
        pii = tup[0].split('_')[0]
        t_idx = int(tup[0].split('_')[1])
        prop_name = tup[1]
        prop_value = tup[2]
        unit = tup[-1]
        
        if not check_non_controv_unit(prop_name, unit):
            unit = unit.replace(' ', '')
            
            if prop_name == 'Density':
                unit = unit.lower()
                
                if unit in ['r', 'calculated']:
                    if prop_value<=25:
                        unit = 'g/cm3'
                    elif prop_value>25:
                        unit = 'kg/m3'
                        
                elif unit == '':
                    if check_table_for_prop(pii, t_idx, prop_name) or check_paper_for_prop(pii, t_idx, prop_name):
                        if prop_value<=25:
                            unit = 'g/cm3'
                        else:
                            unit = 'kg/m3'
                            
                    else:
                        indices_to_remove.add(ind)
                        continue
                        
                else:
                    if any(char in unit for char in ['g', 'm', '3', 'lb']):
                        continue
                        
                    elif check_table_for_prop(pii, t_idx, prop_name) or check_paper_for_prop(pii, t_idx, prop_name):
                        if prop_value<=25:
                            unit = 'g/cm3'
                        elif prop_value>25:
                            unit = 'kg/m3'
                            
                    else:
                        indices_to_remove.add(ind)
                        continue
                        
                        
            elif prop_name in ['Glass transition temperature', 'Crystallization temp', 'Melting temp', 'Softening Point (Temperature)', 'Annealing Point (Temperature)', 'Liquidus temperature','Softening Point (Viscosity)']:
        
                if any(char in unit for char in ['C', 'K', 'deg']):
                    if any(char in unit for char in ['C', 'deg']):
                        unit = 'degC'
                    elif any(char in unit for char in ['K']):
                        unit = 'K'

                elif unit == '' and (check_table_for_prop(pii, t_idx, prop_name) or check_paper_for_prop(pii, t_idx, prop_name)):
                    continue

                else:
                    indices_to_remove.add(ind)
                    continue
                        
                        
            elif prop_name in ['Shear modulus', 'Bulk modulus']: #as precision is high we didnt remove tuples based on units
                
                if check_table_for_prop(pii, t_idx, prop_name) or check_paper_for_prop(pii, t_idx, prop_name):
                    
                    if prop_value<600:
                        unit = 'GPa'
                    elif prop_value>1000:
                        unit = 'MPa'
                            
            elif prop_name in ["Young's modulus"]:
                if any(char in unit for char in ['C/']): #unit for piezoelectric constant
                    indices_to_remove.add(ind)
                    continue
                if any(char in unit for char in ['bar','N/', 'm2', 'm-2']):
                    continue
                elif check_table_for_prop(pii, t_idx, prop_name) or check_paper_for_prop(pii, t_idx, prop_name) or any(char in unit for char in ['E', 'Pa', 'pa']):
                    if prop_value<600:
                        unit = 'GPa'
                    elif prop_value>2000 and prop_value<600000:
                        unit = 'MPa'
                else:
                    indices_to_remove.add(ind)
                    continue
                    
                    
            elif prop_name in ['Electric conductivity']:
                if unit in ['min', 'minute', 's', 'sec', 'second', 'hr', 'hour', 'ms', 'degree', 'degrees', 'Degree', 'K', 'degC', 'C', 'degc', 'ohm', 'MPa', 'GPa', 'eV', 'A']: #unit for other props
                    indices_to_remove.add(ind)
                    continue
                elif unit in ['ohm-cm', 'ohmcm', 'Ohm-cm', 'Ohmcm', 'ohm-m', 'ohmm', 'Ohm-m', 'Ohmm', 'Ocm', 'O-cm', 'O-m', 'Om']:
                    if prop_value == 0:
                        indices_to_remove.add(ind)
                        continue
                    
                    new_value = 1 / prop_value # resistivity has been extracted instead of conductivity
                    if 'cm' in unit:
                        unit = 'O-1 cm-1'  # Set unit for conductivity in cm-based units
                    elif 'm' in unit:
                        unit = 'O-1 m-1'
                    else:
                        new_value = prop_value
                        
                    new_unit_norm = norm_unit(unit, prop_name)

                    # Update both the value and the unit in the tuple
                    new_tuple = (*tup[:2], new_value, new_unit_norm)
#                     ind  = pred_tuples.index(tup)
                    pred_tuples[ind] = new_tuple
                    continue
                    
                elif check_table_for_prop(pii, t_idx, prop_name) or check_paper_for_prop(pii, t_idx, prop_name):
                    if 'cm' in unit and ('-1' in unit or '/' in unit):
                        unit = 'O-1 cm-1'  # Set unit for conductivity in cm-based units
                    elif 'm' in unit and ('-1' in unit or '/' in unit):
                        unit = 'O-1 m-1'
                        
                elif ('O' in unit or 'S' in unit) and 'm' in unit and ('-1' in unit or '/' in unit):
                    continue
                 
                else:
                    indices_to_remove.add(ind)
                    continue
                    
            elif prop_name in ['Thermal expansion coefficient']:
                if (unit in ['min', 's', 'sec', 'second', 'hr', 'hour', 'ms', 'degree', 'degrees', 'Degree', 'K', 'degC', 'degc', 'deg', 'W/mK', 'W/mk', 'w/mk', 'W/cmK', 'W/cmk', 'w/cmk', 'ohm', 'MPa', 'GPa', 'eV']) or ('mm' in unit or 'cm' in unit or 'mK' in unit): #unit for other props
                    indices_to_remove.add(ind)
                    continue
                elif ('C' in unit or 'K' in  unit) and ('-1' in unit or '/' in unit): #{'degC-1', 'K-1'}
                    if 'C' in unit:
                        unit = 'degC-1'
                    elif 'K' in unit:
                        unit = 'K-1'
                      
                elif ('10' in unit) and any(char in unit for char in ['-','−', '–']):
                    continue
                    
                elif check_table_for_prop(pii, t_idx, prop_name) or check_paper_for_prop(pii, t_idx, prop_name):
                    if 'C' in unit:
                        unit = 'degC-1'
                    elif 'K' in unit:
                        unit = 'K-1'
                        
                else:
                    indices_to_remove.add(ind)
                    continue
                    
                    
            elif prop_name in ['Fracture toughness']:
                if (unit in ['min', 's', 'sec', 'second', 'hr', 'hour', 'ms', 'degree', 'degrees', 'Degree', 'K', 'degC', 'degc', 'deg', 'ohm', 'MPa', 'GPa', 'eV', 'Gpa', 'Mpa', 'MPA', 'GPA']) or ('m3' in unit or 'm2' in unit or 'm-3' in unit or 'm-2' in unit): #unit for other props
                    indices_to_remove.add(ind)
                    continue
                    
                elif check_table_for_prop(pii, t_idx, prop_name) or check_paper_for_prop(pii, t_idx, prop_name):
                    unit = 'MPam1/2'
                    
                elif 'J' in unit or 'MPa' in unit or 'm1/2' in unit or 'K' in unit or 'ksi' in unit or 'psi' in unit:
                    continue
                    
                else:
                    indices_to_remove.add(ind)
                    continue
                    
                    
            elif prop_name in ['Activation energy']:
                if (unit in ['min', 's', 'sec', 'second', 'hr', 'hour', 'ms', 'degree', 'degrees', 'Degree', 'K', 'degC', 'degc', 'deg', 'ohm', 'MPa', 'GPa', 'Gpa', 'Mpa', 'MPA', 'GPA']) or ('mm' in unit or 'cm' in unit): #unit for other props
                    indices_to_remove.add(ind)
                    continue
                      
                elif check_table_for_prop(pii, t_idx, prop_name) or check_paper_for_prop(pii, t_idx, prop_name):        
                    if prop_value <= 2:
                        unit =  'eV'  # most likely eV in this range
                    elif 5.0 < prop_value <= 50:
                        unit = 'kcal/mol'  # most likely kcal/mol in this range
                    elif 200 < prop_value <= 500:
                        unit = 'kJ/mol'        
                        
                elif 'eV' in unit or '/mol' in unit or '/at' in unit:
                    continue
                    
                else:
                    indices_to_remove.add(ind)
                    continue       
                
            elif prop_name in ['Vickers hardness']:
                
                if unit.lower() in ['min', 's', 'sec', 'second', 'hour', 'ms', 'degree', 'degrees', 'degree', 'k', 'degc', 'deg', 'ohm',  'ev', 'mev', 'kvcm', 'ghz', 'wmk', 'mgl', 'mpa/m12', 'mgl1', 'rad', 'deg', 'a', 'c']: #unit for other props
                    indices_to_remove.add(ind)
                    continue
                
                papertext = get_paper_text(pii, t_idx)
                is_valid, standardized_unit = process_hardness_data(prop_value, unit, papertext)
                if is_valid:
                    unit = standardized_unit
                elif check_table_for_prop(pii, t_idx, prop_name) or check_paper_for_prop(pii, t_idx, prop_name):
                    unit = 'HV'          
            
                  
                
                
        new_unit_norm = norm_unit(unit, prop_name)            
        new_tuple = (*tup[:-1], new_unit_norm)
#         ind  = pred_tuples.index(tup)
        pred_tuples[ind] = new_tuple
                    
    upd_f_pred = [tup for idx, tup in enumerate(pred_tuples) if idx not in indices_to_remove]                
                    
    return upd_f_pred



def remove_spaces_from_units(pred_tuples): 
    """Remove any space from the units of the tuple, except for 'Shore A' and 'Shore D'"""
    
    for i, tup in enumerate(pred_tuples): 
        unit = tup[-1]
        prop_name = tup[1]
        
        if prop_name == 'Electric conductivity':
            continue
        
        # Handle space removal
        if isinstance(unit, str):  # Check if unit is string
            if 'shore' not in unit.lower():
                unit = unit.replace(' ', '')
        
        # Normalize unit
        try:
            new_unit_norm = norm_unit(unit, prop_name)
            pred_tuples[i] = (*tup[:-1], new_unit_norm)
        except Exception as e:
            continue  # Skip if normalization fails
            
    return pred_tuples



def standardize_hardness_unit(unit_str):
    # Convert to lowercase and remove ALL spaces
    unit = unit_str.lower()
    unit = ''.join(c for c in unit if c.isalnum() or c in '/²')
    
    # Invalid units to reject immediately
    invalid_units = {
        'min', 's', 'sec', 'second', 'hr', 'hour', 'ms', 'degree', 'degrees', 'Degree', 'K', 'degC', 'degc',
        'deg', 'ohm', 'MPa', 'GPa', 'Gpa', 'Mpa', 'MPA', 'GPA', 
        'ev', 'mev', 'kvcm', 'ghz', 'wmk', 'mgl', 'mpa/m12', 'mgl1', 'rad', 
        'deg', 'kgoe', 'kam', 's1', 're2o3', 'nm', 'mn', 'ncm', 'gf', 
        'matrix', 'fibers', 'average', 'initial', 'sharp', 'axial', 'radial',
        'e', 'f', 'n', 'm', 'a', 'r', 'd', 's'
    }
    
    if unit in invalid_units:
        return "Invalid"

    # 1. Vickers patterns with loads
    vickers_with_load = {
        'hv1': 'HV1', 
        'hv2': 'HV2', 
        'hv3': 'HV3',
        'hv5': 'HV5', 
        'hv10': 'HV10',
        'hvn1': 'HV1',
        'hvn10': 'HV10',
        'vhn1': 'HV1',
        'vhn10': 'HV10',
        'hv3mn': 'HV',
        'hv10g': 'HV10',
        'hvit': 'HV',
        'uhv': 'HV',
        'inhv': 'HV',
        'hv/1': 'HV1',
        'hv/10': 'HV10'
    }
    
    # 2. Rockwell scales
    rockwell_specific = {
        'hra': 'HRA',
        'hrb': 'HRB',
        'hrc': 'HRC',
        'hrd': 'HRD',
        'hrf': 'HRF',
        'hrl': 'HRL',
        'hrh': 'HRH',
        'rc': 'HRC',
        'rockwellc': 'HRC',
        'ra': 'HRA',
        'ha': 'HRA',
        'hci': 'HRC'
    }
    
    # 3. Shore scales
    shore_specific = {
        'shorea': 'Shore A',
        'shored': 'Shore D',
        'sha': 'Shore A',
        'sd': 'Shore D',
        'hsd': 'Shore D',
        'hs': 'Shore'
    }
    
    # 4. Basic hardness scales
    hardness_scales = {
        # Vickers variants
        'hv': 'HV',
        'hvn': 'HV',
        'vhn': 'HV',
        'vpn': 'HV',
        'dph': 'HV',
        'vdh': 'HV',
        'vickers': 'HV',
        'vickershv': 'HV',
        
        # Knoop variants
        'hk': 'HK',
        'khn': 'HK',
        'hhk': 'HK',
        'knoop': 'HK',
        
        # Brinell variants
        'hb': 'HB',
        'bhn': 'HB',
        'hbn': 'HB',
        'hbw': 'HB',
        'brinell': 'HB'
    }
    
    # 5. Pressure units with variations
    pressure_units = {
        'gpa': 'GPa',
        'mpa': 'MPa',
        'pa': 'Pa',
        'nmm2': 'N/mm²',
        'kgfmm2': 'kgf/mm²',
        'kgmm2': 'kgf/mm²',
        'kgnmm2': 'kgf/mm²',
        'kgfcm2': 'kgf/cm²',
        'kgcm2': 'kgf/cm²',
        'dynecm2': 'dyne/cm²',
        'gnm2': 'GN/m²',
        'mnm2': 'MN/m²',
        'kgnm2': 'kgf/mm²',
        'gmm2': 'g/mm²',
        'kgmm': 'kgf/mm²',
        'kgxmm2': 'kgf/mm²'
    }
    
    # Check in order of specificity
    if unit in vickers_with_load:
        return vickers_with_load[unit]
        
    if unit in rockwell_specific:
        return rockwell_specific[unit]
        
    if unit in shore_specific:
        return shore_specific[unit]
        
    if unit in hardness_scales:
        return hardness_scales[unit]
        
    if unit in pressure_units:
        return pressure_units[unit]
    
    # Check for numeric patterns in Vickers
    if any(x in unit for x in ['hv', 'hvn', 'vhn']):
        numbers = ''.join(filter(str.isdigit, unit))
        if numbers:
            return f"HV{numbers}"
        return "HV"
        
    if 'mohs' in unit:
        return 'Mohs'
        
    return "Invalid"


def assign_hardness_unit_from_value_and_text(value, papertext):
    
    if len(papertext) == 0:
        return "Invalid"
    
    papertext = papertext.lower()
    
    # More specific text checks with expanded variations
    has_vickers = any(term in papertext for term in ['vicker', 'hv', 'h v', 'vhn', 'hvn', 'microhardness', 'micro-hardness', 'micro hardness'])
#     has_vickers_gpa = ('gpa' in papertext or 'gigapascal' in papertext) and has_vickers
#     has_vickers_mpa = ('mpa' in papertext or 'megapascal' in papertext) and has_vickers
    
    has_knoop = any(term in papertext for term in ['knoop', 'khn', 'hk'])
    has_brinell = any(term in papertext for term in ['brinell', 'bhn', 'hb', 'brinell hardness'])
    has_mohs = any(term in papertext for term in ['mohs', 'moh', 'mohs scale'])
    
    # Specific Shore checks
    has_shore_a = any(term in papertext for term in ['shore a', 'sha ', 'shore-a', 'shore type a'])
    has_shore_d = any(term in papertext for term in ['shore d', 'shd', 'shore-d', 'shore type d'])
    
    # More specific Rockwell checks
    has_rockwell = any(term in papertext for term in ['rockwell', 'hr'])
    has_rockwell_a = any(term in papertext for term in ['hra', 'rockwell a', 'rockwell type a', 'rockwell scale a'])
    has_rockwell_b = any(term in papertext for term in ['hrb', 'rockwell b', 'rockwell type b', 'rockwell scale b'])
    has_rockwell_c = any(term in papertext for term in ['hrc', 'rockwell c', 'rockwell type c', 'rockwell scale c'])

    if value > 0:  # Hardness can't be negative
        # First check explicit unit mentions with value ranges


        # Vickers
        if has_vickers:
            if 5 <= value <= 10000:
                return "HV"
            # Additional check for very small values that might be in GPa
            elif 0.1 <= value < 5 and ('hardness' in papertext):
                return "GPa"
        
        # Rockwell scales - check for explicit scale mentions
        if has_rockwell:
            if has_rockwell_a:
                return "HRA"
            elif has_rockwell_b:
                return "HRB"
            elif has_rockwell_c:
                return "HRC"
            else:
                return "HR"  # If scale not explicitly mentioned
        
        # Shore scales with specific mentions
        if has_shore_a:
            return "Shore A"
        if has_shore_d:
            return "Shore D"
        
        # Other scales requiring specific mentions and value ranges
        if has_mohs:
            return "Mohs"
        if has_knoop:
            return "HK"
        if has_brinell:
            return "HB"
        
    
    return "Invalid"



def validate_hardness_value(value, unit):
    """
    Validates if the hardness value is within theoretical possible ranges
    Args:
        value: numerical value of hardness
        unit: standardized unit string
    Returns:
        bool: True if value is within valid range, False otherwise
    """
    if value <= 0:  # Hardness can never be negative or zero
        return False
        
    # Define extremely wide ranges to include all theoretical possibilities
    ranges = {
        # Standard hardness scales
        'HV': (5, 10000),      # Extended for ultra-hard coatings and very soft materials
        'HK': (10, 8000),      # Extended for extreme cases
        'HB': (5, 1000),       # Extended for all possible materials
        'HRA': (10, 95),       # Full possible scale range
        'HRB': (10, 100),      # Full possible scale range
        'HRC': (10, 100),      # Extended beyond standard scale
        'HR': (10, 100),       # Generic Rockwell (widest range)
        'Shore A': (0, 100),   # Full scale
        'Shore D': (0, 100),   # Full scale
        'Mohs': (1, 20),       # Extended for synthetic super-hard materials
        
        # Pressure units (extended ranges)
        'GPa': (0.05, 100),    # Extended for both extremely soft and ultra-hard materials
        'MPa': (50, 100000),   # Corresponding MPa range
        'kgf/mm²': (5, 10000), # Corresponding kgf/mm² range
        'N/mm²': (50, 100000)  # Same as MPa
    }
    
    # Check if unit exists in our defined ranges
    if unit not in ranges:
        return True
    
    # Get min and max values for the unit
    min_val, max_val = ranges[unit]
    
    # Check if value falls within the valid range
    return min_val <= value <= max_val



def process_hardness_data(value, original_unit, papertext):
    # First try to standardize the unit if it exists
    standardized_unit = standardize_hardness_unit(original_unit)
    
    # If standardization failed, try to assign based on value and text
    if standardized_unit == "Invalid":
        standardized_unit = assign_hardness_unit_from_value_and_text(value, papertext)
    
    # Validate the final unit assignment
    if standardized_unit == "Invalid":
        return False, "Invalid unit"
    
    is_valid = validate_hardness_value(value, standardized_unit)
    return is_valid, standardized_unit

            
    
    
def sort_by_length_and_content(lst):
    return sorted(lst, key=lambda x: (len(x), x), reverse=True)


def find_unit_in_cap(caption, prop_name):
    
    ptext = caption
    
    den_all_unit = ['gm/cm3', 'g cm-3', 'g/cm-3', 'g/cm3', 'g/cm 3', 'gcm -3', 'gcm-3', 'g/cc', 'gm/cc', 'gw/cm3', 'gm cc-1', 'g/ cm3', 'g/mL', 'd gcm-3', 'Mg/m3', 'mgm-3', 'kgm -3', 'kgm-3', 'kg/m3', 'kg m-3', 'g/l', 'g/cm -1', 'gcm -1', 'gcm-1', 'g cm-1', 'g/cm', 'lb/in3', 'lb/ in3', 'lb in-3', 'lbin -3']
    #den_all_unit = sort_by_length_and_content(den_all_unit)

    tg_all_unit = ['degC', 'C', 'K']
    #tg_all_unit = sort_by_length_and_content(tg_all_unit)

    ym_all_unit = ['GPa', 'Mpa', 'MPa', 'psi', 'dyn/cm2', 'kb']
    #ym_all_unit = sort_by_length_and_content(ym_all_unit)

    vh_all_unit = ['Kg/mm2', 'Kgf/mm 2', 'kg/mm2', 'kgfmm-2', 'kgmm -2', 'GPa', 'MPa', 'Pa', 'psi']
    #vh_all_unit = sort_by_length_and_content(vh_all_unit)

    ft_all_unit = ['MPam1/2', 'MPa m', 'MPa m1/2', 'MPam1/2', 'MPa/sqrt(m)', 'MPa(m)1/2']
    ft_all_unit = sort_by_length_and_content(ft_all_unit)

    ec_all_unit = ['O-1 cm- 1', 'O-1 cm-1', 'ohm-1 cm-1', 'Scm-1', 'S cm-1', 'S/cm', '(O-cm)-1', 'O-1 m- 1', 'O-1 m-1', 'ohm-1 m-1', '(Om)-1', '(O m)-1', 'Sm-1', 'S m-1', 'S/m', '(MO m)-1']
    ec_all_unit = sort_by_length_and_content(ec_all_unit)

    exc_all_unit = ['degC-1', '/degC', 'C-1', 'K-1', '/K']
    exc_all_unit = sort_by_length_and_content(exc_all_unit)

    ea_all_unit = ['kJmol-1', 'kJ mol-1', 'kJ/mol', 'kJ / mol', 'kJ/molK', 'kJmol -1', 'eV/at', 'kcal/mol', 'kcalmol-1', 'kcal mol-1', 'Jmol-1', 'J mol-1', 'J/mol', 'J / mol', 'eV']
    ea_all_unit = sort_by_length_and_content(ea_all_unit)
    
    
    if prop_name == 'Density':
        for unit in den_all_unit:
            if unit in ptext:
                return unit

    elif prop_name in ['Glass transition temperature', 'Crystallization temp', 'Melting temp', 'Softening Point (Temperature)', 'Annealing Point (Temperature)', 'Liquidus temperature','Softening Point (Viscosity)']:
        for unit in tg_all_unit:
            # Ensure unit is properly captured by enclosing it in parentheses
            temp_unit_pattern = rf'(\d+(?:\.\d+)?)\s*({unit})(?=[\s),.\]])'

            # Search for match
            match = re.search(temp_unit_pattern, ptext)

            if match:
                value, matched_unit = match.groups()  # Ensure both are correctly extracted
                if len(matched_unit) > 0:
                    return matched_unit  # Return extracted unit

            unit = unit + ' '
            if unit in ptext:
                return unit

    elif prop_name in ['Refractive index', 'Abbe value', 'Poisson ratio','Dielectric constant']:
        return None

    elif prop_name in ["Young's modulus", 'Shear modulus', 'Bulk modulus']:
        for unit in ym_all_unit:
            if unit in ptext:
                return unit

    elif prop_name == 'Vickers hardness':
        return ''
#                 for unit in vh_all_unit:
#                     if unit in ptext:
#                         return unit

    elif prop_name == 'Fracture toughness':
        for unit in ft_all_unit:
            if unit in ptext:
                return unit

    elif prop_name == 'Electric conductivity':
        for unit in ec_all_unit:
            if unit in ptext:
                return unit

    elif prop_name == 'Thermal expansion coefficient':
        for unit in exc_all_unit:
            if unit in ptext:
                return unit

    elif prop_name == 'Activation energy':
        for unit in ea_all_unit:
            if unit in ptext:
                return unit
    


def find_unit_in_paper(pii, t_idx, prop_name):
    
    if prop_name in ['Refractive index', 'Abbe value', 'Poisson ratio','Dielectric constant']:
        return None
    
    
    den_all_unit = ['gm/cm3', 'g cm-3', 'g/cm-3', 'g/cm3', 'g/cm 3', 'gcm -3', 'gcm-3', 'g/cc', 'gm/cc', 'gw/cm3', 'gm cc-1', 'g/ cm3', 'g/mL', 'd gcm-3', 'Mg/m3', 'mgm-3', 'kgm -3', 'kgm-3', 'kg/m3', 'kg m-3', 'g/l', 'g/cm -1', 'gcm -1', 'gcm-1', 'g cm-1', 'g/cm', 'lb/in3', 'lb/ in3', 'lb in-3', 'lbin -3']
    #den_all_unit = sort_by_length_and_content(den_all_unit)

    tg_all_unit = ['degC', 'C', 'K']
    #tg_all_unit = sort_by_length_and_content(tg_all_unit)

    ym_all_unit = ['GPa', 'Mpa', 'MPa', 'psi', 'dyn/cm2', 'kb']
    #ym_all_unit = sort_by_length_and_content(ym_all_unit)

    vh_all_unit = ['Kg/mm2', 'Kgf/mm 2', 'kg/mm2', 'kgfmm-2', 'kgmm -2', 'GPa', 'MPa', 'Pa', 'psi']
    #vh_all_unit = sort_by_length_and_content(vh_all_unit)

    ft_all_unit = ['MPam1/2', 'MPa m', 'MPa m1/2', 'MPam1/2', 'MPa/sqrt(m)', 'MPa(m)1/2']
    ft_all_unit = sort_by_length_and_content(ft_all_unit)

    ec_all_unit = ['O-1 cm- 1', 'O-1 cm-1', 'ohm-1 cm-1', 'Scm-1', 'S cm-1', 'S/cm', '(O-cm)-1', 'O-1 m- 1', 'O-1 m-1', 'ohm-1 m-1', '(Om)-1', '(O m)-1', 'Sm-1', 'S m-1', 'S/m', '(MO m)-1']
    ec_all_unit = sort_by_length_and_content(ec_all_unit)

    exc_all_unit = ['degC-1', '/degC', 'C-1', 'K-1', '/K']
    exc_all_unit = sort_by_length_and_content(exc_all_unit)

    ea_all_unit = ['kJmol-1', 'kJ mol-1', 'kJ/mol', 'kJ / mol', 'kJ/molK', 'kJmol -1', 'eV/at', 'kcal/mol', 'kcalmol-1', 'kcal mol-1', 'Jmol-1', 'J mol-1', 'J/mol', 'J / mol', 'eV']
    ea_all_unit = sort_by_length_and_content(ea_all_unit)

    
#     paper_text = paper_data[pii]
    try:
        paper_text = paper_data[pii]
        paper_text['table_caption'] = comp_data_dict[(pii, t_idx)]['caption']
    except KeyError:
#         print(f"KeyError: The key '{pii}' was not found in paper_data.")
        return ''

    cap_unit = find_unit_in_cap(paper_text['table_caption'], prop_name)
    if cap_unit and len(cap_unit) > 0:
        return cap_unit


    for key in paper_text.keys():
        des_key = ['experiment', 'result', 'discussion', 'method']
        if any([elem in key.lower() for elem in des_key]): #or (key == 'table_caption'):
            ptext = paper_text[key] #Dont do lower here, this is case sensitive
            
            if prop_name == 'Density' and 'density' in ptext:
                for unit in den_all_unit:
                    if unit in ptext:
                        return unit

            elif prop_name in ['Glass transition temperature', 'Crystallization temp', 'Melting temp', 'Softening Point (Temperature)', 'Annealing Point (Temperature)', 'Liquidus temperature','Softening Point (Viscosity)']  and 'temp' in ptext:
                for unit in tg_all_unit:
                    
                    # Ensure unit is properly captured by enclosing it in parentheses
                    temp_unit_pattern = rf'(\d+(?:\.\d+)?)\s*({unit})(?=[\s),.\]])'

                    # Search for match
                    match = re.search(temp_unit_pattern, ptext)

                    if match:
                        value, matched_unit = match.groups()  # Ensure both are correctly extracted
                        if len(matched_unit) > 0:
                            return matched_unit  # Return extracted unit

                    unit = unit + ' '
                    if unit in ptext:
                        return unit

#             elif prop_name in ['Refractive index', 'Abbe value', 'Poisson ratio','Dielectric constant']:
#                 return None

            elif prop_name in ["Young's modulus", 'Shear modulus', 'Bulk modulus']  and 'modul' in ptext:
                for unit in ym_all_unit:
                    if unit in ptext:
                        return unit

            elif prop_name == 'Vickers hardness'  and 'hardness' in ptext:
                return ''
#                 for unit in vh_all_unit:
#                     if unit in ptext:
#                         return unit

            elif prop_name == 'Fracture toughness' and 'toughness' in ptext:
                for unit in ft_all_unit:
                    if unit in ptext:
                        return unit

            elif prop_name == 'Electric conductivity'  and 'conduct' in ptext:
                for unit in ec_all_unit:
                    if unit in ptext:
                        return unit

            elif prop_name == 'Thermal expansion coefficient' and ('thermal' in ptext or 'tec' in ptext):
                for unit in exc_all_unit:
                    if unit in ptext:
                        return unit

            elif prop_name == 'Activation energy' and 'activation' in ptext:
                for unit in ea_all_unit:
                    if unit in ptext:
                        return unit
                    
    return ''


def check_heading_for_unit(heading, prop_name):
    
    den_all_unit = ['gm/cm3', 'g cm-3', 'g/cm-3', 'g/cm3', 'g/cm 3', 'gcm -3', 'gcm-3', 'g/cc', 'gm/cc', 'gw/cm3', 'gm cc-1', 'g/ cm3', 'g/mL', 'd gcm-3', 'Mg/m3', 'mgm-3', 'kgm -3', 'kgm-3', 'kg/m3', 'kg m-3', 'g/l', 'g/cm -1', 'gcm -1', 'gcm-1', 'g cm-1', 'g/cm', 'lb/in3', 'lb/ in3', 'lb in-3', 'lbin -3']
    #den_all_unit = sort_by_length_and_content(den_all_unit)

    tg_all_unit = ['degC', 'C', 'K']
    #tg_all_unit = sort_by_length_and_content(tg_all_unit)

    ym_all_unit = ['GPa', 'Mpa', 'Pa', 'psi', 'dyn/cm2', 'kb']
    #ym_all_unit = sort_by_length_and_content(ym_all_unit)

    vh_all_unit = ['Kg/mm2', 'Kgf/mm 2', 'kg/mm2', 'kgfmm-2', 'kgmm -2', 'GPa', 'MPa', 'Pa', 'psi']
    #vh_all_unit = sort_by_length_and_content(vh_all_unit)

    ft_all_unit = ['MPam1/2', 'MPa m', 'MPa m1/2', 'MPam1/2', 'MPa/sqrt(m)']
    ft_all_unit = sort_by_length_and_content(ft_all_unit)

    ec_all_unit = ['O-1 cm- 1', 'O-1 cm-1', 'ohm-1 cm-1', 'Scm-1', 'S cm-1', 'S/cm', '(O-cm)-1', 'O-1 m- 1', 'O-1 m-1', 'ohm-1 m-1', '(Om)-1', '(O m)-1', 'Sm-1', 'S m-1', 'S/m', '(MO m)-1', 'mSm-1', 'mS m-1', 'mScm-1', 'mS cm-1']
    ec_all_unit = sort_by_length_and_content(ec_all_unit)

    exc_all_unit = ['degC-1', '/degC', 'C-1', 'K-1', '/K', 'K -1', 'C -1']
    exc_all_unit = sort_by_length_and_content(exc_all_unit)

    ea_all_unit = ['kJmol-1', 'kJ mol-1', 'kJ/mol', 'kJ / mol', 'kJ/molK', 'kJmol -1', 'eV/at', 'kcal/mol', 'kcalmol-1', 'kcal mol-1', 'Jmol-1', 'J mol-1', 'J/mol', 'J / mol', 'eV']
    ea_all_unit = sort_by_length_and_content(ea_all_unit)

            
    if prop_name == 'Density':
        for unit in den_all_unit:
            if unit in heading:
                return unit

    elif prop_name in ['Glass transition temperature', 'Crystallization temp', 'Melting temp', 'Softening Point (Temperature)', 'Annealing Point (Temperature)', 'Liquidus temperature', 'Softening Point (Viscosity)']:
        for unit in tg_all_unit:
            unit = unit + ' '
            if unit in heading:
                return unit

    elif prop_name in ['Refractive index', 'Abbe value', 'Poisson ratio','Dielectric constant']:
        return None

    elif prop_name in ["Young's modulus", 'Shear modulus', 'Bulk modulus']:
        for unit in ym_all_unit:
            if unit in heading:
                return unit

    elif prop_name == 'Vickers hardness':
        for unit in vh_all_unit:
            if unit in heading:
                return unit

    elif prop_name == 'Fracture toughness':
        for unit in ft_all_unit:
            if unit in heading:
                return unit

    elif prop_name == 'Electric conductivity':
        for unit in ec_all_unit:
            if unit in heading:
                return unit
        
        #'O-1 cm-1', 'O-1 m-1'
        if ('mO' in heading or 'mS' in heading) and 'cm' in heading and ('-1' in heading or '/' in heading):
            return 'mO-1 cm-1'
        elif ('mO' in heading or 'mS' in heading) and 'm' in heading and ('-1' in heading or '/' in heading):
            return 'mO-1 m-1'
        elif ('O' in heading or 'S' in heading) and 'cm' in heading and ('-1' in heading or '/' in heading):
            return 'O-1 cm-1'
        elif ('O' in heading or 'S' in heading) and 'm' in heading and ('-1' in heading or '/' in heading):
            return 'O-1 m-1'
        

    elif prop_name == 'Thermal expansion coefficient':
        for unit in exc_all_unit:
            if unit in heading:
                return unit

    elif prop_name == 'Activation energy':
        for unit in ea_all_unit:
            if unit in heading:
                return unit
                    
    return ''