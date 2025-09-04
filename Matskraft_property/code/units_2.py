import re

# Property type mapping to standard property names
prop_type_mapping = {
    'Density': 'density',
    'Glass transition temperature': 'temperature',
    'Refractive index': 'refractive_index',
    'Abbe value': 'refractive_index',
    "Young's modulus": 'youngs_modulus',
    'Shear modulus': 'youngs_modulus',
    'Vickers hardness': 'vickers_hardness',
    'Poisson ratio': 'refractive_index',
    'Fracture toughness': 'fracture_toughness',
    'Crystallization temp': 'temperature',
    'Melting temp': 'temperature',
    'Electric conductivity': 'electric_conductivity',
    'Dielectric constant': 'refractive_index',
    'Softening Point (Temperature)': 'temperature',
    'Annealing Point (Temperature)': 'temperature',
    'Thermal expansion coefficient': 'thermal_expansion',
    'Liquidus temperature': 'temperature',
    'Bulk modulus': 'youngs_modulus',
    'Activation energy': 'activation_energy'
}


patterns = {
    
    'density': r'(?i)[\(\[\{]?\b((?:g\/cm3|g\/cm³|gcm-3|gcm−3|g\.cm-3|g\.cm−3|g\/cc|g\/ml|gml-1|gm\/cm3|gm\/cc|gw\/cm3|gmcc-1|mg\/m3|mg\/m³|mgm-3|mgm−3|kg\/m3|kg\/m³|kgm-3|kgm−3|g\/l|g\s?cm-3|g\s?cm−3|g\/cm\s?3|lb\/in3|lbin-3))\b[\)\]\}]?',
    
    
#     'temperature': r'[\(\[\{]?\b((?:degC|c|degc|°c|(?<!\w)K(?!\w)|kelvin))\b[\)\]\}]?',
    'temperature': r'(?i)[\(\[\{]?\b((?:degC|°C|(?<![a-zA-Z])C(?![a-zA-Z])|(?<![a-zA-Z])K(?![a-zA-Z])|kelvin))\b[\)\]\}]?',
    
    
    'youngs_modulus': r'(?i)[\(\[\{]?\b((?:gpa|mpa|pa|psi|dyn\s?\/?\s?cm2|dyn\s?cm[-−]2|kb))\b[\)\]\}]?',
    
    
    'vickers_hardness': r'(?i)[\(\[\{]?\b((?:kg\/mm2|kgf\/mm2|kgfmm-2|kgmm-2|gpa|mpa|pa|psi))\b[\)\]\}]?',
    
    
    'fracture_toughness': r'(?i)[\(\[\{]?\b((?:mpam|mpam1\/2|mpa\/sqrt\(m\)|mpa\s?\(m\)1\/2))\b[\)\]\}]?',
    
    
    'electric_conductivity': r'(?i)[\(\[\{]?\b((?:s\/cm|scm-1|ohm-1cm-1|\(o-cm\)-1|s\/m|sm-1|ohm-1m-1|\(om\)-1|\(mo\s?m\)-1))\b[\)\]\}]?',
    
    
    'thermal_expansion': r'(?i)[\(\[\{]?\b((?:degc-1|c-1|\/degc|\/c|k-1|\/k))\b[\)\]\}]?',
    
    
    'activation_energy': r'(?i)[\(\[\{]?\b((?:kj\/mol|kjmol-1|kj\/molk|ev\/at|kcal\/mol|kcalmol-1|j\/mol|jmol-1|ev))\b[\)\]\}]?'
}


def find_unit_further(unit, prop_name, heading):
    property_type = prop_type_mapping.get(prop_name, '')
    pattern = patterns.get(property_type, '')
    if not pattern:
        return None
    
    # Search for matches using the refined regex pattern
    matches = re.findall(pattern, heading)
    extracted_unit = matches[-1].strip() if matches else None
    
    # Remove extra spaces from the extracted unit if any
    if extracted_unit:
        extracted_unit = re.sub(r'\s+', '', extracted_unit)
    
    return extracted_unit


def normalize_unit(unit, prop_name):
    property_type = prop_type_mapping.get(prop_name, '')
    
    if property_type == '':
        raise RuntimeError('Uncovered property')
    
    if unit is None:
        return None if property_type == 'refractive_index' else ''
    
    
    unit = unit.lower()  
    unit = unit.replace('·', '').replace('–', '-').replace('−', '-')  # Replace special characters
    
    
    normalizations = {
        'density': {
            'g/cm3': ['g/cm3', 'g/cm³', 'gcm-3', 'gcm−3', 'g.cm-3', 'g.cm−3', 'g/cc', 'g/ml', 'gml-1', 'gm/cm3', 'gm/cc', 'gw/cm3', 'gmcc-1'],
            'mg/m3': ['mg/m3', 'mg/m³', 'mgm-3', 'mgm−3', 'mg/m3'],
            'kg/m3': ['kg/m3', 'kg/m³', 'kgm-3', 'kgm−3', 'g/l'],
            'g/cm': ['g/cm', 'g/cm-1', 'gcm-1', 'gcm−1'],
            'lb/in3': ['lb/in3', 'lbin-3']
        },
        'temperature': {
            'degC': ['c', 'degc', '°c'],
            'K': ['k', 'kelvin']
        },
        'youngs_modulus': {
            'GPa': ['gpa'],
            'MPa': ['mpa'],
            'Pa': ['pa'],
            'psi': ['psi'],
            'dyn/cm2': ['dyn/cm2', 'dyncm−2', 'dyncm-2', 'dyn/cm2', 'dyn cm2'],
            'kb': ['kb']
        },
        'vickers_hardness': {
            'kgf/mm2': ['Kgf/mm2', 'kgfmm-2', 'Kgf/mm2', 'kgf/mm2'],
            'kg/mm2': ['kg/mm2', 'kgmm-2'],
            'GPa': ['gpa'],
            'MPa': ['mpa'],
            'Pa': ['pa'],
            'psi': ['psi']
        },
        'fracture_toughness': {
            'MPam1/2': ['mpam', 'mpam1/2', 'mpa/sqrt(m)', 'mpa(m)1/2']
        },
        'electric_conductivity': {
            'S/cm': ['s/cm', 'scm-1', 'ohm-1cm-1', '(o-cm)-1'],
            'S/m': ['s/m', 'sm-1', 'ohm-1m-1', '(om)-1'],
            '(MO m)-1': ['(mom)-1']
        },
        'thermal_expansion': {
            'degC-1': ['degc-1', 'c-1', '/degc', '/c'],
            'K-1': ['k-1', '/k']
        },
        'activation_energy': {
            'kJ/mol': ['kj/mol', 'kjmol-1', 'kj/molk'],
            'eV/at': ['ev/at'],
            'kcal/mol': ['kcal/mol', 'kcalmol-1'],
            'J/mol': ['j/mol', 'jmol-1'],
            'eV': ['ev']
        }
    }
    
    # Normalize the unit based on variations
    for normalized, variations in normalizations.get(property_type, {}).items():
        if unit in variations:
            return normalized
    
    return unit
