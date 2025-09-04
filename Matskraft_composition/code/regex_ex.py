from regex_lib import parse_composition
import ast
import operator as op
import os
import pathlib
import pickle
import re

elements_compounds_path = os.path.join(pathlib.Path(__file__).parent.resolve(), '../data/elements_compounds.pkl')
elements_compounds = pickle.load(open(elements_compounds_path, 'rb'))
all_elements, all_compounds = elements_compounds['elements'], elements_compounds['compounds']

operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv}

elem_l = all_elements
elem = r'(' + r'|'.join(elem_l) + r')'
#print(elem)
num = r'(\d*\.\d+|0|[1-9]\d*)'
comp_vars = ['x', 'y', 'z', 'X']
var = r'(' + r'|'.join(comp_vars) + r')'
#print(var)

regex_1a = r'(' + elem + r'\s*' + var + ')'
regex_2a = r'(' + elem + num + r')'
regex_2b = r'([\(\[\{](' + elem + num + r'\s*)+[\)\}\]]' + num + r')'
regex_2c = r'(' + num + r'[+-]' + var + r')'
regex = r'(\s*' + regex_1a + r'\s*([\(\[\{]' + regex_2a + r'\s*' + regex_2b + r'\s*[\)\}\]]' + regex_2c + r'))'


def extr_comp(cap):
    material_list = []
    regg = re.compile(regex)
    #print(regg.findall(cap))
    for re_match in regg.findall(cap):
#         print(re_match)
        st = cap.index(re_match[0])
        end = st + len(re_match[0])
        #print(st, end)
        if len(re_match) >= 10:
            element_list = []
            extr_comp = re_match[0]
            total = '(' + re_match[3] + '+' + re_match[-3] + ')'
            first_elem, first_val = re_match[2], '(' + re_match[3] + ')' + '/' + total 
            second_whole_elem, whole_val = re_match[4], '(' + re_match[-3] + ')' + '/' + total
            #print(first_elem, first_val, second_whole_elem, whole_val)
            sec_third_value_den = '(' + re_match[7] + '+' + re_match[-4] + ')'
            second_elem, second_val = re_match[6], '(' + '(' + re_match[7] + ')' + '/' + sec_third_value_den + ')' + '*' + '(' + re_match[-3] + ')'
            #print(second_elem, second_val)
            element_list.append((first_elem, first_val))
            element_list.append((second_elem, second_val))
            third_multiple_comp_val = re_match[8]
            mul_elem = []
            reg_elem_num = elem + num
            for m in re.finditer(reg_elem_num, third_multiple_comp_val):
                for ele in elem_l:
                    #print(m.group(1))
                    if ele == m.group(1):
                        mul_elem.append((ele, m.group(2)))
            #print(mul_elem)
            tot_m_v = ''
            for m in mul_elem:
                eleme, valu = m[0],m[1]
                tot_m_v += valu + '+'
            tot_m_v = tot_m_v[:-1]
            #print(tot_m_v)
            upd_mul_elem = []
            for i,m in enumerate(mul_elem):
                cons_f = '(' + '(' + re_match[-4] + ')' + '/' + sec_third_value_den + ')' + '*' + '(' + re_match[-3] + ')'
                new_val = '(' + '(' + m[1] + ')' + '/' '(' + tot_m_v + ')' + ')' + '*' + cons_f
                upd_mul_elem.append((m[0], new_val))
                element_list.append((m[0], new_val))
            #print(element_list)
            if len(element_list)>0:
                material_list.append((element_list, (st, end)))
    return material_list