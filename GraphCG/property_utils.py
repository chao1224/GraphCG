import numpy as np
from tdc import Oracle
from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs
import pickle


SA_scorer = Oracle(name = 'SA')
DRD2_scorer_oracle = Oracle(name = 'DRD2')
JNK3_scorer_oracle = Oracle(name = 'JNK3')
GSK3B_scorer_oracle = Oracle(name = 'GSK3B')


def check_SA(smiles):
    score = SA_scorer(smiles)
    return score

class drd2:
    def __init__(self):
        drd2_model_path = 'oracle/drd2.pkl'
        try:
            with open(drd2_model_path, 'rb') as f:
                self.drd2_model = pickle.load(f)
                print(self.drd2_model)
        except EOFError:
            import sys
            sys.exit("TDC is hosted in Harvard Dataverse and it is currently under maintenance, please check back in a few hours or checkout https://dataverse.harvard.edu/.")
        return
    
    def __call__(self, fp):
        drd_score = self.drd2_model.predict_proba(fp)[:, 1]
        return drd_score


class jnk3:
    def __init__(self):
        jnk3_model_path = 'oracle/jnk3.pkl'
        try:
            with open(jnk3_model_path, 'rb') as f:
                self.jnk3_model = pickle.load(f)
                print(self.jnk3_model)
        except EOFError:
            import sys
            sys.exit("TDC is hosted in Harvard Dataverse and it is currently under maintenance, please check back in a few hours or checkout https://dataverse.harvard.edu/.")
        return
          
    def __call__(self, fp):
        jnk3_score = self.jnk3_model.predict_proba(fp)[:, 1]
        return jnk3_score


class gsk3b:
    def __init__(self):
        gsk3_model_path = 'oracle/gsk3b.pkl'
        try:
            with open(gsk3_model_path, 'rb') as f:
                self.gsk3_model = pickle.load(f)
                print(self.gsk3_model)
        except EOFError:
            import sys
            sys.exit("TDC is hosted in Harvard Dataverse and it is currently under maintenance, please check back in a few hours or checkout https://dataverse.harvard.edu/.")
        return

    def __call__(self, fp):
        gsk3_score = self.gsk3_model.predict_proba(fp)[:, 1]
        return gsk3_score


DRD2_scorer = drd2()
JNK3_scorer = jnk3()
GSK3B_scorer = gsk3b()


def check_DRD2(fp_array):
    score = DRD2_scorer(fp_array)
    return score


def check_JNK3(fp_array):
    score = JNK3_scorer(fp_array)
    return score


def check_GSK3B(fp_array):
    score = GSK3B_scorer(fp_array)
    return score


def check_tanimoto(fp_array, original_index):
    L = len(fp_array)
    original_fp = fp_array[original_index]

    tanimoto_similarity_array = []
    for i in range(L):
        edited_fp = fp_array[i]
        if edited_fp is None:
            continue
        sim = DataStructs.TanimotoSimilarity(edited_fp, original_fp)
        if i == original_index:
            assert sim == 1
        elif i > original_index:
            sim = 2 - sim
        tanimoto_similarity_array.append(sim)
    return tanimoto_similarity_array


def get_fragment_count(mol, prop_name_list):
    prop_func_dict = {}
    for prop_name, function in Descriptors.descList:
        if prop_name in prop_name_list:
            prop_func_dict[prop_name] = function

    eval_value = 0
    for prop_name, function in prop_func_dict.items():
        eval_value += function(mol)
    return eval_value


def check_fr_Al_Ar_OH(mol):
    prop_name_list = ["fr_Al_OH", "fr_Ar_OH",]
    return get_fragment_count(mol, prop_name_list)


def cache_prop_func_dict():
    # This is to use all ~200 properties
    prop_func_dict = {}
    for prop_name, function in Descriptors.descList:
        prop_func_dict[prop_name] = function
        # if "fr_" in prop_name:
        #     print("\"{}\",".format(prop_name))

    prop_func_dict['sa']        = check_SA
    prop_func_dict['drd2']      = check_DRD2
    prop_func_dict['jnk3']      = check_JNK3
    prop_func_dict['gsk3b']     = check_GSK3B
    prop_func_dict['tanimoto']  = check_tanimoto

    prop_func_dict['fr_Al_Ar_OH'] = check_fr_Al_Ar_OH
    return prop_func_dict
    

def check_monotonic(value_list, molecule_count_threshold, non_monotonic_ratio_threshold, temp_smiles_list):
    temp_smiles_set = set(temp_smiles_list)
    if len(temp_smiles_set) < molecule_count_threshold:
        return False

    # clean-up value_list
    neo_value_list, prev_value = [], -1
    for value in value_list:
        if value == prev_value:
            continue
        neo_value_list.append(value)
        prev_value = value
    value_list = neo_value_list
    value_len = len(value_list)
    non_monotonic_count_threshold = value_len * non_monotonic_ratio_threshold

    # first test increasing
    current_non_monotonic_count = 0
    for i in range(1, value_len):
        if value_list[i-1] <= value_list[i]:
            continue
        else:
            current_non_monotonic_count += 1
    if current_non_monotonic_count <= non_monotonic_count_threshold:
        return True

    # first test decreasing
    current_non_monotonic_count = 0
    for i in range(1, value_len):
        if value_list[i-1] >= value_list[i]:
            continue
        else:
            current_non_monotonic_count += 1
    if current_non_monotonic_count <= non_monotonic_count_threshold:
        return True

    return False


def check_strict_monotonic(value_list, molecule_count_threshold, non_monotonic_ratio_threshold, temp_smiles_list):
    is_monotonic = check_monotonic(value_list, molecule_count_threshold, non_monotonic_ratio_threshold, temp_smiles_list)
    count = len(set(value_list))
    return is_monotonic and count > 1


if __name__ == "__main__":
    for prop_name, function in Descriptors.descList:
        print(prop_name)