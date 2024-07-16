import copy
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

from rdkit.Chem import AllChem

from .property_utils import cache_prop_func_dict, check_monotonic


def step_03_evaluate_manipuated_data(manipulation_folder, prop_list, molecule_count_threshold_list, non_monotonic_ratio_threshold_list, args, check_monotonic_function=check_monotonic, verbose=True, fp_radius=2, fp_nbits=1024, count_invalid=1):
    prop_func_dict = cache_prop_func_dict()
    selected_prop_func_dict = {}
    for prop_name in prop_list:
        selected_prop_func_dict[prop_name] = prop_func_dict[prop_name]
    base_dir = os.path.join(manipulation_folder, "manipulated_data")

    # pre-process
    data_and_step_idx2smiles, data_and_step_idx2mol = {}, {}
    data_and_step_idx2raw_fp, data_and_step_idx2fp = {}, {}
    valid_direction_idx_list = []
    for direction_idx in range(args.num_directions):
        direction_path = os.path.join(base_dir, "direction_{}.csv".format(direction_idx))
        if not os.path.exists(direction_path):
            continue
        valid_direction_idx_list.append(direction_idx)
        
        # convert from csv file into smiles/molecule/fingerprint list
        df = pd.read_csv(direction_path)
        data_idx_list, step_idx_list, smiles_list = df["data_idx"].tolist(), df["step_idx"].tolist(), df["smiles"].tolist()
        for data_idx, step_idx, smiles in zip(data_idx_list, step_idx_list, smiles_list):
            data_and_step_idx2smiles[(direction_idx, data_idx, step_idx)] = smiles
            mol = Chem.MolFromSmiles(smiles)
            data_and_step_idx2mol[(direction_idx, data_idx, step_idx)] = mol

            try:
                '''credit to https://github.com/mims-harvard/TDC/blob/main/tdc/chem_utils/oracle/oracle.py#L55'''
                mol_copy = copy.copy(mol)
                Chem.SanitizeMol(mol_copy)
                raw_fp = AllChem.GetMorganFingerprintAsBitVect(mol_copy, fp_radius, nBits=fp_nbits)
                fp = np.array(list(raw_fp.ToBitString())).astype(float).reshape(1, -1)
            except:
                raw_fp = None
                fp = None
            data_and_step_idx2raw_fp[(direction_idx, data_idx, step_idx)] = raw_fp
            data_and_step_idx2fp[(direction_idx, data_idx, step_idx)] = fp

    if verbose:
        print("valid direction index {}".format(valid_direction_idx_list))

    # evaluate
    summary = defaultdict(int)
    valid_property_summary = {}
    for prop_name, func in selected_prop_func_dict.items():
        for direction_idx in valid_direction_idx_list:
            invalid_count = 0

            for data_idx in range(args.num_manipulation):
                temp_mol_list, temp_smiles_list, temp_fp_list, temp_raw_fp_list, temp_prop_list = [], [], [], [], []
                temp_step_list = []

                ##### need to check if the center point (step-size=0) is valid #####
                oracle_center_step_idx = args.alpha_split_num // 2
                assert oracle_center_step_idx * 2 + 1 == args.alpha_split_num
                center_mol = data_and_step_idx2mol[(direction_idx, data_idx, oracle_center_step_idx)]
                if center_mol is None:
                    invalid_count += 1
                    continue

                for step_idx in range(args.alpha_split_num):
                    smiles = data_and_step_idx2smiles[(direction_idx, data_idx, step_idx)]
                    mol = data_and_step_idx2mol[(direction_idx, data_idx, step_idx)]
                    fp = data_and_step_idx2fp[(direction_idx, data_idx, step_idx)]
                    raw_fp = data_and_step_idx2raw_fp[(direction_idx, data_idx, step_idx)]

                    try:
                        assert mol is not None
                        assert fp is not None
                        temp_mol_list.append(mol)
                        temp_fp_list.append(fp)
                        temp_smiles_list.append(smiles)
                        temp_step_list.append(step_idx)
                        temp_raw_fp_list.append(raw_fp)
                        if step_idx == oracle_center_step_idx:
                            center_step_idx = len(temp_raw_fp_list) - 1
                    except:
                        pass

                valid_length = len(temp_mol_list)
                if prop_name in ['drd2', 'jnk3', 'gsk3b'] and valid_length > 0:
                    temp_fp_list = np.array(temp_fp_list).squeeze(axis=1)
                    temp_prop_list = func(temp_fp_list)
                elif prop_name in ['sa']:
                    temp_prop_list = [func(x) for x in temp_smiles_list]
                elif prop_name in ['tanimoto']:
                    temp_prop_list = func(temp_raw_fp_list, center_step_idx)
                else:
                    temp_prop_list = [func(x) for x in temp_mol_list]
                    # print("data {}, direction {}, {}".format(data_idx, direction_idx, temp_prop_list))

                for molecule_count_threshold in molecule_count_threshold_list:
                    for non_monotonic_ratio_threshold in non_monotonic_ratio_threshold_list:
                        if check_monotonic_function(temp_prop_list, molecule_count_threshold, non_monotonic_ratio_threshold, temp_smiles_list):
                            key_ = (prop_name, direction_idx, molecule_count_threshold, non_monotonic_ratio_threshold)
                            summary[key_] += 1

                            step_idx2prop = {}
                            for step_idx, prop_value in zip(temp_step_list, temp_prop_list):
                                step_idx2prop[step_idx] = prop_value
                            valid_key_ = (prop_name, direction_idx, data_idx, molecule_count_threshold, non_monotonic_ratio_threshold)
                            valid_property_summary[valid_key_] = step_idx2prop

            if verbose:
                print("invalid count: {}".format(invalid_count))
            for molecule_count_threshold in molecule_count_threshold_list:
                for non_monotonic_ratio_threshold in non_monotonic_ratio_threshold_list:
                    key_ = (prop_name, direction_idx, molecule_count_threshold, non_monotonic_ratio_threshold)
                    if args.num_manipulation - invalid_count == 0:
                        summary[key_] = 0
                    else:
                        if count_invalid == 1:
                            summary[key_] = summary[key_] * 100. / (args.num_manipulation - invalid_count)
                        else:
                            summary[key_] = summary[key_] * 100. / args.num_manipulation
                    if verbose:
                        print("({}, {})\t{}\tdirection-{}\tsuccess ratio: {}".format(
                            molecule_count_threshold, non_monotonic_ratio_threshold, prop_name, direction_idx, summary[key_]))

    if verbose:
        print(summary)

        row = 'prop'
        for prop_name in prop_list:
            row = '{} & {}'.format(row, prop_name)
        row = '{}\\\\'.format(row)
        print(row)

        for molecule_count_threshold in molecule_count_threshold_list:
            for non_monotonic_ratio_threshold in non_monotonic_ratio_threshold_list:
                row = 'result {} {:.2f}'.format(molecule_count_threshold, non_monotonic_ratio_threshold)
                for prop_name in prop_list:
                    value_list = []
                    for direction_idx in valid_direction_idx_list:
                        key_ = (prop_name, direction_idx, molecule_count_threshold, non_monotonic_ratio_threshold)
                        value_list.append(summary[key_])
                    row = '{} & {:.1f}'.format(row, np.max(value_list))
                row = '{}\\\\'.format(row)
                print(row)
    return (summary, valid_direction_idx_list), valid_property_summary