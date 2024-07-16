import argparse
import os
import sys

from distutils.util import strtobool

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Draw
from descriptastorus.descriptors import rdDescriptors

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

import functools
import time

from rdkit.Chem import Descriptors
from tqdm import tqdm

from data import transform_qm9, transform_zinc250k, transform_chembl
from data.data_loader import NumpyTupleDataset
from data.transform_zinc250k import (transform_fn_zinc250k, zinc250_atomic_num_list)
from data.transform_chembl import (chembl_atomic_num_list)

import mflow
from mflow.models.hyperparams import Hyperparameters
from mflow.models.model import MoFlow, rescale_adj
from mflow.models.utils import (_to_numpy_array, adj_to_smiles, check_novelty,
                                check_validity, construct_mol, correct_mol,
                                valid_mol, valid_mol_can_with_seg)
from mflow.utils.model_utils import get_latent_vec, load_model
from mflow.utils.timereport import TimeReport

from GraphCG.disentanglement_utils import (do_Beta_VAE, do_Factor_VAE, do_MIG, do_DCI, do_Modularity, do_SAP)

RDKIT_fragments = [
    'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN',
    'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2',
    'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0',
    'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
    'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide',
    'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl',
    'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine',
    'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester',
    'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',
    'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone',
    'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine',
    'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho',
    'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol',
    'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine',
    'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN',
    'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole',
    'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']


def generate_molecules_from_reconstruction(model, train_dataset, batch_size, n_sample):
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    adj_list, x_list, z_list = [], [], []

    count = 0
    for i, batch in enumerate(train_dataloader):
        x = batch[0].to(device)  # (256, 9, 5)
        adj = batch[1].to(device)  # (256, 4, 9, 9)
        adj_normalized = rescale_adj(adj).to(device)
        z, sum_log_det_jacs = model(adj, x, adj_normalized)
        z0 = z[0].reshape(z[0].shape[0], -1)
        z1 = z[1].reshape(z[1].shape[0], -1)
        z = torch.cat([z0, z1], dim=1)

        adj_rev, x_rev = model.reverse(z)

        count += x.size()[0]

        adj_list.append(adj_rev)
        x_list.append(x_rev)
        z_list.append(z)

        if count >= n_sample:
            break

    adj_list = torch.cat(adj_list)[:n_sample]
    x_list = torch.cat(x_list)[:n_sample]
    z_list = torch.cat(z_list)[:n_sample]

    return adj_list, x_list, z_list


def generate_molecules_from_random(model, temp=0.7, z_mu=None, batch_size=20, true_adj=None, device=-1):
    if isinstance(device, torch.device):
        pass
    elif isinstance(device, int):
        if device >= 0:
            # device = args.gpu
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu", int(device))
        else:
            device = torch.device("cpu")
    else:
        raise ValueError("only 'torch.device' or 'int' are valid for 'device', but '%s' is "'given' % str(device))

    z_dim = model.b_size + model.a_size  # 324 + 45 = 369
    mu = np.zeros(z_dim)  # (369,)
    sigma_diag = np.ones(z_dim)  # (369,)

    if model.hyper_params.learn_dist:
        if len(model.ln_var) == 1:
            sigma_diag = np.sqrt(np.exp(model.ln_var.item())) * sigma_diag
        elif len(model.ln_var) == 2:
            sigma_diag[:model.b_size] = np.sqrt(np.exp(model.ln_var[0].item())) * sigma_diag[:model.b_size]
            sigma_diag[model.b_size+1:] = np.sqrt(np.exp(model.ln_var[1].item())) * sigma_diag[model.b_size+1:]

    sigma = temp * sigma_diag

    with torch.no_grad():
        if z_mu is not None:
            mu = z_mu
            sigma = 0.01 * np.eye(z_dim)
        z = np.random.normal(mu, sigma, (batch_size, z_dim))  # .astype(np.float32)
        z = torch.from_numpy(z).float().to(device)
        adj, x = model.reverse(z, true_adj=true_adj)

    return adj, x, z  # (bs, 4, 9, 9), (bs, 9, 5), (bs, 369)


def filter(latent_list, smiles_list, label_generator):
    neo_latent_list, neo_smiles_list, labels_list = [], [], []
    for latent, smiles in zip(latent_list, smiles_list):
        if smiles is None:
            print("None")
            continue
        neo_latent_list.append(latent)
        neo_smiles_list.append(smiles)
        labels = label_generator.process(smiles)[1:]
        labels_list.append(labels)

    neo_latent_list = np.array(neo_latent_list)
    labels_list = np.array(labels_list)
    labels_list[labels_list != 0] = 1
    return neo_latent_list, neo_smiles_list, labels_list


def check_disentanglement():
    generated_latent_list, generated_mols_list = [], []
    for i in range(args.n_experiments):
        if args.codes_option == "random":
            print("Generate latent with random distribution.")
            adj, x, z = generate_molecules_from_random(model, batch_size=args.n_sample, true_adj=None, temp=args.temperature, device=device)
        elif args.codes_option == "reconstruction":
            print("Generate latent with reconstruction.")
            adj, x, z  = generate_molecules_from_reconstruction(model, train_dataset, batch_size=args.batch_size, n_sample=args.n_sample)

        adj, x, z = _to_numpy_array(adj), _to_numpy_array(x), _to_numpy_array(z)
        mols = [valid_mol(construct_mol(x_elem, adj_elem, atomic_num_list)) for x_elem, adj_elem in zip(x, adj)]
        
        generated_latent_list.extend(z)
        generated_mols_list.extend(mols)

    generated_latent_list = np.array(generated_latent_list)
    smiles_list = [Chem.MolToSmiles(mol, canonical=True) for mol in generated_mols_list]
    print(smiles_list[:10])

    label_generator = rdDescriptors.RDKit2D(RDKIT_fragments)
    generated_latent_list, smiles_list, labels_list = filter(generated_latent_list, smiles_list, label_generator)

    n_mol = labels_list.shape[0]
    n_factor = labels_list.shape[1]
    n_train = int(0.8 * n_mol)
    label_pos_count = np.sum(labels_list, axis=0)
    print("labels_pos_count: {}\n\n\n".format(label_pos_count.shape))

    for label_idx in range(n_factor):
        print("label: {}\tfragment: {}\tcount: {}".format(label_idx, RDKIT_fragments[label_idx], label_pos_count[label_idx]))
    print('\n\n\n')

    selected_index = [1, 2, 10, 11, 15, 16, 41]
    label_list = labels_list[:, selected_index]
    print("after selection: {}\t{}".format(label_list.shape, np.sum(label_list, axis=0)))

    beta_vae_measure = do_Beta_VAE(
        generated_latent_list, label_list, n_train, batch_size=16, random_state=args.random_state, verbose=args.verbose)
    print("Beta VAE: {}\n\n\n".format(beta_vae_measure))

    factor_vae_measure = do_Factor_VAE(
        generated_latent_list, label_list, n_train, batch_size=16, random_state=args.random_state, verbose=args.verbose)
    print("Factor VAE: {}\n\n\n".format(factor_vae_measure))

    mig_measure = do_MIG(generated_latent_list, label_list, n_train)
    print("MIG: {}\n\n\n".format(mig_measure))

    dci_measure = do_DCI(generated_latent_list, label_list, n_train, verbose=args.verbose)
    print("DCI: {}\n\n\n".format(dci_measure))

    modularity_measure = do_Modularity(generated_latent_list, label_list, n_train)
    print("Modularity: {}\n\n\n".format(modularity_measure))

    sap_measure = do_SAP(generated_latent_list, label_list, n_train)
    print("SAP: {}\n\n\n".format(sap_measure))

    print("{:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}".format(
        beta_vae_measure["eval_accuracy"], factor_vae_measure["eval_accuracy"], mig_measure["val"]["discrete_mig"],
        dci_measure["disentanglement"], modularity_measure["modularity_score"], sap_measure["SAP_score"]
    ))
    return


if __name__ == "__main__":
    #################### for argumenets ####################
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./results")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--data_name", type=str, default="qm9", choices=["qm9", "zinc250k", "chembl"], help="dataset name")
    parser.add_argument("--snapshot-path", "-snapshot", type=str, required=True)
    parser.add_argument("--hyperparams-path", type=str, default="moflow-params.json", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature of the gaussian distribution")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--random_state", type=int, default=123)

    # step 1
    parser.add_argument("--n_experiments", type=int, default=1, help="number of times generation to be run")
    parser.add_argument("--correct_validity", type=strtobool, default="true", help="if apply validity correction after the generation")
    parser.add_argument("--codes_option", type=str, default="reconstruction", choices=["random", "reconstruction"])
    parser.add_argument("--n_sample", type=int, default=100)

    args = parser.parse_args()
    #################### for argumenets ####################
    np.random.seed(args.random_state)

    start = time.time()
    print("Start at Time: {}".format(time.ctime()))
    snapshot_path = os.path.join(args.model_dir, args.snapshot_path)
    hyperparams_path = os.path.join(args.model_dir, args.hyperparams_path)
    print("Loading model from {}".format(snapshot_path))
    print("Loading hyperparamaters from {}".format(hyperparams_path))

    manipulation_folder = os.path.join("{}_manipulation".format(args.model_dir))
    os.makedirs(manipulation_folder, exist_ok=True)
    manipulation_generated_latent_path = os.path.join(manipulation_folder, "generated_latent_z.npy")
    manipulation_direction_path = os.path.join(manipulation_folder, "direction_list.npy")

    model_params = Hyperparameters(path=hyperparams_path)
    model = load_model(snapshot_path, model_params, debug=True)
    if len(model.ln_var) == 1:
        print("model.ln_var: {:.2f}".format(model.ln_var.item()))
    elif len(model.ln_var) == 2:
        print("model.ln_var[0]: {:.2f}, model.ln_var[1]: {:.2f}".format(model.ln_var[0].item(), model.ln_var[1].item()))

    if args.gpu >= 0:
        device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()

    if args.data_name == "qm9":
        atomic_num_list = [6, 7, 8, 9, 0]
        transform_fn = transform_qm9.transform_fn
        valid_idx = transform_qm9.get_val_ids(file_path="./data/valid_idx_qm9.json")
        molecule_file = "qm9_relgcn_kekulized_ggnp.npz"
    elif args.data_name == "zinc250k":
        atomic_num_list = zinc250_atomic_num_list
        transform_fn = transform_zinc250k.transform_fn_zinc250k
        valid_idx = transform_zinc250k.get_val_ids(file_path="./data/valid_idx_zinc.json")
        molecule_file = "zinc250k_relgcn_kekulized_ggnp.npz"
    elif args.data_name == "chembl":
        atomic_num_list = chembl_atomic_num_list
        transform_fn = transform_chembl.transform_fn_chembl
        valid_idx = transform_chembl.get_val_ids(file_path="./data/valid_idx_chembl.json")
        molecule_file = "chembl_relgcn_kekulized_ggnp.npz"

    dataset = NumpyTupleDataset.load(os.path.join(args.data_dir, molecule_file), transform=transform_fn)

    assert len(valid_idx) > 0
    train_idx = [t for t in range(len(dataset)) if t not in valid_idx]
    n_train = len(train_idx)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, valid_idx)
    print("{} in total, {}  training data, {}  testing data".format(len(dataset), len(train_dataset), len(test_dataset)))

    train_x = [a[0] for a in train_dataset]
    train_adj = [a[1] for a in train_dataset]
    train_smiles = adj_to_smiles(train_adj, train_x, atomic_num_list)
    print("Loading trained model and data done! Time {:.2f} seconds".format(time.time() - start))

    check_disentanglement()
