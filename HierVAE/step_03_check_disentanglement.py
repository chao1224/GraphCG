import os

import torch
from torch.utils.data import DataLoader, Dataset

import random, sys
import numpy as np
import argparse

import rdkit
from rdkit import Chem
from descriptastorus.descriptors import rdDescriptors

import sys
sys.path.insert(0, 'hgraph2graph')
from GraphCG.disentanglement_utils import (do_Beta_VAE, do_Factor_VAE, do_MIG, do_DCI, do_Modularity, do_SAP)

from hgraph import *
from hgraph import MolGraph


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


class MoleculeDataset(Dataset):
    def __init__(self, data, vocab, avocab, batch_size):
        self.batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
        self.vocab = vocab
        self.avocab = avocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return MolGraph.tensorize(self.batches[idx], self.vocab, self.avocab)


def check_validity(generated_all_smiles):
    count = 0
    for sm in generated_all_smiles:
        mol = Chem.MolFromSmiles(sm)
        if mol is not None:
            count += 1
    return count


def check_unique(generated_all_smiles):
    return len(set(generated_all_smiles))


def check_novelty(generated_all_smiles, train_smiles):
    new_molecules = 0
    for sm in generated_all_smiles:
        if sm not in train_smiles:
            new_molecules += 1
    return new_molecules


def generate_molecules_from_random():
    # TODO: not used
    z = torch.randn(args.n_sample, args.latent_size)
    return z


def generate_molecules_from_reconstruction():
    dataset = MoleculeDataset(train_smiles, args.vocab, args.atom_vocab, args.batch_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0])

    smiles_list = []
    count, limit = 0, args.n_sample
    latent_repr = []
    for i, batch in enumerate(loader):
        original_smiles = train_smiles[args.batch_size * i : args.batch_size * (i + 1)]
        reconstructured_smiles, latent_ = model.reconstruct(batch, return_latent=True)

        smiles_list.extend(reconstructured_smiles)
        latent_repr.append(latent_.detach().cpu())

        count += len(original_smiles)
        if count >= limit:
            break

    smiles_list = smiles_list[:args.n_sample]
    latent_repr = torch.cat(latent_repr, dim=0).detach().numpy()[:args.n_sample]

    return smiles_list, latent_repr


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
    if args.codes_option == "random":
        print("Generate latent with random distribution.")
        smiles_list, generated_latent_list = model.sample(args.n_sample, greedy=True)
        generated_latent_list = generated_latent_list.cpu().numpy()
        generated_latent_list = np.array(generated_latent_list)
    elif args.codes_option == "reconstruction":
        smiles_list, generated_latent_list = generate_molecules_from_reconstruction()

    generated_latent_list = np.array(generated_latent_list)
    print(smiles_list[:10])

    label_generator = rdDescriptors.RDKit2D(RDKIT_fragments)
    generated_latent_list, smiles_list, labels_list = filter(generated_latent_list, smiles_list, label_generator)

    n_mol = labels_list.shape[0]
    n_factor = labels_list.shape[1]
    n_train = int(0.8 * n_mol)
    label_pos_count = np.sum(labels_list, axis=0)
    print("labels_pos_count: {}\n\n\n".format(label_pos_count.shape))
    for i in range(labels_list.shape[1]):
        print(i, set(list(labels_list[:, i])))

    for label_idx in range(n_factor):
        print("label: {}\tfragment: {}\tcount: {}".format(label_idx, RDKIT_fragments[label_idx], label_pos_count[label_idx]))
    print('\n\n\n')

    selected_index = [1, 2, 5, 6, 15, 16, 17, 23, 27, 41, 57, 78]
    selected_index = [1, 2, 7, 10, 11, 15, 16, 27, 41, 44, 83]
    selected_index = [1, 2, 10, 11, 15, 16, 41]
    label_list = labels_list[:, selected_index]
    print("after selection: {}\t{}".format(label_list.shape, np.sum(label_list, axis=0)))
    for label_idx in selected_index:
        print("label: {}\tfragment: {}\tcount: {}".format(label_idx, RDKIT_fragments[label_idx], label_pos_count[label_idx]))
    print('\n\n\n')
    print('\n\n\n')

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


if __name__ == '__main__':
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--atom_vocab', default=common_atom_vocab)
    parser.add_argument('--model', required=True)
    parser.add_argument('--train', type=str, default=None)

    parser.add_argument('--seed', type=int, default=7)

    parser.add_argument('--rnn_type', type=str, default='LSTM')
    parser.add_argument('--hidden_size', type=int, default=250)
    parser.add_argument('--embed_size', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--latent_size', type=int, default=32)
    parser.add_argument('--depthT', type=int, default=15)
    parser.add_argument('--depthG', type=int, default=15)
    parser.add_argument('--diterT', type=int, default=1)
    parser.add_argument('--diterG', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument("--model_dir", type=str, default="./results")
    parser.add_argument("--data_name", type=str, default="qm9", choices=["qm9", "zinc250k"], help="dataset name")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature of the gaussian distribution")

    parser.add_argument("--codes_option", type=str, default="random", choices=["random", "reconstruction"])
    parser.add_argument("--n_sample", type=int, default=100)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--random_state", type=int, default=123)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
    args.vocab = PairVocab(vocab)

    # with open('./hgraph2graph/data/{}/all.txt'.format(args.dataset)) as f:
        # train_smiles = [line.strip("\r\n ") for line in f] 
    # train_smiles = [line.strip("\r\n ") for line in open('./hgraph2graph/data/{}/all.txt'.format(args.dataset))]
    train_smiles = [line.strip("\r\n ") for line in open(args.train)]

    model = HierVAE(args).cuda()

    model.load_state_dict(torch.load(args.model)[0])
    print('Done loading from {}.'.format(args.model))
    model.eval()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    manipulation_folder = os.path.join("{}_manipulation".format(args.model_dir))
    os.makedirs(manipulation_folder, exist_ok=True)
    manipulation_generated_latent_path = os.path.join(manipulation_folder, "generated_latent_z.npy")
    manipulation_direction_path = os.path.join(manipulation_folder, "direction_list.npy")

    ########## step 01 ##########
    check_disentanglement()
