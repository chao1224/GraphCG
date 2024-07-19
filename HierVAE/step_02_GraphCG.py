import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rdkit import Chem
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, 'hgraph2graph')

import rdkit
from GraphCG import (Direction_Embedding_01, Direction_Embedding_02,
                     Direction_Embedding_03, Direction_Embedding_04,
                     GraphCG_editing_01, GraphCG_editing_02)
from GraphCG.molecule_utils import step_03_evaluate_manipuated_data
from hgraph import *
from hgraph import MolGraph


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


def generate_molecules_from_reconstruction():
    dataset = MoleculeDataset(train_smiles, args.vocab, args.atom_vocab, args.batch_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0])

    smiles_list = []
    count, limit = 0, args.num_sample
    latent_repr = []
    for i, batch in enumerate(loader):
        original_smiles = train_smiles[args.batch_size * i : args.batch_size * (i + 1)]
        reconstructured_smiles, latent_ = model.reconstruct(batch, return_latent=True)

        smiles_list.extend(reconstructured_smiles)
        latent_repr.append(latent_.detach().cpu())

        count += len(original_smiles)
        if count >= limit:
            break

    smiles_list = smiles_list[:args.num_sample]
    latent_repr = torch.cat(latent_repr, dim=0).detach().numpy()[:args.num_sample]

    return smiles_list, latent_repr


def step_01_generate_samples():
    start = time.time()

    if args.codes_option == "random":
        print("Generate latent with random distribution.")
        smiles_list, generated_latent = model.sample(args.num_sample, greedy=True)
        generated_latent = generated_latent.cpu().numpy()
        generated_latent = np.array(generated_latent)
    elif args.codes_option == "reconstruction":
        smiles_list, generated_latent = generate_molecules_from_reconstruction()

    print("Save generated latent to {}.".format(manipulation_generated_latent_path))
    np.save(manipulation_generated_latent_path, generated_latent)

    validity = check_validity(smiles_list)
    novelty = check_novelty(smiles_list, train_smiles)
    uniqueness = check_unique(smiles_list)
    print ('validity', validity, '/', args.num_sample)
    print ('novelty', novelty, '/', args.num_sample)
    print ('uniqueness', uniqueness, '/', args.num_sample)

    print("Task random generation done! Time {:.2f} seconds, Data: {}".format(time.time() - start, time.ctime()))
    return


def save_manipulation(
    embedding_function, args, model, z, direction_list, direction_basis_list, device, atomic_num_list=[6, 7, 8, 9, 0]):

    with torch.no_grad():
        alpha_list = np.linspace(-args.alpha, args.alpha, args.alpha_split_num)
        alpha_list = alpha_list.tolist()
        assert len(alpha_list) == args.alpha_split_num
        
        for direction_idx in range(args.num_directions):
            try:
                manipulated_data_folder = os.path.join(manipulation_folder, "manipulated_data")
                os.makedirs(manipulated_data_folder, exist_ok=True)
                output_file = os.path.join(manipulated_data_folder, "direction_{}.csv".format(direction_idx))

                data_idx_list, step_idx_list, smiles_list = [], [], []
                if args.verbose:
                    L = tqdm(range(args.num_manipulation))
                else:
                    L = range(args.num_manipulation)
                for data_idx in L:
                    z_manipulated_list = []

                    for step_idx in range(args.alpha_split_num):
                        if embedding_function is not None:
                            z_neo_manipulated = embedding_function.get_latent(z[data_idx:data_idx+1], direction_basis_list[direction_idx], alpha_list[step_idx])
                            z_neo_manipulated = z_neo_manipulated.squeeze()
                            z_manipulated_list.append(z_neo_manipulated)
                        else:
                            z_neo_manipulated = z[data_idx] + direction_list[direction_idx] * alpha_list[step_idx]
                            z_manipulated_list.append(z_neo_manipulated)

                    z_manipulated_list = torch.stack(z_manipulated_list, dim=0)
                    temp_smiles_list = model.sample(args.num_manipulation, greedy=True, direction=z_manipulated_list)

                    data_idx_list.extend([data_idx for _ in temp_smiles_list])
                    step_idx_list.extend([step_idx for step_idx in range(args.alpha_split_num)])
                    smiles_list.extend(temp_smiles_list)
                
                assert len(data_idx_list) == len(step_idx_list) == len(smiles_list)

                df = pd.DataFrame(data={"data_idx": data_idx_list, "step_idx": step_idx_list, "smiles": smiles_list})
                df.to_csv(output_file, index=False)
            except:
                print("Got exception in direction {}".format(direction_idx))
        return


def step_02_Random_SSL_saving():
    print("Loading from {}.".format(manipulation_generated_latent_path))
    training_z = np.load(manipulation_generated_latent_path, allow_pickle=True)

    z_space_dim = training_z.shape[1]    
    direction_list = np.random.normal(0, 1, (args.num_directions, z_space_dim))
    direction_list = torch.from_numpy(direction_list).to(device).float()
    direction_list = F.normalize(direction_list, dim=-1)

    #################### save manipulated molecules ####################
    mu = np.zeros(z_space_dim)
    sigma_diag = np.ones(z_space_dim)
    sigma_diag = args.temperature * sigma_diag

    z = np.random.normal(mu, sigma_diag, (args.num_manipulation, z_space_dim))
    z = torch.from_numpy(z).to(device).float()

    save_manipulation(
        model=model,
        embedding_function=None,
        direction_list=direction_list,
        direction_basis_list=None,
        z=z, device=device, args=args)
    return


def step_02_Variance_SSL_saving():
    print("Loading from {}.".format(manipulation_generated_latent_path))
    training_z = np.load(manipulation_generated_latent_path, allow_pickle=True)
    print("training_z", training_z.shape)

    z_space_dim = training_z.shape[1]
    direction_basis_list = torch.eye(z_space_dim)

    # calculate variance
    var_array = np.var(training_z, axis=0)
    assert len(var_array) == z_space_dim
    var_index = np.argsort(var_array)
    
    sampled_direction_idx = var_index[-args.num_directions:]
    
    print("sampled_direction_idx: ", sampled_direction_idx)

    #################### sample direction ####################
    print("\ndirection basis list: {}\n{}".format(direction_basis_list.size(), direction_basis_list))
    direction_list = []

    for i in sampled_direction_idx:
        direction = direction_basis_list[i]
        direction_list.append(direction.cpu().detach().numpy())
    direction_list = np.array(direction_list)
    direction_list = torch.from_numpy(direction_list).to(device).float()
    print("direction_list", direction_list.size())

    #################### save manipulated molecules ####################
    mu = np.zeros(z_space_dim)
    sigma_diag = np.ones(z_space_dim)
    sigma_diag = args.temperature * sigma_diag

    z = np.random.normal(mu, sigma_diag, (args.num_manipulation, z_space_dim))
    z = torch.from_numpy(z).to(device).float()

    save_manipulation(
        model=model,
        embedding_function=None,
        direction_list=direction_list,
        direction_basis_list=None,
        z=z, device=device, args=args)
    return


def step_02_PCA_SSL_saving():
    print("Loading from {}.".format(manipulation_generated_latent_path))
    training_z = np.load(manipulation_generated_latent_path, allow_pickle=True)
    codes = torch.from_numpy(training_z).to(device)
    print("codes", codes.shape)

    z_space_dim = training_z.shape[1]

    direction_list = PCA(n_components=10).fit_transform(training_z.T)
    direction_list = direction_list.T
    direction_list = torch.from_numpy(direction_list).to(device).float()
    print("direction_list: ", direction_list.shape)
    print("direction_list", direction_list)

    #################### save manipulated molecules ####################
    mu = np.zeros(z_space_dim)
    sigma_diag = np.ones(z_space_dim)
    sigma_diag = args.temperature * sigma_diag

    z = np.random.normal(mu, sigma_diag, (args.num_manipulation, z_space_dim))
    z = torch.from_numpy(z).to(device).float()

    save_manipulation(
        model=model,
        embedding_function=None,
        direction_list=direction_list,
        direction_basis_list=None,
        z=z, device=device, args=args)
    return


def step_02_SSL_training_and_saving():
    print("Loading from {}.".format(manipulation_generated_latent_path))
    training_z = np.load(manipulation_generated_latent_path, allow_pickle=True)
    codes = torch.from_numpy(training_z).to(device)

    z_space_dim = training_z.shape[1]

    if args.embedding_function == "Direction_Embedding_01":
        embedding_function = Direction_Embedding_01(z_space_dim).to(device)
    elif args.embedding_function == "Direction_Embedding_02":
        embedding_function = Direction_Embedding_02(z_space_dim).to(device)
    elif args.embedding_function == "Direction_Embedding_03":
        embedding_function = Direction_Embedding_03(z_space_dim).to(device)
    elif args.embedding_function == "Direction_Embedding_04":
        embedding_function = Direction_Embedding_04(z_space_dim).to(device)
    else:
        raise ValueError(
            "Energy function {} not included.".format(args.embedding_function)
        )

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    direction_basis_list = torch.normal(0, 0.1, (args.num_directions, z_space_dim))
    direction_basis_list = torch.nn.Parameter(direction_basis_list, requires_grad=False).to(device)
    model_param_group = [
        {"params": embedding_function.parameters(), "lr": args.lr},
    ]
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    if args.GraphCG_editing == "GraphCG_editing_01":
        embedding_function = GraphCG_editing_01(args, codes, embedding_function, direction_basis_list, criterion, optimizer)
    elif args.GraphCG_editing == "GraphCG_editing_02":
        embedding_function = GraphCG_editing_02(args, codes, embedding_function, direction_basis_list, criterion, optimizer)
    else:
        raise ValueError(
            "Contrastive SSL function {} not included.".format(args.embedding_function)
        )

    #################### save direction ####################
    print("\ndirection basis list: {}\n{}".format(direction_basis_list.size(), direction_basis_list))

    #################### save manipulated molecules ####################
    mu = np.zeros(z_space_dim)
    sigma_diag = np.ones(z_space_dim)
    sigma_diag = args.temperature * sigma_diag

    z = np.random.normal(mu, sigma_diag, (args.num_manipulation, z_space_dim))
    # z = torch.randn(args.num_manipulation, args.latent_size).numpy()
    z = torch.from_numpy(z).to(device).float()

    save_manipulation(
        model=model,
        embedding_function=embedding_function,
        direction_list=None,
        direction_basis_list=direction_basis_list,
        z=z, device=device, args=args)
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

    parser.add_argument("--data_name", type=str, default="qm9", choices=["qm9", "zinc250k", "chembl"], help="dataset name")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature of the gaussian distribution")
    parser.add_argument("--verbose", type=int, default=0, choices=[0, 1])

    # step 1
    parser.add_argument("--codes_option", type=str, default="reconstruction", choices=["random", "reconstruction"])
    parser.add_argument("--num_sample", type=int, default=100)
    parser.add_argument("--output_folder", type=str, default="")

    # step 2
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--embedding_function", type=str, default="Direction_Embedding_01")
    parser.add_argument("--GraphCG_editing", type=str, default="GraphCG_editing_01")
    parser.add_argument("--num_directions", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--alpha_split_num", type=int, default=21)
    parser.add_argument("--SSL_noise_level", type=float, default=1)
    parser.add_argument("--num_manipulation", type=int, default=1000, help="number of manipulated data")
    parser.add_argument("--normalize_codes", type=int, default=1, help="useful to get a converged SSL loss")
    parser.add_argument("--alpha_01", type=float, default=1, help="coeff for latent SSL")
    parser.add_argument("--alpha_02", type=float, default=1, help="coeff for direction SSL")
    parser.add_argument("--alpha_03", type=float, default=0, help="coeff for sparsity")
    parser.add_argument("--alpha_step_option", type=str, default="random", choices=["random", "first_last"])
    parser.add_argument("--alpha_step_option_random_num", type=int, default=20)

    # step 3
    parser.add_argument("--epsilon", type=str, default=0.2)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("args", args)
    #################### for argumenets ####################
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
    args.vocab = PairVocab(vocab, cuda=torch.cuda.is_available())

    train_smiles = [line.strip("\r\n ") for line in open(args.train)]

    model = HierVAE(args).to(device)

    if not args.cuda:
        model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu'))[0])
    else:
        model.load_state_dict(torch.load(args.model)[0])
    print('Done loading from {}.'.format(args.model))
    model.eval()

    if args.output_folder == "":
        manipulation_folder = os.path.join("temp/{}_manipulation".format(args.data_name))
        os.makedirs(manipulation_folder, exist_ok=True)
        manipulation_generated_latent_path = os.path.join(manipulation_folder, "generated_latent_z.npy")
    else:
        manipulation_folder = args.output_folder
        os.makedirs(manipulation_folder, exist_ok=True)
        manipulation_generated_latent_path = os.path.join(manipulation_folder, "generated_latent_z.npy")
    print("manipulation_folder", manipulation_folder)

    ########## step 01 ##########
    step_01_generate_samples()

    ########## step 02 ##########
    if args.GraphCG_editing == "GraphCG_editing_random":
        step_02_Random_SSL_saving()
    elif args.GraphCG_editing == "GraphCG_editing_variance_high":
        step_02_Variance_SSL_saving()
    elif args.GraphCG_editing == "GraphCG_editing_PCA":
        step_02_PCA_SSL_saving()
    else:
        step_02_SSL_training_and_saving()

    ########## step 03 ##########
    # props_list = ["tanimoto", "MolLogP", "TPSA", "MolWt", "qed", "sa", "drd2", "jnk3", "gsk3b"]
    props_list = ["tanimoto"]
    molecule_count_threshold_list = [2, 3, 4, 5]
    non_monotonic_ratio_threshold_list = [0, 0.2]
    step_03_evaluate_manipuated_data(manipulation_folder, props_list, molecule_count_threshold_list, non_monotonic_ratio_threshold_list, args)
