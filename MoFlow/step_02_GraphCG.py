import argparse
import os
import sys
import random
import time
from sklearn.decomposition import PCA
from distutils.util import strtobool
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import transform_qm9, transform_zinc250k, transform_chembl
from data.data_loader import NumpyTupleDataset
from data.transform_zinc250k import (transform_fn_zinc250k, zinc250_atomic_num_list)
from data.transform_chembl import (chembl_atomic_num_list)

from mflow.models.hyperparams import Hyperparameters
from mflow.models.model import MoFlow, rescale_adj
from mflow.models.utils import (_to_numpy_array, adj_to_smiles, check_novelty, check_validity, construct_mol, correct_mol, valid_mol, valid_mol_can_with_seg)
from mflow.utils.model_utils import get_latent_vec, load_model

from GraphCG import Direction_Embedding_01, Direction_Embedding_02, Direction_Embedding_03, Direction_Embedding_04
from GraphCG import contrastive_SSL_01, contrastive_SSL_02, contrastive_SSL_03, contrastive_SSL_04, contrastive_SSL_01_with_batch
from GraphCG import step_03_evaluate_manipuated_data


def generate_molecules_from_reconstruction(model, train_dataset, batch_size, num_sample):
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    adj_list, x_list, z_list = [], [], []

    with torch.no_grad():
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

            adj_list.append(adj_rev.detach())
            x_list.append(x_rev.detach())
            z_list.append(z.detach())

            if count >= num_sample:
                break

        adj_list = torch.cat(adj_list)[:num_sample]
        x_list = torch.cat(x_list)[:num_sample]
        z_list = torch.cat(z_list)[:num_sample]

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
        print(adj.size(), x.size(), z.size())

    return adj, x, z  # (bs, 4, 9, 9), (bs, 9, 5), (bs, 369)


def step_01_generate_samples():
    valid_ratio_list, novel_ratio_list, unique_ratio_list, abs_novel_ratio_list, abs_unique_ratio_list_list = [], [], [], [], []
    generated_latent = []
    for i in range(args.n_experiments):
        if args.codes_option == "random":
            print("Generate latent with random distribution.")
            adj, x, z = generate_molecules_from_random(model, batch_size=args.batch_size, true_adj=None, temp=args.temperature, device=device)
        elif args.codes_option == "reconstruction":
            print("Generate latent with reconstruction.")
            adj, x, z  = generate_molecules_from_reconstruction(model, train_dataset, batch_size=args.batch_size, num_sample=args.num_sample)

        val_res = check_validity(adj, x, atomic_num_list, correct_validity=args.correct_validity)
        novel_r, abs_novel_r = check_novelty(val_res["valid_smiles"], train_smiles, x.shape[0])

        valid_ratio_list.append(val_res["valid_ratio"])
        novel_ratio_list.append(novel_r)
        unique_ratio_list.append(val_res["unique_ratio"])
        abs_novel_ratio_list.append(abs_novel_r)
        abs_unique_ratio_list_list.append(val_res["abs_unique_ratio"])

        n_valid = len(val_res["valid_mols"])
        adj, x, z = _to_numpy_array(adj), _to_numpy_array(x), _to_numpy_array(z)
        generated_mols = [valid_mol(construct_mol(x_elem, adj_elem, atomic_num_list)) for x_elem, adj_elem in zip(x, adj)]
        
        if args.verbose:
            L = tqdm(range(len(generated_mols)))
        else:
            L = range(len(generated_mols))
        for i in L:
            generated_latent.append(z[i])
        
    generated_latent = np.array(generated_latent)
    print("generated_latent", generated_latent.shape)

    print("Save generated latent to {}.".format(manipulation_generated_latent_path))
    np.save(manipulation_generated_latent_path, generated_latent)

    print("validity:\tmean={:.2f}%,sd={:.2f}%, vals={}".format(np.mean(valid_ratio_list), np.std(valid_ratio_list), valid_ratio_list))
    print("novelty:\tmean={:.2f}%, sd={:.2f}%, vals={}".format(np.mean(novel_ratio_list), np.std(novel_ratio_list), novel_ratio_list))
    print("uniqueness:\tmean={:.2f}%, sd={:.2f}%, vals={}".format(np.mean(unique_ratio_list), np.std(unique_ratio_list), unique_ratio_list))
    print("abs_novelty:\tmean={:.2f}%, sd={:.2f}%, vals={}".format(np.mean(abs_novel_ratio_list), np.std(abs_novel_ratio_list), abs_novel_ratio_list))
    print("abs_uniqueness:\tmean={:.2f}%, sd={:.2f}%, vals={}".format(np.mean(abs_unique_ratio_list_list), np.std(abs_unique_ratio_list_list), abs_unique_ratio_list_list))
    print("Task random generation done! Time {:.2f} seconds, Data: {}".format(time.time() - start, time.ctime()))
    return


def save_manipulation(
    embedding_function, args, model, z, direction_list, direction_basis_list, device, atomic_num_list=[6, 7, 8, 9, 0]):

    with torch.no_grad():
        alpha_list = np.linspace(-args.alpha, args.alpha, args.alpha_split_num)
        alpha_list = alpha_list.tolist()
        assert len(alpha_list) == args.alpha_split_num
        
        for direction_idx in range(args.num_directions):
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
                adj, x = model.reverse(z_manipulated_list)
                temp_smiles_list = adj_to_smiles(adj.cpu(), x.cpu(), atomic_num_list)

                data_idx_list.extend([data_idx for _ in temp_smiles_list])
                step_idx_list.extend([step_idx for step_idx in range(args.alpha_split_num)])
                smiles_list.extend(temp_smiles_list)
            
            assert len(data_idx_list) == len(step_idx_list) == len(smiles_list)

            df = pd.DataFrame(data={"data_idx": data_idx_list, "step_idx": step_idx_list, "smiles": smiles_list})
            df.to_csv(output_file, index=False)
        return


def step_02_Random_SSL_saving():
    training_z = np.load(manipulation_generated_latent_path)

    z_space_dim = training_z.shape[1]    
    direction_list = np.random.normal(0, 1, (args.num_directions, z_space_dim))
    direction_list = torch.from_numpy(direction_list).to(device).float()
    direction_list = F.normalize(direction_list, dim=-1)

    #################### save manipulated molecules ####################
    mu = np.zeros(z_space_dim)  # (369,)
    sigma_diag = np.ones(z_space_dim)  # (369,)
    if model.hyper_params.learn_dist:
        if len(model.ln_var) == 1:
            sigma_diag = np.sqrt(np.exp(model.ln_var.item())) * sigma_diag
        elif len(model.ln_var) == 2:
            sigma_diag[:model.b_size] = np.sqrt(np.exp(model.ln_var[0].item())) * sigma_diag[:model.b_size]
            sigma_diag[model.b_size+1:] = np.sqrt(np.exp(model.ln_var[1].item())) * sigma_diag[model.b_size+1:]
    print("sigma_diag\t", sigma_diag.shape)
    sigma_diag = args.temperature * sigma_diag

    z = np.random.normal(mu, sigma_diag, (args.num_manipulation, z_space_dim))
    z = torch.from_numpy(z).to(device).float()

    save_manipulation(
        model=model,
        embedding_function=None,
        direction_list=direction_list,
        direction_basis_list=None,
        z=z, device=device, args=args,
        atomic_num_list=atomic_num_list)
    return


def step_02_Variance_SSL_saving():
    training_z = np.load(manipulation_generated_latent_path)
    print("training_z", training_z.shape)

    z_space_dim = training_z.shape[1]
    direction_basis_list = torch.eye(z_space_dim)

    # calculate variance
    var_array = np.var(training_z, axis=0)
    assert len(var_array) == z_space_dim
    var_index = np.argsort(var_array)
    
    if high:
        # select index with the highest variance
        sampled_direction_idx = var_index[-args.num_directions:]
    else:
        # select index with the lowest variance
        sampled_direction_idx = var_index[:args.num_directions]
    print("sampled_direction_idx: ", sampled_direction_idx)

    #################### sample direction ####################
    print("\ndirection basis list: {}\n{}".format(direction_basis_list.size(), direction_basis_list))
    direction_list = []

    for i in sampled_direction_idx:
        direction = direction_basis_list[i]
        direction_list.append(direction.cpu().detach().numpy())
    direction_list = np.array(direction_list)
    direction_list = torch.from_numpy(direction_list).to(device).float()

    #################### save manipulated molecules ####################
    mu = np.zeros(z_space_dim)  # (369,)
    sigma_diag = np.ones(z_space_dim)  # (369,)
    if model.hyper_params.learn_dist:
        if len(model.ln_var) == 1:
            sigma_diag = np.sqrt(np.exp(model.ln_var.item())) * sigma_diag
        elif len(model.ln_var) == 2:
            sigma_diag[:model.b_size] = np.sqrt(np.exp(model.ln_var[0].item())) * sigma_diag[:model.b_size]
            sigma_diag[model.b_size+1:] = np.sqrt(np.exp(model.ln_var[1].item())) * sigma_diag[model.b_size+1:]
    print("sigma_diag\t", sigma_diag.shape)
    sigma_diag = args.temperature * sigma_diag

    z = np.random.normal(mu, sigma_diag, (args.num_manipulation, z_space_dim))
    z = torch.from_numpy(z).to(device).float()

    save_manipulation(
        model=model,
        embedding_function=None,
        direction_list=direction_list,
        direction_basis_list=None,
        z=z, device=device, args=args,
        atomic_num_list=atomic_num_list)
    return


def step_02_PCA_SSL_saving():
    training_z = np.load(manipulation_generated_latent_path)
    print("training_z", training_z.shape)

    z_space_dim = training_z.shape[1]

    #################### sample direction ####################
    direction_list = PCA(n_components=args.num_directions).fit_transform(training_z.T)
    direction_list = direction_list.T
    direction_list = torch.from_numpy(direction_list).to(device).float()
    print("direction_list: ", direction_list.shape)

    #################### save manipulated molecules ####################
    mu = np.zeros(z_space_dim)  # (369,)
    sigma_diag = np.ones(z_space_dim)  # (369,)
    if model.hyper_params.learn_dist:
        if len(model.ln_var) == 1:
            sigma_diag = np.sqrt(np.exp(model.ln_var.item())) * sigma_diag
        elif len(model.ln_var) == 2:
            sigma_diag[:model.b_size] = np.sqrt(np.exp(model.ln_var[0].item())) * sigma_diag[:model.b_size]
            sigma_diag[model.b_size+1:] = np.sqrt(np.exp(model.ln_var[1].item())) * sigma_diag[model.b_size+1:]
    print("sigma_diag\t", sigma_diag.shape)
    sigma_diag = args.temperature * sigma_diag

    z = np.random.normal(mu, sigma_diag, (args.num_manipulation, z_space_dim))
    z = torch.from_numpy(z).to(device).float()

    save_manipulation(
        model=model,
        embedding_function=None,
        direction_list=direction_list,
        direction_basis_list=None,
        z=z, device=device, args=args,
        atomic_num_list=atomic_num_list)
    return


def step_02_SSL_training_and_saving():
    training_z = np.load(manipulation_generated_latent_path)
    codes = training_z
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
    elif args.embedding_function == "MoFlowDisCo":
        embedding_function = Direction_Embedding_MoFlowDisCo(z_space_dim, model).to(device)
    else:
        raise ValueError(
            "Energy function {} not included.".format(args.embedding_function)
        )

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    direction_basis_list = torch.normal(0, 0.1, (args.num_directions, z_space_dim))
    direction_basis_list = torch.nn.Parameter(direction_basis_list, requires_grad=False).to(device)
    model_param_group = [{"params": embedding_function.parameters(), "lr": args.lr},]
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    
    if args.contrastive_SSL == "contrastive_SSL_01":
        embedding_function = contrastive_SSL_01(args, codes, embedding_function, direction_basis_list, criterion, optimizer)
    elif args.contrastive_SSL == "contrastive_SSL_02":
        embedding_function = contrastive_SSL_02(args, codes, embedding_function, direction_basis_list, criterion, optimizer)
    elif args.contrastive_SSL == "contrastive_SSL_03":
        embedding_function = contrastive_SSL_03(args, codes, embedding_function, direction_basis_list, criterion, optimizer)
    elif args.contrastive_SSL == "contrastive_SSL_04":
        embedding_function = contrastive_SSL_04(args, codes, embedding_function, direction_basis_list, criterion, optimizer)
    elif args.contrastive_SSL == "contrastive_SSL_DisCo":
        embedding_function = contrastive_SSL_01_with_batch(args, codes, embedding_function, direction_basis_list, criterion, optimizer)
    else:
        raise ValueError(
            "Contrastive SSL function {} not included.".format(args.embedding_function)
        )

    #################### save direction ####################
    print("\ndirection basis list: {}\n{}".format(direction_basis_list.size(), direction_basis_list))

    #################### save manipulated molecules ####################
    mu = np.zeros(z_space_dim)  # (369,)
    sigma_diag = np.ones(z_space_dim)  # (369,)
    if model.hyper_params.learn_dist:
        if len(model.ln_var) == 1:
            sigma_diag = np.sqrt(np.exp(model.ln_var.item())) * sigma_diag
        elif len(model.ln_var) == 2:
            sigma_diag[:model.b_size] = np.sqrt(np.exp(model.ln_var[0].item())) * sigma_diag[:model.b_size]
            sigma_diag[model.b_size+1:] = np.sqrt(np.exp(model.ln_var[1].item())) * sigma_diag[model.b_size+1:]
    print("sigma_diag\t", sigma_diag.shape)
    sigma_diag = args.temperature * sigma_diag

    z = np.random.normal(mu, sigma_diag, (args.num_manipulation, z_space_dim))
    z = torch.from_numpy(z).to(device).float()

    save_manipulation(
        model=model,
        embedding_function=embedding_function,
        direction_list=None,
        direction_basis_list=direction_basis_list,
        z=z, device=device, args=args,
        atomic_num_list=atomic_num_list)
    return


class Direction_Embedding_MoFlowDisCo(torch.nn.Module):
    def __init__(self, emb_dim, moflow_model, normalization=True):
        super().__init__()
        self.emb_dim = emb_dim
        self.moflow_model = moflow_model
        self.direction_basis_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
        )
        self.normalization = normalization
        return
    
    def get_direction(self, direction_basis):
        direction = self.direction_basis_mlp(direction_basis)
        if self.normalization:
            direction = F.normalize(direction, p=2, dim=-1)
        return direction

    def get_latent(self, z, direction_basis, alpha, **kwargs):
        direction = self.get_direction(direction_basis)
        h = z + alpha * direction

        adj, x = self.moflow_model.reverse(h)
        adj_normalized = rescale_adj(adj).to(device)
        h, _ = self.moflow_model(adj, x, adj_normalized)

        h0 = h[0].reshape(h[0].shape[0], -1)
        h1 = h[1].reshape(h[1].shape[0], -1)
        h = torch.cat([h0, h1], dim=1)
        return h

    def forward(self, z, direction_basis, alpha, **kwargs):
        direction = self.get_direction(direction_basis)
        h = z + alpha * direction

        adj, x = self.moflow_model.reverse(h)
        adj_normalized = rescale_adj(adj).to(device)
        h, _ = self.moflow_model(adj, x, adj_normalized)

        h0 = h[0].reshape(h[0].shape[0], -1)
        h1 = h[1].reshape(h[1].shape[0], -1)
        h = torch.cat([h0, h1], dim=1)
        return h, direction


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
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature of the gaussian distribution")
    parser.add_argument("--verbose", type=int, default=0, choices=[0, 1])

    # step 1
    parser.add_argument("--n_experiments", type=int, default=1, help="number of times generation to be run")
    parser.add_argument("--correct_validity", type=strtobool, default="true", help="if apply validity correction after the generation")
    parser.add_argument("--codes_option", type=str, default="reconstruction", choices=["random", "reconstruction"])
    parser.add_argument("--num_sample", type=int, default=100)
    parser.add_argument("--output_folder", type=str, default="")

    # step 2
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--embedding_function", type=str, default="Direction_Embedding_01")
    parser.add_argument("--contrastive_SSL", type=str, default="contrastive_SSL_01")
    parser.add_argument("--num_directions", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--alpha_split_num", type=int, default=21)
    parser.add_argument("--SSL_noise_level", type=float, default=1)
    parser.add_argument("--num_manipulation", type=int, default=3, help="number of manipulated data")
    parser.add_argument("--normalize_codes", type=int, default=1, help="useful to get a converged SSL loss")
    parser.add_argument("--alpha_01", type=float, default=1, help="coeff for latent SSL")
    parser.add_argument("--alpha_02", type=float, default=1, help="coeff for direction SSL")
    parser.add_argument("--alpha_03", type=float, default=0, help="coeff for sparsity")
    parser.add_argument("--alpha_step_option", type=str, default="random", choices=["random", "first_last"])
    parser.add_argument("--alpha_step_option_random_num", type=int, default=20)

    # step 3
    parser.add_argument("--epsilon", type=str, default=0.2)

    args = parser.parse_args()
    print("args", args)
    #################### for argumenets ####################
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    start = time.time()
    print("Start at Time: {}".format(time.ctime()))
    snapshot_path = os.path.join(args.model_dir, args.snapshot_path)
    hyperparams_path = os.path.join(args.model_dir, args.hyperparams_path)
    print("Loading model from {}".format(snapshot_path))
    print("Loading hyperparamaters from {}".format(hyperparams_path))

    if args.output_folder == "":
        manipulation_folder = os.path.join("temp/{}_manipulation".format(args.model_dir))
        os.makedirs(manipulation_folder, exist_ok=True)
        manipulation_generated_latent_path = os.path.join(manipulation_folder, "generated_latent_z.npy")
    else:
        manipulation_folder = args.output_folder
        os.makedirs(manipulation_folder, exist_ok=True)
        manipulation_generated_latent_path = os.path.join(manipulation_folder, "generated_latent_z.npy")
    print("manipulation_folder", manipulation_folder)

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
    print("Device: {}".format(device))
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
    train_idx = random.sample(train_idx, args.num_sample)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)

    train_x = [a[0] for a in train_dataset]
    train_adj = [a[1] for a in train_dataset]
    train_smiles = adj_to_smiles(train_adj, train_x, atomic_num_list)
    print("Loading trained model and data done! Time {:.2f} seconds".format(time.time() - start))

    ########## step 01 ##########
    step_01_generate_samples()

    ########## step 02 ##########
    if args.contrastive_SSL == "contrastive_SSL_random":
        step_02_Random_SSL_saving()
    elif args.contrastive_SSL == "contrastive_SSL_variance_high":
        step_02_Variance_SSL_saving()
    elif args.contrastive_SSL == "contrastive_SSL_PCA":
        step_02_PCA_SSL_saving()
    elif args.contrastive_SSL == "contrastive_SSL_DisCo":
        assert args.embedding_function == "MoFlowDisCo"
        step_02_SSL_training_and_saving()
    else:
        step_02_SSL_training_and_saving()

    ########## step 03 ##########
    # props_list = ["tanimoto", "MolLogP", "TPSA", "MolWt", "qed", "sa", "drd2", "jnk3", "gsk3b"]
    props_list = ["tanimoto"]
    molecule_count_threshold_list = [2, 3, 4, 5]
    non_monotonic_ratio_threshold_list = [0, 0.2]
    step_03_evaluate_manipuated_data(manipulation_folder, props_list, molecule_count_threshold_list, non_monotonic_ratio_threshold_list, args)
