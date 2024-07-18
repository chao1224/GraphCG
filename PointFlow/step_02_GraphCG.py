import argparse
import copy
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, "./PointFlow")
from datasets import get_datasets
from GraphCG import (Direction_Embedding_01, Direction_Embedding_02,
                     Direction_Embedding_03, Direction_Embedding_04,
                     contrastive_SSL_01, contrastive_SSL_02,
                     contrastive_SSL_03, contrastive_SSL_04)
from models.networks import PointFlow

from visualization_point_cloud import plot_matrix3d_three_views_plt

NONLINEARITIES = ["tanh", "relu", "softplus", "elu", "swish", "square", "identity"]
SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
LAYERS = ["ignore", "concat", "concat_v2", "squash", "concatsquash", "scale", "concatscale"]


def get_test_loader(args):
    _, test_dataset = get_datasets(args)
    if args.resume_dataset_mean is not None and args.resume_dataset_std is not None:
        mean = np.load(args.resume_dataset_mean)
        std = np.load(args.resume_dataset_std)
        test_dataset.renormalize(mean, std)
    loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False)
    return loader


def generate_from_random(model, args):
    loader = get_test_loader(args)
    all_sample = []
    all_ref = []
    all_latent_z = []
    for data in loader:
        idx_b, test_pc = data['idx'], data['test_points']
        test_pc = test_pc.to(device)
        B, N = test_pc.size(0), test_pc.size(1)
        latent_z, out_pc = model.sample(B, N)

        # denormalize
        mean, std = data['mean'].float(), data['std'].float()
        mean = mean.to(device)
        std = std.to(device)
        out_pc = out_pc * std + mean
        test_pc = test_pc * std + mean

        all_sample.append(out_pc)
        all_ref.append(test_pc)
        all_latent_z.append(latent_z)

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    all_latent_z = torch.cat(all_latent_z, dim=0)
    print("Generation latent: {}, sample size: {}, reference size: {}".format(all_latent_z.size(), sample_pcs.size(), ref_pcs.size()))
    mean, std = mean[0], std[0]
    
    return all_latent_z, sample_pcs, ref_pcs, mean, std  # (405, 128), (405, 2048, 3), (405, 2048, 3)


def generate_from_reconstruction(model, args):
    print("cates: {}".format(args.cates))
    loader = get_test_loader(args)

    all_sample = []
    all_ref = []
    all_latent_z = []
    for data in loader:
        idx_b, train_pc, test_pc = data['idx'], data['train_points'], data['test_points']
        test_pc = test_pc.to(device)
        train_pc = train_pc.to(device)
        B, N = test_pc.size(0), test_pc.size(1)
        latent_z, out_pc = model.reconstruct(test_pc, num_points=N, return_latent=True)
        mean, std = data['mean'].float(), data['std'].float()
        mean = mean.to(device)
        std = std.to(device)
        out_pc = out_pc * std + mean
        test_pc = test_pc * std + mean

        all_sample.append(out_pc)
        all_ref.append(test_pc)
        all_latent_z.append(latent_z)

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    all_latent_z = torch.cat(all_latent_z, dim=0)
    print("Generation latent: {}, sample size: {}, reference size: {}".format(all_latent_z.size(), sample_pcs.size(), ref_pcs.size()))
    mean, std = mean[0], std[0]
    return all_latent_z, sample_pcs, ref_pcs, mean, std  # (405, 128), (405, 2048, 3), (405, 2048, 3)


@torch.no_grad()
def step_01_generate_samples():
    if args.codes_option == "random":
        latent_z, sample_pcs, ref_pcs, mean, std= generate_from_random(model, args)
    else:
        latent_z, sample_pcs, ref_pcs, mean, std = generate_from_reconstruction(model, args)
        
    latent_z = latent_z.cpu().detach().numpy()
    sample_pcs = sample_pcs.cpu().detach().numpy()
    ref_pcs = ref_pcs.cpu().detach().numpy()
    mean = mean.cpu().detach().numpy()
    std = std.cpu().detach().numpy()
    print(latent_z.shape, sample_pcs.shape, ref_pcs.shape, mean.shape, std.shape)
    np.savez_compressed(
        manipulation_generated_latent_path,
        latent_z=latent_z,
        sample_pcs=sample_pcs,
        ref_pcs=ref_pcs,
        mean=mean,
        std=std,
    )
    return


def step_02_SSL_training_and_saving():
    data = np.load(manipulation_generated_latent_path+".npz")
    latent_z = data['latent_z']
    sample_pcs = data['sample_pcs']
    ref_pcs = data['ref_pcs']
    mean = data['mean']
    std = data['std']

    N = latent_z.shape[0]
    print("sample {} from {}".format(args.num_sample, N))
    if args.num_sample < N:
        sampled_N = random.sample(range(N), args.num_sample)
        sampled_N = np.array(sampled_N)
        print("sampled_N", sampled_N)
    else:
        sampled_N = np.arange(N)
    training_z = latent_z[sampled_N]
    np.savez_compressed(manipulation_sampled_index_path, sampled_N=sampled_N)

    codes = torch.from_numpy(training_z).to(device)
    mean = torch.from_numpy(mean).to(device)
    std = torch.from_numpy(std).to(device)
    z_space_dim = training_z.shape[1]

    if args.embedding_function == "Direction_Embedding_01":
        embedding_function = Direction_Embedding_01(z_space_dim).cuda()
    elif args.embedding_function == "Direction_Embedding_02":
        embedding_function = Direction_Embedding_02(z_space_dim).cuda()
    elif args.embedding_function == "Direction_Embedding_03":
        embedding_function = Direction_Embedding_03(z_space_dim).cuda()
    elif args.embedding_function == "Direction_Embedding_04":
        embedding_function = Direction_Embedding_04(z_space_dim).cuda()
    else:
        raise ValueError(
            "Energy function {} not included.".format(args.embedding_function)
        )

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    direction_basis_list = torch.normal(0, 0.1, (args.num_directions, z_space_dim))
    direction_basis_list = torch.nn.Parameter(direction_basis_list, requires_grad=False).cuda()
    model_param_group = [
        {"params": embedding_function.parameters(), "lr": args.lr},
    ]
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    
    if args.contrastive_SSL == "contrastive_SSL_01":
        embedding_function = contrastive_SSL_01(args, codes, embedding_function, direction_basis_list, criterion, optimizer)
    elif args.contrastive_SSL == "contrastive_SSL_02":
        embedding_function = contrastive_SSL_02(args, codes, embedding_function, direction_basis_list, criterion, optimizer)
    elif args.contrastive_SSL == "contrastive_SSL_03":
        embedding_function = contrastive_SSL_03(args, codes, embedding_function, direction_basis_list, criterion, optimizer)
    elif args.contrastive_SSL == "contrastive_SSL_04":
        embedding_function = contrastive_SSL_04(args, codes, embedding_function, direction_basis_list, criterion, optimizer)
    else:
        raise ValueError(
            "Contrastive SSL function {} not included.".format(args.embedding_function)
        )

    #################### save direction ####################
    print("\ndirection basis list: {}\n{}".format(direction_basis_list.size(), direction_basis_list))

    #################### save manipulated point clouds ####################
    w = model.sample_gaussian((args.num_manipulation, z_space_dim))
    z = model.latent_cnf(w, None, reverse=True).view(*w.size())

    step_03_save_and_draw_manipulation(
        model=model,
        embedding_function=embedding_function,
        direction_list=None,
        direction_basis_list=direction_basis_list,
        z=z, mean=mean, std=std, device=device, args=args)
    
    selected_idx = sampled_N[:5]
    visualization_output_dir = os.path.join(manipulation_folder, "visualization")
    visualization_output_file = os.path.join(visualization_output_dir, "sample")
    plot_matrix3d_three_views_plt(visualization_output_file, sample_pcs[selected_idx], selected_idx)
    visualization_output_file = os.path.join(visualization_output_dir, "ref")
    plot_matrix3d_three_views_plt(visualization_output_file, ref_pcs[selected_idx], selected_idx)

    z, test_x = model.sample(args.num_manipulation, 2048)
    test_x = test_x * std + mean
    test_x = test_x.detach().cpu().numpy()
    selected_idx = range(5)
    visualization_output_file = os.path.join(visualization_output_dir, "reconstruct_01")
    plot_matrix3d_three_views_plt(visualization_output_file, test_x[selected_idx], selected_idx)

    _, data_pc = model.decode(z, num_points=2048)
    data_pc = data_pc * std + mean
    data_pc = data_pc.detach().cpu().numpy()
    selected_idx = range(5)
    visualization_output_file = os.path.join(visualization_output_dir, "reconstruct_02")
    plot_matrix3d_three_views_plt(visualization_output_file, data_pc[selected_idx], selected_idx)

    return


@torch.no_grad()
def step_03_save_and_draw_manipulation(
    embedding_function, args, model, z, mean, std, direction_list, direction_basis_list, device):
    print("z: ", z.shape)

    alpha_list = np.linspace(-args.alpha, args.alpha, args.alpha_split_num_for_visual)
    alpha_list = alpha_list.tolist()
    assert len(alpha_list) == args.alpha_split_num_for_visual
    
    for direction_idx in range(args.num_directions):
        manipulated_data_folder = os.path.join(manipulation_folder, "manipulated_data")
        os.makedirs(manipulated_data_folder, exist_ok=True)
        output_file = os.path.join(manipulated_data_folder, "direction_{}".format(direction_idx))

        data_idx_list, step_idx_list, data_list = [], [], []
        for data_idx in tqdm(range(args.num_manipulation)):
            z_manipulated_list = []
            print(data_idx , args.num_manipulation)

            for step_idx in range(args.alpha_split_num_for_visual):
                if embedding_function is not None:
                    z_neo_manipulated = embedding_function.get_latent(z[data_idx:data_idx+1], direction_basis_list[direction_idx], alpha_list[step_idx])
                    z_neo_manipulated = z_neo_manipulated.squeeze()
                    z_manipulated_list.append(z_neo_manipulated)
                else:
                    z_neo_manipulated = z[data_idx] + direction_list[direction_idx] * alpha_list[step_idx]
                    z_manipulated_list.append(z_neo_manipulated)

            z_manipulated_list = torch.stack(z_manipulated_list, dim=0)  # 21, 128
            print("z_manipulated_list", z_manipulated_list.shape)

            _, data_pc = model.decode(z_manipulated_list, num_points=2048)
            data_pc = data_pc * std + mean
            data_pc = data_pc.cpu().detach().numpy()  # 21, 2048, 3

            data_idx_list.extend([data_idx for _ in range(args.alpha_split_num_for_visual)])
            step_idx_list.extend([step_idx for step_idx in range(args.alpha_split_num_for_visual)])
            data_list.extend(data_pc)

            visualization_output_dir = os.path.join(manipulation_folder, "visualization", "data_{}".format(data_idx))
            os.makedirs(visualization_output_dir, exist_ok=True)
            visualization_output_file = os.path.join(visualization_output_dir, "direction_{}".format(direction_idx))
            print("Visualizing direction {} data {}...".format(direction_idx, data_idx))
            plot_matrix3d_three_views_plt(visualization_output_file, data_pc, titles=alpha_list)

        data_list = np.stack(data_list, axis=0)  # 21 * num_manipulation, 2048, 3
        
        assert len(data_idx_list) == len(step_idx_list) == len(data_list)

        np.savez_compressed(output_file, data_idx=data_idx_list, step_idx=step_idx_list, data=data_list)
    return


if __name__ == "__main__":
    #################### for argumenets ####################
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default="shapenet15k", choices=['shapenet15k', 'modelnet40_15k', 'modelnet10_15k'])
    parser.add_argument('--cates', type=str, nargs='+', default=["airplane"])
    parser.add_argument('--data_dir', type=str, default="data/ShapeNetCore.v2.PC15k")

    # model architecture options
    parser.add_argument('--input_dim', type=int, default=3,
                        help='Number of input dimensions (3 for 3D point clouds)')
    parser.add_argument('--dims', type=str, default='256')
    parser.add_argument('--latent_dims', type=str, default='256')
    parser.add_argument("--num_blocks", type=int, default=1,
                        help='Number of stacked CNFs.')
    parser.add_argument("--latent_num_blocks", type=int, default=1,
                        help='Number of stacked CNFs.')
    parser.add_argument("--layer_type", type=str, default="concatsquash", choices=LAYERS)
    parser.add_argument('--time_length', type=float, default=0.5)
    parser.add_argument('--train_T', type=eval, default=True, choices=[True, False])
    parser.add_argument("--nonlinearity", type=str, default="tanh", choices=NONLINEARITIES)
    parser.add_argument('--use_adjoint', type=eval, default=True, choices=[True, False])
    parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--batch_norm', type=eval, default=True, choices=[True, False])
    parser.add_argument('--sync_bn', type=eval, default=False, choices=[True, False])
    parser.add_argument('--bn_lag', type=float, default=0)
    parser.add_argument('--use_latent_flow', action='store_true')
    parser.add_argument('--use_deterministic_encoder', action='store_true')
    parser.add_argument('--zdim', type=int, default=128)
    parser.add_argument('--recon_weight', type=float, default=1.)
    parser.add_argument('--prior_weight', type=float, default=1.)
    parser.add_argument('--entropy_weight', type=float, default=1.)
    parser.add_argument('--resume_checkpoint', type=str, default=None)

    parser.add_argument('--distributed', action='store_true')
    parser.add_argument("--tr_max_sample_points", type=int, default=2048)
    parser.add_argument("--te_max_sample_points", type=int, default=2048)
    parser.add_argument('--dataset_scale', type=float, default=1.)
    parser.add_argument('--normalize_per_shape', action='store_true')
    parser.add_argument('--normalize_std_per_axis', action='store_true')
    parser.add_argument('--resume_dataset_mean', type=str, default=None)
    parser.add_argument('--resume_dataset_std', type=str, default=None)

    parser.add_argument('--num_sample_shapes', default=10, type=int)
    parser.add_argument('--num_sample_points', default=2048, type=int)

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)

    # step 1
    parser.add_argument("--codes_option", type=str, default="reconstruction", choices=["random", "reconstruction"])
    parser.add_argument("--num_sample", type=int, default=100)
    parser.add_argument("--output_folder", type=str, default="")

    # step 2
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--embedding_function", type=str, default="Direction_Embedding_01")
    parser.add_argument("--contrastive_SSL", type=str, default="contrastive_SSL_01")
    parser.add_argument("--num_directions", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--alpha_split_num", type=int, default=11)
    parser.add_argument("--alpha_split_num_for_visual", type=int, default=5)
    parser.add_argument("--SSL_noise_level", type=float, default=1)
    parser.add_argument("--num_manipulation", type=int, default=3, help="number of manipulated data")
    parser.add_argument("--normalize_codes", type=int, default=1, help="useful to get a converged SSL loss")
    parser.add_argument("--alpha_01", type=float, default=1, help="coeff for latent SSL")
    parser.add_argument("--alpha_02", type=float, default=1, help="coeff for direction SSL")
    parser.add_argument("--alpha_03", type=float, default=0, help="coeff for sparsity")
    parser.add_argument("--alpha_step_option", type=str, default="random", choices=["random", "first_last"])
    parser.add_argument("--alpha_step_option_random_num", type=int, default=20)

    args = parser.parse_args()
    print("arguments:", args)
    #################### for argumenets ####################
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.output_folder == "":
        manipulation_folder = os.path.join("temp_manipulation")
        os.makedirs(manipulation_folder, exist_ok=True)
        manipulation_generated_latent_path = os.path.join(manipulation_folder, "generated_latent_z")
        manipulation_sampled_index_path = os.path.join(manipulation_folder, "sampled_index")
    else:
        manipulation_folder = args.output_folder
        os.makedirs(manipulation_folder, exist_ok=True)
        manipulation_generated_latent_path = os.path.join(manipulation_folder, "generated_latent_z")
        manipulation_sampled_index_path = os.path.join(manipulation_folder, "sampled_index")

    if args.gpu >= 0:
        device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print("New model...")
    def _transform_(m):
        return nn.DataParallel(m)
    model = PointFlow(args)
    model = model.to(device)
    model.multi_gpu_wrapper(_transform_)

    print("Resume Path:%s" % args.resume_checkpoint)
    checkpoint = torch.load(args.resume_checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()

    ########## step 01 ##########
    step_01_generate_samples()

    ########## step 02 and step 03 ##########
    step_02_SSL_training_and_saving()

    