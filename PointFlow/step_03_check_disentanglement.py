import argparse
import copy
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, "./PointFlow")
import open3d as o3d
from datasets import get_datasets
from GraphCG.disentanglement_utils import (do_Beta_VAE, do_DCI, do_Factor_VAE,
                                           do_MIG, do_Modularity, do_SAP)
from models.networks import PointFlow

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
    for data in tqdm(loader):
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
    
    for data in tqdm(loader):
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
def step_1_get_latent():
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
    return latent_z, ref_pcs


def load_descriptors_from_file(descriptors_file):
    print("descriptor file: ", descriptors_file)

    trigger = False
    descriptors_list = []
    with open(descriptors_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("VFH 308bins."):
                trigger = True
            elif trigger:
                line = line.strip().split('\t')
                descriptors_list.append(line)
                trigger = False
    descriptors_list = np.array(descriptors_list).astype(float)
    return descriptors_list


def step_3_check_disentanglement(generated_latent_list, label_list):
    n_data = label_list.shape[0]
    n_factor = label_list.shape[1]
    n_train = int(0.8 * n_data)

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
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--random_state", type=int, default=123)

    # step 1
    parser.add_argument("--codes_option", type=str, default="reconstruction", choices=["random", "reconstruction"])
    parser.add_argument("--num_sample", type=int, default=100)
    parser.add_argument("--output_folder", type=str, default="")
    parser.add_argument("--step", type=str, default='01', choices=['01', '03'])

    args = parser.parse_args()
    print("arguments:", args)
    #################### for argumenets ####################
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.step == "01":
            
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

        latent, points = step_1_get_latent()
        print("latent", latent.shape)
        print("points", points.shape)

        np.savez("{}/latent".format(args.output_folder), latent=latent)
        N = points.shape[0]
        for i in range(N):
            point = points[i]

            pcl = o3d.geometry.PointCloud()
            pcl.points = o3d.utility.Vector3dVector(point)
            o3d.io.write_point_cloud("{}/points/{}.pcd".format(args.output_folder, i), pcl)

    else:
        latent_file = '{}/latent.npz'.format(args.output_folder)
        data = np.load(latent_file)
        latent_list = data['latent']

        descriptors_file = '{}/descriptors.out'.format(args.output_folder)
        descriptors_labels = load_descriptors_from_file(descriptors_file)
        print("descriptors_labels: ", descriptors_labels.shape)

        # some preprocessing
        candidate_idx_list_airplane = set([
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 242, 243, 244, 245, 246, 247, 248
        ])
        candidate_idx_list_car = set([
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 240, 241, 242, 243, 244, 245, 246, 247
        ])
        candidate_idx_list_chair = set([
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250
        ])

        valid_idx_list = []
        print(descriptors_labels)
        n_factors = descriptors_labels.shape[1]
        candidate_idx_list = np.arange(n_factors)

        candidate_idx_list = list(candidate_idx_list_airplane.intersection(candidate_idx_list_car).intersection(candidate_idx_list_chair))
        print("len of joint candidate list: ", len(candidate_idx_list))

        neo_descriptors_labels = []
        for i in candidate_idx_list:
            temp = descriptors_labels[:, i]
            sorted_temp = sorted(temp)
            sorted_temp = np.array(sorted_temp)

            mid = np.median(sorted_temp)
            temp = (temp > mid).astype(int)
            print(i, set(temp), np.sum(temp==1), np.sum(temp==0))
            neo_descriptors_labels.append(temp)

        print("latent: ", latent_list.shape)
        neo_descriptors_labels = np.stack(neo_descriptors_labels).T
        print("neo_descriptors_labels: ", neo_descriptors_labels.shape)

        step_3_check_disentanglement(latent_list, neo_descriptors_labels)
