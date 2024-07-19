import random
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .core_editing_function import do_ssl_on_codes, do_ssl_on_directions


class LatentDataset(Dataset):
    def __init__(self, latent_codes):
        self.latent_codes = latent_codes.detach().cpu().numpy()

    def __len__(self):
        return len(self.latent_codes)

    def __getitem__(self, idx):
        return self.latent_codes[idx]


def GraphCG_editing_01_with_batch(args, whole_latent, embedding_function, direction_basis_list, criterion, optimizer):
    alpha_list = np.linspace(-args.alpha, args.alpha, args.alpha_split_num)
    alpha_list = alpha_list.tolist()
    print("alpha_list", alpha_list)

    dataset = LatentDataset(whole_latent)
    # Currently, this is only for DisCo
    loader = DataLoader(dataset, batch_size=10, num_workers=8, shuffle=True)

    for epoch in tqdm(range(args.epochs)):
        embedding_function.train()

        comb_list = None
        if args.alpha_step_option == "random":
            comb_list = list(combinations(range(args.alpha_split_num), 2))
            comb_list = comb_list[: args.alpha_step_option_random_num]
            random.shuffle(comb_list)
        elif args.alpha_step_option == "first_last":
            comb_list = [[-1, 0]]
        
        for codes in loader:
            codes = codes.cuda()

            for i ,j in comb_list:
                loss_accum, loss_accum_count = 0, 0

                #################### build up pos and negative pairs ####################
                codes_01_list, codes_02_list, codes_03_list, codes_04_list = [], [], [], []
                direction_list = []
                codes_perturbed = codes + torch.randn_like(codes) * args.noise_level
                for d in range(args.num_directions):
                    codes_01, direction_temp = embedding_function(codes, direction_basis_list[d], alpha_list[i])
                    codes_02, _ = embedding_function(codes_perturbed, direction_basis_list[d], alpha_list[i])

                    codes_03, _ = embedding_function(codes, direction_basis_list[d], alpha_list[j])
                    codes_04, _ = embedding_function(codes_perturbed, direction_basis_list[d], alpha_list[j])

                    if args.normalize_codes:
                        codes_01 = F.normalize(codes_01, p=2, dim=-1)
                        codes_02 = F.normalize(codes_02, p=2, dim=-1)
                        codes_03 = F.normalize(codes_03, p=2, dim=-1)
                        codes_04 = F.normalize(codes_04, p=2, dim=-1)

                    codes_01_list.append(codes_01)
                    codes_02_list.append(codes_02)
                    codes_03_list.append(codes_03)
                    codes_04_list.append(codes_04)
                    direction_list.append(direction_temp)

                codes_01_list = torch.stack(codes_01_list, dim=0)  # num_directions, num_sample, emb_dim
                codes_02_list = torch.stack(codes_02_list, dim=0)  # num_directions, num_sample, emb_dim
                codes_03_list = torch.stack(codes_03_list, dim=0)  # num_directions, num_sample, emb_dim
                codes_04_list = torch.stack(codes_04_list, dim=0)  # num_directions, num_sample, emb_dim
                direction_list = torch.stack(direction_list, dim=0)  # num_directions, embedding_dim

                ssl_loss_on_codes_01 = do_ssl_on_codes(codes_01_list, codes_02_list, codes_03_list, criterion, args)
                ssl_loss_on_codes_02 = do_ssl_on_codes(codes_02_list, codes_01_list, codes_04_list, criterion, args)

                loss = 0.5 * args.alpha_01 * (ssl_loss_on_codes_01 + ssl_loss_on_codes_02)

                if args.alpha_02 > 0:
                    ssl_loss_on_directions = do_ssl_on_directions(direction_list, criterion, args)
                    loss += args.alpha_02 * ssl_loss_on_directions
                if args.alpha_03 > 0:
                    sparsity_loss = torch.mean(torch.abs(direction_list))
                    loss += args.alpha_03 * sparsity_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_accum += loss.item()
                loss_accum_count += 1

            loss_accum /= loss_accum_count
            print("epoch: {}\tloss: {}".format(epoch, loss_accum))
    return embedding_function


def GraphCG_editing_02_with_batch(args, whole_latent, embedding_function, direction_basis_list, criterion, optimizer):
    alpha_list = np.linspace(-args.alpha, args.alpha, args.alpha_split_num)
    alpha_list = alpha_list.tolist()
    print("alpha_list", alpha_list)

    dataset = LatentDataset(whole_latent)
    loader = DataLoader(dataset, batch_size=500, num_workers=8, shuffle=True)

    for epoch in tqdm(range(args.epochs)):
        embedding_function.train()

        comb_list = None
        if args.alpha_step_option == "random":
            comb_list = list(combinations(range(args.alpha_split_num), 2))
            comb_list = comb_list[: args.alpha_step_option_random_num]
            random.shuffle(comb_list)
        elif args.alpha_step_option == "first_last":
            comb_list = [[-1, 0]]

        for codes in loader:
            codes = codes.cuda()

            for i ,j in comb_list:
                loss_accum, loss_accum_count = 0, 0

                #################### build up pos and negative pairs ####################
                codes_01_list, codes_02_list, codes_03_list, codes_04_list = [], [], [], []
                direction_list = []
                for d in range(args.num_directions):
                    codes_01, direction_temp = embedding_function(codes, direction_basis_list[d], alpha_list[i])
                    codes_03, _ = embedding_function(codes, direction_basis_list[d], alpha_list[j])
                    if args.normalize_codes:
                        codes_01 = F.normalize(codes_01, p=2, dim=-1)
                        codes_03 = F.normalize(codes_03, p=2, dim=-1)

                    B = codes_01.shape[0]
                    perm_idx = torch.randperm(B)
                    codes_02 = codes_01[perm_idx]  # num_sample, emb_dim
                    codes_04 = codes_03[perm_idx]  # num_sample, emb_dim
                    
                    codes_01_list.append(codes_01)
                    codes_02_list.append(codes_02)
                    codes_03_list.append(codes_03)
                    codes_04_list.append(codes_04)
                    direction_list.append(direction_temp)

                codes_01_list = torch.stack(codes_01_list, dim=0)  # num_directions, num_sample, emb_dim
                codes_02_list = torch.stack(codes_02_list, dim=0)  # num_directions, num_sample, emb_dim
                codes_03_list = torch.stack(codes_03_list, dim=0)  # num_directions, num_sample, emb_dim
                codes_04_list = torch.stack(codes_04_list, dim=0)  # num_directions, num_sample, emb_dim
                direction_list = torch.stack(direction_list, dim=0)  # num_directions, embedding_dim

                ssl_loss_on_codes_01 = do_ssl_on_codes(codes_01_list, codes_02_list, codes_03_list, criterion, args)
                ssl_loss_on_codes_02 = do_ssl_on_codes(codes_02_list, codes_01_list, codes_04_list, criterion, args)

                loss = 0.5 * args.alpha_01 * (ssl_loss_on_codes_01 + ssl_loss_on_codes_02)

                if args.alpha_02 > 0:
                    ssl_loss_on_directions = do_ssl_on_directions(direction_list, criterion, args)
                    loss += args.alpha_02 * ssl_loss_on_directions
                if args.alpha_03 > 0:
                    sparsity_loss = torch.mean(torch.abs(direction_list))
                    loss += args.alpha_03 * sparsity_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_accum += loss.item()
                loss_accum_count += 1

            loss_accum /= loss_accum_count
            print("epoch: {}\tloss: {}".format(epoch, loss_accum))
    return embedding_function


def GraphCG_editing_03_with_batch(args, whole_latent, embedding_function, direction_basis_list, criterion, optimizer):
    alpha_list = np.linspace(-args.alpha, args.alpha, args.alpha_split_num)
    alpha_list = alpha_list.tolist()

    dataset = LatentDataset(whole_latent)
    loader = DataLoader(dataset, batch_size=500, num_workers=8, shuffle=True)

    for epoch in tqdm(range(args.epochs)):
        embedding_function.train()

        comb_list = None
        if args.alpha_step_option == "random":
            comb_list = list(combinations(range(args.alpha_split_num), 2))
            comb_list = comb_list[: args.alpha_step_option_random_num]
            random.shuffle(comb_list)
        elif args.alpha_step_option == "first_last":
            comb_list = [[-1, 0]]
        
        for codes in loader:
            codes = codes.cuda()
            
            for i ,j in comb_list:
                loss_accum, loss_accum_count = 0, 0

                #################### build up pos and negative pairs ####################
                codes_01_list, codes_02_list, codes_03_list = [], [], []
                direction_list = []
                codes_perturbed = codes + torch.randn_like(codes) * args.noise_level
                for d in range(args.num_directions):
                    codes_01, direction_temp = embedding_function(codes, direction_basis_list[d], alpha_list[i])
                    codes_02, _ = embedding_function(codes_perturbed, direction_basis_list[d], alpha_list[i])
                    codes_03, _ = embedding_function(codes, direction_basis_list[d], alpha_list[j])

                    if args.normalize_codes:
                        codes_01 = F.normalize(codes_01, p=2, dim=-1)
                        codes_02 = F.normalize(codes_02, p=2, dim=-1)
                        codes_03 = F.normalize(codes_03, p=2, dim=-1)

                    codes_01_list.append(codes_01)
                    codes_02_list.append(codes_02)
                    codes_03_list.append(codes_03)
                    direction_list.append(direction_temp)

                codes_01_list = torch.stack(codes_01_list, dim=0)  # num_directions, num_sample, emb_dim
                codes_02_list = torch.stack(codes_02_list, dim=0)  # num_directions, num_sample, emb_dim
                codes_03_list = torch.stack(codes_03_list, dim=0)  # num_directions, num_sample, emb_dim
                direction_list = torch.stack(direction_list, dim=0)  # num_directions, embedding_dim

                ssl_loss_on_codes_01 = do_ssl_on_codes(codes_01_list, codes_02_list, codes_03_list, criterion, args)
        
                loss = args.alpha_01 * ssl_loss_on_codes_01

                if args.alpha_02 > 0:
                    ssl_loss_on_directions = do_ssl_on_directions(direction_list, criterion, args)
                    loss += args.alpha_02 * ssl_loss_on_directions
                if args.alpha_03 > 0:
                    sparsity_loss = torch.mean(torch.abs(direction_list))
                    loss += args.alpha_03 * sparsity_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_accum += loss.item()
                loss_accum_count += 1

            loss_accum /= loss_accum_count
            print("epoch: {}\tloss: {}".format(epoch, loss_accum))
    return embedding_function


def GraphCG_editing_04_with_batch(args, whole_latent, embedding_function, direction_basis_list, criterion, optimizer):
    alpha_list = np.linspace(-args.alpha, args.alpha, args.alpha_split_num)
    alpha_list = alpha_list.tolist()

    dataset = LatentDataset(whole_latent)
    loader = DataLoader(dataset, batch_size=500, num_workers=8, shuffle=True)

    for epoch in tqdm(range(args.epochs)):
        embedding_function.train()

        comb_list = None
        if args.alpha_step_option == "random":
            comb_list = list(combinations(range(args.alpha_split_num), 2))
            comb_list = comb_list[: args.alpha_step_option_random_num]
            random.shuffle(comb_list)
        elif args.alpha_step_option == "first_last":
            comb_list = [[-1, 0]]

        for codes in loader:
            codes = codes.cuda()

            for i ,j in comb_list:
                loss_accum, loss_accum_count = 0, 0

                #################### build up pos and negative pairs ####################
                codes_01_list, codes_02_list, codes_03_list = [], [], []
                direction_list = []
                for d in range(args.num_directions):
                    codes_01, direction_temp = embedding_function(codes, direction_basis_list[d], alpha_list[i])
                    codes_03, _ = embedding_function(codes, direction_basis_list[d], alpha_list[j])
                    if args.normalize_codes:
                        codes_01 = F.normalize(codes_01, p=2, dim=-1)
                        codes_03 = F.normalize(codes_03, p=2, dim=-1)

                    B = codes_01.shape[0]
                    perm_idx = torch.randperm(B)
                    codes_02 = codes_01[perm_idx]  # num_sample, emb_dim
                    
                    codes_01_list.append(codes_01)
                    codes_02_list.append(codes_02)
                    codes_03_list.append(codes_03)
                    direction_list.append(direction_temp)

                codes_01_list = torch.stack(codes_01_list, dim=0)  # num_directions, num_sample, emb_dim
                codes_02_list = torch.stack(codes_02_list, dim=0)  # num_directions, num_sample, emb_dim
                codes_03_list = torch.stack(codes_03_list, dim=0)  # num_directions, num_sample, emb_dim
                direction_list = torch.stack(direction_list, dim=0)  # num_directions, embedding_dim

                ssl_loss_on_codes_01 = do_ssl_on_codes(codes_01_list, codes_02_list, codes_03_list, criterion, args)

                loss = args.alpha_01 * ssl_loss_on_codes_01

                if args.alpha_02 > 0:
                    ssl_loss_on_directions = do_ssl_on_directions(direction_list, criterion, args)
                    loss += args.alpha_02 * ssl_loss_on_directions
                if args.alpha_03 > 0:
                    sparsity_loss = torch.mean(torch.abs(direction_list))
                    loss += args.alpha_03 * sparsity_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_accum += loss.item()
                loss_accum_count += 1

            loss_accum /= loss_accum_count
            print("epoch: {}\tloss: {}".format(epoch, loss_accum))
    return embedding_function
