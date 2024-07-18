import random
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class Direction_Embedding_01(torch.nn.Module):
    def __init__(self, emb_dim, normalization=True):
        super().__init__()
        self.emb_dim = emb_dim
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
        return h

    def forward(self, z, direction_basis, alpha, **kwargs):
        direction = self.get_direction(direction_basis)
        h = z + alpha * direction
        return h, direction


class Direction_Embedding_02(torch.nn.Module):
    def __init__(self, emb_dim, normalization=True):
        super().__init__()
        self.emb_dim = emb_dim
        self.direction_basis_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )
        self.normalization = normalization
        return
    
    def get_direction(self, direction_basis):
        direction = self.direction_basis_mlp(direction_basis)
        if self.normalization:
            direction = F.normalize(direction, p=2, dim=-1)
            direction = direction**2

        return direction
    
    def get_latent(self, z, direction_basis, alpha, **kwargs):
        direction = self.get_direction(direction_basis)
        h = z + alpha * direction
        return h

    def forward(self, z, direction_basis, alpha, **kwargs):
        direction = self.get_direction(direction_basis)
        h = z + alpha * direction
        return h, direction


class Direction_Embedding_03(torch.nn.Module):
    def __init__(self, emb_dim, normalization=True):
        super().__init__()
        self.emb_dim = emb_dim
        self.direction_basis_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.z_projection = nn.Sequential(
            nn.Linear(2*emb_dim+1, emb_dim),
            nn.ReLU(),
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
        N = z.shape[0]
        alpha_list = torch.ones(N, 1).to(z.device)
        direction_temp = direction.unsqueeze(0).expand(N, -1)
        z_prime = torch.cat([z, direction_temp, alpha_list], dim=-1)
        h = z + alpha * direction + self.z_projection(z_prime)
        return h

    def forward(self, z, direction_basis, alpha, **kwargs):
        direction = self.get_direction(direction_basis)
        N = z.shape[0]
        alpha_list = torch.ones(N, 1).to(z.device)
        direction_temp = direction.unsqueeze(0).expand(N, -1)
        z_prime = torch.cat([z, direction_temp, alpha_list], dim=-1)
        h = z + alpha * direction + self.z_projection(z_prime)
        return h, direction


class Direction_Embedding_04(torch.nn.Module):
    def __init__(self, emb_dim, normalization=True):
        super().__init__()
        self.emb_dim = emb_dim
        self.direction_basis_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.z_projection = nn.Sequential(
            nn.Linear(emb_dim+1, emb_dim),
            nn.ReLU(),
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
        n = z.shape[0]
        alpha_list = torch.ones(n, 1).to(z.device)
        z_with_alpha = torch.cat([z, alpha_list], dim=-1)
        h = z + alpha * direction + self.z_projection(z_with_alpha)
        return h

    def forward(self, z, direction_basis, alpha, **kwargs):
        direction = self.get_direction(direction_basis)
        n = z.shape[0]
        alpha_list = torch.ones(n, 1).to(z.device)
        z_with_alpha = torch.cat([z, alpha_list], dim=-1)
        h = z + alpha * direction + self.z_projection(z_with_alpha)
        return h, direction


def GraphCG_editing_01(args, codes, embedding_function, direction_basis_list, criterion, optimizer):
    alpha_list = np.linspace(-args.alpha, args.alpha, args.alpha_split_num)
    alpha_list = alpha_list.tolist()
    print("alpha_list", alpha_list)

    for epoch in tqdm(range(args.epochs)):
        embedding_function.train()

        comb_list = None
        if args.alpha_step_option == "random":
            comb_list = list(combinations(range(args.alpha_split_num), 2))
            comb_list = comb_list[: args.alpha_step_option_random_num]
            random.shuffle(comb_list)
        elif args.alpha_step_option == "first_last":
            comb_list = [[-1, 0]]
        
        for i ,j in comb_list:
            loss_accum, loss_accum_count = 0, 0

            #################### build up pos and negative pairs ####################
            codes_01_list, codes_02_list, codes_03_list, codes_04_list = [], [], [], []
            direction_list = []
            codes_perturbed = codes + torch.randn_like(codes) * args.SSL_noise_level
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


def GraphCG_editing_02(args, codes, embedding_function, direction_basis_list, criterion, optimizer):
    alpha_list = np.linspace(-args.alpha, args.alpha, args.alpha_split_num)
    alpha_list = alpha_list.tolist()
    print("alpha_list", alpha_list)

    for epoch in tqdm(range(args.epochs)):
        embedding_function.train()

        comb_list = None
        if args.alpha_step_option == "random":
            comb_list = list(combinations(range(args.alpha_split_num), 2))
            comb_list = comb_list[: args.alpha_step_option_random_num]
            random.shuffle(comb_list)
        elif args.alpha_step_option == "first_last":
            comb_list = [[-1, 0]]

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


def GraphCG_editing_03(args, codes, embedding_function, direction_basis_list, criterion, optimizer):
    alpha_list = np.linspace(-args.alpha, args.alpha, args.alpha_split_num)
    alpha_list = alpha_list.tolist()

    for epoch in tqdm(range(args.epochs)):
        embedding_function.train()

        comb_list = None
        if args.alpha_step_option == "random":
            comb_list = list(combinations(range(args.alpha_split_num), 2))
            comb_list = comb_list[: args.alpha_step_option_random_num]
            random.shuffle(comb_list)
        elif args.alpha_step_option == "first_last":
            comb_list = [[-1, 0]]
        
        for i ,j in comb_list:
            loss_accum, loss_accum_count = 0, 0

            #################### build up pos and negative pairs ####################
            codes_01_list, codes_02_list, codes_03_list = [], [], []
            direction_list = []
            codes_perturbed = codes + torch.randn_like(codes) * args.SSL_noise_level
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


def GraphCG_editing_04(args, codes, embedding_function, direction_basis_list, criterion, optimizer):
    alpha_list = np.linspace(-args.alpha, args.alpha, args.alpha_split_num)
    alpha_list = alpha_list.tolist()

    for epoch in tqdm(range(args.epochs)):
        embedding_function.train()

        comb_list = None
        if args.alpha_step_option == "random":
            comb_list = list(combinations(range(args.alpha_split_num), 2))
            comb_list = comb_list[: args.alpha_step_option_random_num]
            random.shuffle(comb_list)
        elif args.alpha_step_option == "first_last":
            comb_list = [[-1, 0]]

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


def do_ssl_on_codes(codes_01_list, codes_02_list, codes_03_list, criterion, args):
    '''
    codes_01_list: direction_number, num_sample, emb_dim
    codes_01_list: direction_number, num_sample, emb_dim
    codes_01_list: direction_number, num_sample, emb_dim
    '''
    loss, count = 0, 0
    pred_pos = torch.sum(codes_01_list * codes_02_list, dim=-1)  # direction_number, num_sample
    pred_pos = pred_pos.view(-1)  # direction_number * num_sample
    temp_loss_01 = criterion(pred_pos.double(), torch.ones_like(pred_pos).double())  # direction_number * num_sample
    loss += temp_loss_01.sum()
    count += pred_pos.size()[0]

    pred_neg = torch.sum(codes_01_list * codes_03_list, dim=-1)  # direction_number, num_sample
    pred_neg = pred_neg.view(-1)  # direction_number * num_sample
    temp_loss_02 = criterion(pred_neg.double(), torch.zeros_like(pred_neg).double())  # direction_number * num_sample
    loss += temp_loss_02.sum()
    count += pred_neg.size()[0]

    codes_01_list_temp = codes_01_list.transpose(0, 1)  # num_sample, direction_number, emb_dim
    pred_neg = torch.bmm(codes_01_list_temp, codes_01_list_temp.transpose(1, 2))  # num_sample, direction_number, direction_number
    temp_loss_03 = criterion(pred_neg.double(), torch.zeros_like(pred_neg).double())  # num_sample, direction_number, direction_number
    N = temp_loss_03.size()[0]  # num_sample

    # comment: We average them over the first dimension (num_sample) for efficiency, will multiply it back
    temp_loss_03 = temp_loss_03.mean(0)  # direction_number, direction_number
    triu_pred = torch.triu(temp_loss_03, diagonal=1)  # direction_number, direction_number
    triu_indices = triu_pred.nonzero().T  # tuple: (M, M)  // # Let M = direction_number * (direction_number-1) / 2
    temp_loss_03 = temp_loss_03[triu_indices[0], triu_indices[1]]  # M

    # comment: When calculating the loss and count, need to multiply back N
    loss += temp_loss_03.sum() * N
    count += temp_loss_03.size()[0] * N

    loss /= count
    return loss


def do_ssl_on_directions(direction_list, criterion, args):
    loss, count = 0, 0

    pred = torch.matmul(direction_list, direction_list.T)  # direction_number, direction_number

    triu_pred = torch.triu(pred, diagonal=1)  # direction_number, direction_number
    triu_indices = triu_pred.nonzero().T  # tuple: (M, M)  // # Let M = direction_number * (direction_number-1) / 2
    pred_neg = pred[triu_indices[0], triu_indices[1]]  # M
    loss = criterion(pred_neg.double(), torch.zeros_like(pred_neg).double()).sum()
    count = pred_neg.size()[0]

    loss /= count
    return loss
