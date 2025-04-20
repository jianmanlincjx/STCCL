import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2
import torch
from torch import nn
import random
import torch.nn.functional as F
from sklearn.manifold import TSNE
import os
import torchvision
from model.base_model import iresnet50, mlp, Normalize, TimeSeriesTransformer



class Spatial_Coherent_Correlation_Learning(nn.Module):
    def __init__(self):
        super(Spatial_Coherent_Correlation_Learning, self).__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        
        self.resnet50 = iresnet50()
        self.size = 224
        self.mlp = mlp
        self.start_layer = 0
        self.end_layer = 5
        self.Normalize = Normalize(2)

        self.time_models = [
            TimeSeriesTransformer(input_features=64*256, num_heads=8, num_layers=4, dim_feedforward=512).cuda(),
            TimeSeriesTransformer(input_features=128*128, num_heads=8, num_layers=4, dim_feedforward=512).cuda(),
            TimeSeriesTransformer(input_features=256*64, num_heads=8, num_layers=4, dim_feedforward=512).cuda(),
            TimeSeriesTransformer(input_features=512*32, num_heads=8, num_layers=4, dim_feedforward=512).cuda()
        ]



    def img2intermediate(self, input):
        return self.resnet50(input)

    def location2neighborhood(self, location):
        sample_nums = location.shape[0]
        offsets = torch.tensor([[-1, -1], [-1, 0], [-1, 1],
                        [0, -1],           [0, 1],
                        [1, -1], [1, 0], [1, 1]]).reshape(1, 8, 2).repeat(sample_nums, 1, 1)
        neighbors = location.reshape(sample_nums,1, 2).repeat(1, 8, 1) + offsets
        return location, neighbors

    def sample_location(self, feature_map_size, sub_region_size, samples_num):
        # 计算子区域边界大小
        border_size = (feature_map_size - sub_region_size) // 2
        # 生成采样索引
        indices = np.indices((sub_region_size, sub_region_size)).reshape(2, -1).T + border_size
        np.random.shuffle(indices)
        ## torch.Size(num, 2])
        sampled_indices = torch.from_numpy(indices[:samples_num])
        # torch.Size([num, 2])
        location, neighborhood = self.location2neighborhood(sampled_indices)
        location = location.reshape(samples_num,1,2).repeat(1,8,1)
        return location.reshape(-1, 2).cuda(), neighborhood.reshape(-1, 2).cuda()
    
    ## PatchNCELoss code from: https://github.com/taesungp/contrastive-unpaired-translation 
    def PatchNCELoss(self, f_q, f_k, weight_pos, tau=0.07):
        # batch size, channel size, and number of sample locations
        B, C, S = f_q.shape
        f_k = f_k.detach()
        # calculate v * v+: BxSx1
        l_pos = (f_k * f_q).sum(dim=1)[:, :, None]
        # calculate v * v-: BxSxS
        l_neg = torch.bmm(f_q.transpose(1, 2), f_k)
        # The diagonal entries are not negatives. Remove them.
        identity_matrix = torch.eye(S,dtype=torch.bool)[None, :, :].to(f_q.device)
        l_neg.masked_fill_(identity_matrix, -float('inf'))
        # calculate logits: (B)x(S)x(S+1)
        logits = torch.cat((l_pos, l_neg), dim=2) / tau
        # return PatchNCE loss
        predictions = logits.flatten(0, 1)
        targets = torch.zeros(B * S, dtype=torch.long).to(f_q.device)
        return torch.mean(self.cross_entropy_loss(predictions, targets) * weight_pos)
        # return torch.mean(self.cross_entropy_loss(predictions, targets))

    def uniform_sampling(self, features, N, random_indices=None):
        """
        在所有B, T, C维度上使用相同的空间采样索引对特征张量进行采样。
        
        参数:
            features: 输入的特征张量，维度为(B, T, C, H, W)。
            N: 要采样的像素数量。
        
        返回:
            一个形状为(B, T, C, N)的张量，包含从每个空间位置随机采样的特征值。
        """
        B, T, C, H, W = features.size()  # 获取输入特征的维度信息
        
        # 生成全局随机采样索引，这些索引在H*W上为所有B, T, C使用
        if random_indices == None:
            random_indices = torch.randint(0, H*W, size=(N,), device=features.device)
            
        # 重新组织特征张量以便于采样，形状变为(B*T*C, H*W)
        features_flat = features.view(B*T*C, H*W)
        
        # 对每个特征进行采样，结果形状为(B*T*C, N)
        sampled_features_flat = torch.index_select(features_flat, 1, random_indices)
        
        # 将采样后的特征重新组织为(B, T, C, N)
        sampled_features = sampled_features_flat.view(B, T, C, N)
        
        return sampled_features, random_indices
    
    def forward(self, source_input, generator_output, sample_nums=[32, 16, 8, 4], tau=0.07):
        loss_spatial = 0.0
        pos_spatial= 0.0
        neg_spatial = 0.0
        
        loss_temproal = 0.0
        pos_temproal= 0.0
        neg_temproal = 0.0
    
        B_1, T, C, H, W = source_input.shape
        # R = T * (T - 1) // 2
        source_feature = self.img2intermediate(source_input.reshape(B_1*T, C, H, W))
        generator_feature = self.img2intermediate(generator_output.reshape(B_1*T, C, H, W))
        
        # NCE
        for i in range(self.start_layer, self.end_layer-1):
            
            B, C, H, W = source_feature[i].shape
            
            ##############################spatial part########################################
            feat_q = source_feature[i]
            feat_k = generator_feature[i]
            
            location, neighborhood = self.sample_location(H, int(((H//2) + H*0.45)), sample_nums[i])
        
            feat_q_location = feat_q[:, :, location[:,0], location[:,1]]
            feat_q_neighborhood = feat_q[:, :, neighborhood[:,0], neighborhood[:,1]]
            
            ## 自适应调整学习权重
            t = torch.nn.functional.sigmoid(torch.abs((feat_q_location - feat_q_neighborhood)))
            adaptive_weight = torch.ones_like(t)
            adaptive_weight[t > 0.8] = 2 * (t[t > 0.8]) ** 2
            
            f_q = (feat_q_location - feat_q_neighborhood).permute(0, 2, 1)
            for j in range(3):
                f_q =self.mlp[3*i+j](f_q)
            flow_q = self.Normalize(f_q.permute(0, 2, 1))
     
            feat_k_location = feat_k[:, :, location[:,0], location[:,1]] 
            feat_k_neighborhood = feat_k[:, :, neighborhood[:,0], neighborhood[:,1]] 
            f_k = (feat_k_location - feat_k_neighborhood).permute(0, 2, 1)
            for j in range(3):
                f_k =self.mlp[3*i+j](f_k)
            flow_k = self.Normalize(f_k.permute(0, 2, 1))   
            
            ## 计算正负样本的相似性
            last_dimension_size = flow_k.size(-1)
            # 生成一个随机的索引排列
            permuted_indices = torch.randperm(last_dimension_size)
            # 使用 permuted_indices 对最后一个维度进行打乱
            shuffled_flow_k = flow_k[..., permuted_indices].detach().cpu()
            cosine_similarity_pos = torch.mean(F.cosine_similarity(flow_q.detach().cpu(), flow_k.detach().cpu(), dim=-1))
            cosine_similarity_neg = torch.mean(F.cosine_similarity(flow_q.detach().cpu(), shuffled_flow_k.detach().cpu(), dim=-1))
            pos_spatial += cosine_similarity_pos
            neg_spatial += cosine_similarity_neg
            loss_spatial += self.PatchNCELoss(flow_q, flow_k, adaptive_weight, tau)
            ##############################spatial part########################################
            
            ##############################temproal part########################################
            feat_q = source_feature[i].reshape(B_1, T, C, H, W)
            feat_k = generator_feature[i].reshape(B_1, T, C, H, W)

            feat_q_location, sample_index_q = self.uniform_sampling(features=feat_q, N=sample_nums[i]*8, random_indices=None)
            feat_k_location, _ = self.uniform_sampling(features=feat_k, N=sample_nums[i]*8, random_indices=sample_index_q)

            f_q = self.time_models[i](feat_q_location).reshape(B_1*T, C, sample_nums[i]*8)
            f_k = self.time_models[i](feat_k_location).reshape(B_1*T, C, sample_nums[i]*8)

            flow_q = self.Normalize(f_q)
            flow_k = self.Normalize(f_k)   

            ## 计算正负样本的相似性
            last_dimension_size = flow_k.size(-1)
            # 生成一个随机的索引排列
            permuted_indices = torch.randperm(last_dimension_size)
            # 使用 permuted_indices 对最后一个维度进行打乱
            shuffled_flow_k = flow_k[..., permuted_indices].detach().cpu()
            cosine_similarity_pos = torch.mean(F.cosine_similarity(flow_q.detach().cpu(), flow_k.detach().cpu(), dim=-1))
            cosine_similarity_neg = torch.mean(F.cosine_similarity(flow_q.detach().cpu(), shuffled_flow_k.detach().cpu(), dim=-1))
            pos_temproal += cosine_similarity_pos
            neg_temproal += cosine_similarity_neg

            loss_temproal += self.PatchNCELoss(flow_q, flow_k, tau)
            ##############################temproal part########################################
            
            
        return loss_spatial, pos_spatial/4, neg_spatial/4, loss_temproal*5, pos_temproal/4, neg_temproal/4
    


