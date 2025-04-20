import numpy as np
import torch
import torch.nn as nn
import cv2
import torch
from torch import nn
import random
import torch.nn.functional as F
from itertools import combinations
import torchvision

from model.base_model import iresnet50, mlp, Normalize, TimeSeriesTransformer

class Temproal_Coherent_Correlation_Learning(nn.Module):
    def __init__(self):
        super(Temproal_Coherent_Correlation_Learning, self).__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        
        self.resnet50 = iresnet50()
        # self.time_models = [
        #     TimeSeriesTransformer(input_features=64*256, num_heads=8, num_layers=4, dim_feedforward=512).cuda(),
        #     TimeSeriesTransformer(input_features=128*128, num_heads=8, num_layers=4, dim_feedforward=512).cuda(),
        #     TimeSeriesTransformer(input_features=256*64, num_heads=8, num_layers=4, dim_feedforward=512).cuda(),
        #     TimeSeriesTransformer(input_features=512*32, num_heads=8, num_layers=4, dim_feedforward=512).cuda()
        # ]
        self.mlp = mlp
        self.size = 224
        self.start_layer = 0
        self.end_layer = 5

        self.Normalize = Normalize(2)

    def img2intermediate(self, input):
        return self.resnet50(input)
    
    def sample_location(self, feature_map_size, sub_region_size, samples_num):
        # 计算子区域边界大小
        border_size = (feature_map_size - sub_region_size) // 2
        # 生成采样索引
        indices = np.indices((sub_region_size, sub_region_size)).reshape(2, -1).T + border_size
        np.random.shuffle(indices)
        ## torch.Size(num, 2])
        sampled_indices = torch.from_numpy(indices[:samples_num])
        return sampled_indices.reshape(-1, 2).cuda()

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
    

    def compute_sequence_aggregation(self, features):
        """
        计算整个序列的关联信息。

        参数:
            features: 输入的特征张量，维度为(B, T, C, N)。

        返回:
            一个形状为(B, C, N)的张量，其中包含了整个时间序列的聚合信息。
        """
        # 计算时间维度上的平均值
        aggregated_features = features.mean(dim=1)  # 对时间维度T进行平均
        
        return aggregated_features


    def compute_time_accumulated_differences(self, features):
        """
        计算特征在时间维度上与第一帧的累积差异。

        参数:
            features: 输入的特征张量，维度为(B, T, C, N)。

        返回:
            一个形状为(B, T-1, C, N)的张量，其中每个时间步与第一时间步的累积差异。
        """
        B, T, C, N = features.size()
        
        # 初始化输出张量
        # 注意，我们只需要 T-1 因为我们用第一帧作为参照
        output = torch.zeros(B, T-1, C, N, device=features.device)
        
        # 使用第一帧作为参照
        first_frame = features[:, 0, :, :].unsqueeze(1)  # 增加一个时间维度以便广播
        
        # 计算每一帧与第一帧之间的差异
        # 我们从第二帧开始计算，因为第一帧与自身的差异为0
        output = features[:, 1:, :, :] - first_frame
        
        return output

    def shuffle_and_subtract(self, A, index=None):
        # 获取A的维度
        B, T, C, N = A.shape
        
        # 在T维度上生成随机排列的索引
        if index == None:
            idx = torch.randperm(T)
        else:
            idx = index
        # 使用随机索引打乱A的T维度
        B = A[:, idx, :, :]
        
        # 计算A与B的差值
        C = A - B
        
        return C, idx


    def compute_time_differences(self, features):
        """
        计算特征在时间维度上的差异。
        
        参数:
            features: 输入的特征张量，维度为(B, T, C, N)。
            
        返回:
            一个形状为(B, R, C, N)的张量,其中R是时间步T的排列组合个数。
        """
        B, T, C, N = features.size()
        
        # 计算组合个数 R = T * (T - 1) / 2
        R = T * (T - 1) // 2
        
        # 初始化输出张量
        output = torch.zeros(B, R, C, N, device=features.device)
        
        # 计算所有可能的时间步组合之间的差异
        idx = 0
        for i, j in combinations(range(T), 2):
            output[:, idx, :, :] = features[:, i, :, :] - features[:, j, :, :]
            idx += 1
        
        return output
    
    ## PatchNCELoss code from: https://github.com/taesungp/contrastive-unpaired-translation 
    def PatchNCELoss(self, f_q, f_k, tau=0.07):
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
        # return torch.mean(self.cross_entropy_loss(predictions, targets) * weight_pos)
        return torch.mean(self.cross_entropy_loss(predictions, targets))

    def uniform_sampling_batch(self, features, N, random_indices=None):
        """
        在 C 维度上使用相同的空间采样索引对特征张量进行采样，并将这些索引应用到所有的 T 维度中。
        
        参数:
            features: 输入的特征张量，维度为(B, T, C, H, W)。
            N: 要采样的像素数量。
        
        返回:
            一个形状为(B, T, C, N)的张量，包含从每个空间位置随机采样的特征值。
        """
        B, T, C, H, W = features.size()  # 获取输入特征的维度信息
        
        # 生成全局随机采样索引，这些索引在H*W上为所有 B, C 使用，并应用到所有的 T 中
        if random_indices is None:
            random_indices = torch.randint(0, H*W, size=(B, C, N), device=features.device)
        
        # 将特征张量重新组织以便于采样，形状变为(B, T, C, H*W)
        features_flat = features.view(B, T, C, H*W)
        
        # 生成扩展的采样索引，形状变为(B, T, C, N)
        expanded_indices = random_indices.unsqueeze(1).expand(-1, T, -1, -1).contiguous().view(B, T, C, N)
        
        # 对特征进行采样，结果形状为(B, T, C, N)
        sampled_features = torch.gather(features_flat, 3, expanded_indices)
        
        return sampled_features, random_indices

    
    def forward(self, source_input, generator_output, sample_nums=[32*8, 16*8, 8*8, 4*8], tau=0.07):
        loss_ccp = 0.0
        pos = 0.0
        neg = 0.0
        
        B_1, T, C, H, W = source_input.shape
        R = T * (T - 1) // 2
        source_feature = self.img2intermediate(source_input.reshape(B_1*T, C, H, W))
        generator_feature = self.img2intermediate(generator_output.reshape(B_1*T, C, H, W))
        
        # NCE
        for i in range(self.start_layer, self.end_layer-1):
            assert source_feature[i].shape == generator_feature[i].shape
            _, C, H, W = source_feature[i].shape
            feat_q = source_feature[i].reshape(B_1, T, C, H, W)
            feat_k = generator_feature[i].reshape(B_1, T, C, H, W)

            # location = self.sample_location(H, int(((H//2) + H*0.45)), sample_nums[i])

            # feat_q_location = feat_q[..., location[:, 0], location[:, 1]]
            # feat_k_location = feat_k[..., location[:, 0], location[:, 1]]
            # permuted_indices = torch.randperm(T)
            # feat_q_location_neg = feat_q_location[:, permuted_indices, ...]
            # feat_k_location_neg = feat_k_location[:, permuted_indices, ...]
            # f_q = (feat_q_location - feat_q_location_neg).reshape(B_1*T, C, sample_nums[i]).permute(0, 2, 1)
            # f_k = (feat_k_location - feat_k_location_neg).reshape(B_1*T, C, sample_nums[i]).permute(0, 2, 1)

            feat_q_location, sample_index_q = self.uniform_sampling(features=feat_q, N=sample_nums[i], random_indices=None)
            feat_k_location, sample_index_k = self.uniform_sampling(features=feat_k, N=sample_nums[i], random_indices=sample_index_q)

            f_q = self.compute_time_differences(feat_q_location).reshape(B_1*R, C, sample_nums[i]).permute(0, 2, 1)
            f_k = self.compute_time_differences(feat_k_location).reshape(B_1*R, C, sample_nums[i]).permute(0, 2, 1)

            # last_dimension_size = f_k.size(-1)
            # # 生成一个随机的索引排列
            # permuted_indices = torch.randperm(last_dimension_size)
            # # 使用 permuted_indices 对最后一个维度进行打乱
            # shuffled_flow_k = f_k[..., permuted_indices].detach().cpu()
            # cosine_similarity_pos = torch.mean(F.cosine_similarity(f_q.detach().cpu(), f_k.detach().cpu(), dim=-1))
            # cosine_similarity_neg = torch.mean(F.cosine_similarity(f_q.detach().cpu(), shuffled_flow_k.detach().cpu(), dim=-1))
            # with open("/data2/JM/code/STCCL/test_compute_time_differences.txt", "a") as file:
            #     file.write(f"{cosine_similarity_pos}   {cosine_similarity_neg} \n")

            for j in range(3):
                f_q =self.mlp[3*i+j](f_q)
            flow_q = self.Normalize(f_q.permute(0, 2, 1))
            
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
            pos += cosine_similarity_pos
            neg += cosine_similarity_neg

            loss_ccp += self.PatchNCELoss(flow_q, flow_k, tau)
        return loss_ccp, pos/4, neg/4
    


def fixed_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark =  True


