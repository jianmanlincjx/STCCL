import torch
from torch.utils.data import DataLoader
import sys
import os
sys.path.append("/data2/JM/code/STCCL")
from dataloader.temporal_coherent_loader import TemporalDataloader
from model.spatial_temproal_coherent_model import Spatial_Coherent_Correlation_Learning
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def fixed_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark =  True



if __name__ == "__main__":
    fixed_seed()
    log_dir = "/data2/JM/code/STCCL/log/stccl_without_backbone_corr"
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    train_data = TemporalDataloader("train")
    test_data = TemporalDataloader("test")
    model = Spatial_Coherent_Correlation_Learning().cuda()
    # model.load_state_dict(torch.load("/data2/JM/code/STCCL/model_ckpt/stccl_without_backbone/20_STCCL.pth"), strict=True)
    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=32)
    test_dataloader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=32)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    data_len_train = len(train_dataloader)
    data_len_test = len(test_dataloader)

    iter = 0
    for epoch in range(100):
        # model.train()
        # spatial_loss = 0.0
        # temproal_loss = 0.0
        # pos_spatial_iter = 0.0
        # neg_spatial_iter = 0.0
        # pos_temproal_iter = 0.0
        # neg_temproal_iter = 0.0
        # iter_epoch = 0

        # for batch in train_dataloader:
        #     neutral_img = batch['source_img'].cuda()
        #     emotion_img = batch['target_img'].cuda()

        #     optimizer.zero_grad()
        #     loss_spatial, pos_spatial, neg_spatial, loss_temproal, pos_temproal, neg_temproal = model(neutral_img, emotion_img)
        #     loss_all = loss_spatial + loss_temproal
        #     loss_all.backward()
        #     optimizer.step()
        #     loss_spatial = loss_spatial.item()
        #     loss_temproal = loss_temproal.item()
            
        #     spatial_loss += loss_spatial
        #     temproal_loss += loss_temproal
            
        #     pos_spatial_iter += pos_spatial
        #     neg_spatial_iter += neg_spatial
            
        #     pos_temproal_iter += pos_temproal
        #     neg_temproal_iter += neg_temproal
            
        #     iter += 1
        #     iter_epoch += 1
        #     print(f"epoch: {epoch}  iter: {iter}  spatial: loss: {loss_spatial:.3f} pos_score: {pos_spatial:.3f} neg_score: {neg_spatial:.3f} temproal: loss: {loss_temproal:.6f} pos_score: {pos_temproal:.6f} neg_score: {neg_temproal:.6f}")
        #     if iter % 100 == 0:
        #         writer.add_scalar(f"train_iter/Patch_NCELoss_iter", ((spatial_loss + temproal_loss) / 2) / iter_epoch, iter)   
        #         writer.add_scalar(f"train_iter/pos_iter", ((pos_spatial_iter + pos_temproal_iter)/ 2) / iter_epoch, iter)   
        #         writer.add_scalar(f"train_iter/neg_iter", ((neg_spatial_iter + neg_temproal_iter)/ 2) / iter_epoch, iter)   
        # writer.add_scalar(f"train_epoch/Patch_NCELoss_epoch", ((spatial_loss + temproal_loss) / 2)/data_len_train, epoch)
        # writer.add_scalar(f"train_epoch/pos_epoch", ((pos_spatial_iter + pos_temproal_iter) / 2) / data_len_train, epoch)
        # writer.add_scalar(f"train_epoch/neg_epoch", ((neg_spatial_iter + neg_temproal_iter)/ 2) / data_len_train, epoch)

        # if epoch % 5 == 0:
        #     torch.save(model.state_dict(), f'/data2/JM/code/STCCL/model_ckpt/stccl_without_backbone/{epoch}_STCCL.pth')
            
        # model.eval()
        spatial_loss = 0.0
        temproal_loss = 0.0
        pos_spatial_iter = 0.0
        neg_spatial_iter = 0.0
        pos_temproal_iter = 0.0
        neg_temproal_iter = 0.0
        with torch.no_grad():  
            for batch in test_dataloader:
                neutral_img = batch['source_img'].cuda()
                emotion_img = batch['target_img'].cuda()

                loss_spatial, pos_spatial, neg_spatial, loss_temproal, pos_temproal, neg_temproal = model(neutral_img, emotion_img)
                
                loss_spatial = loss_spatial.item()
                loss_temproal = loss_temproal.item()
                
                spatial_loss += loss_spatial
                temproal_loss += loss_temproal
                
                pos_spatial_iter += pos_spatial
                neg_spatial_iter += neg_spatial
                
                pos_temproal_iter += pos_temproal
                neg_temproal_iter += neg_temproal
                

                print(f"epoch: {epoch}  iter: {iter}  spatial: loss: {loss_spatial:.3f} pos_score: {pos_spatial:.3f} neg_score: {neg_spatial:.3f} temproal: loss: {loss_temproal:.3f} pos_score: {pos_temproal:.3f} neg_score: {neg_temproal:.3f}")
    
            writer.add_scalar(f"test_epoch/Patch_NCELoss_epoch", ((spatial_loss + temproal_loss) / 2)/data_len_train, epoch)
            writer.add_scalar(f"test_epoch/pos_epoch", ((pos_spatial_iter + pos_temproal_iter) / 2) / data_len_train, epoch)
            writer.add_scalar(f"test_epoch/neg_epoch", ((neg_spatial_iter + neg_temproal_iter)/ 2) / data_len_train, epoch)

