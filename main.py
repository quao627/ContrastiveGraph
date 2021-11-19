from data import load_dataset, ContrastiveLearningDataset
from torch_geometric.loader import DataLoader
import torch
import params
from framework import SimCLR
from models import *


if __name__ == '__main__':
    data = load_dataset("PTC_MR")
    train_dataset = ContrastiveLearningDataset(data)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    model = GCN(in_features=data.data.x.shape[1], out_features=128)

    optimizer = torch.optim.Adam(model.parameters(), params.LEARNING_RATE, weight_decay=params.WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    with torch.cuda.device(params.GPU_INDEX):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler)
        simclr.train(train_loader)