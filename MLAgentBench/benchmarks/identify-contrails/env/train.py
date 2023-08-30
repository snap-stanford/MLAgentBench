import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from tqdm import tqdm
from encode import rle_encode, list_to_string

def dice_score(y_p, y_t, smooth=1e-6):
    y_p = y_p[:, :, 2:-2, 2:-2]
    y_p = F.softmax(y_p, dim=1)
    y_p = torch.argmax(y_p, dim=1, keepdim=True)
    i = torch.sum(y_p * y_t, dim=(2, 3))
    u = torch.sum(y_p, dim=(2, 3)) + torch.sum(y_t, dim=(2, 3))
    score = (2 * i + smooth)/(u + smooth)
    return torch.mean(score)

def ce_loss(y_p, y_t):
    y_p = y_p[:, :, 2:-2, 2:-2]
    y_t = y_t.squeeze(dim=1)
    weight = torch.Tensor([0.57, 4.17]).to(y_t.device)
    criterion = nn.CrossEntropyLoss(weight)
    loss = criterion(y_p, y_t)
    return loss

def false_color(band11, band14, band15):
    def normalize(band, bounds):
        return (band - bounds[0]) / (bounds[1] - bounds[0])    
    _T11_BOUNDS = (243, 303)
    _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
    _TDIFF_BOUNDS = (-4, 2)
    r = normalize(band15 - band14, _TDIFF_BOUNDS)
    g = normalize(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize(band14, _T11_BOUNDS)
    return np.clip(np.stack([r, g, b], axis=2), 0, 1)

class ICRGWDataset(Dataset):
    def __init__(self, tar_path, ids, padding_size):
        self.tar_path = tar_path
        self.ids = ids
        self.padding_size = padding_size
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        N_TIMES_BEFORE = 4
        sample_path = f"{self.tar_path}/{self.ids[idx]}"
        band11 = np.load(f"{sample_path}/band_11.npy")[..., N_TIMES_BEFORE]
        band14 = np.load(f"{sample_path}/band_14.npy")[..., N_TIMES_BEFORE]
        band15 = np.load(f"{sample_path}/band_15.npy")[..., N_TIMES_BEFORE]
        image = false_color(band11, band14, band15)
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1)
        padding_size = self.padding_size
        image = F.pad(image, (padding_size, padding_size, padding_size, padding_size), mode='reflect')
        try:
            label = np.load(f"{sample_path}/human_pixel_masks.npy")
            label = torch.Tensor(label).to(torch.int64)
            label = label.permute(2, 0, 1)
        except FileNotFoundError:
            # label does not exist
            label = torch.zeros((1, image.shape[1], image.shape[2]))
        return image, label


if __name__ == "__main__":

    data_path = "./train" 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_ids = os.listdir(data_path)
        
    ids_train, ids_valid = train_test_split(image_ids, test_size=0.1, random_state=42)
    print(f"TrainSize: {len(ids_train)}, ValidSize: {len(ids_valid)}")

    batch_size = 8
    epochs = 1
    lr = 1e-5
    
    train_dataset = ICRGWDataset(data_path, ids_train, 2)
    valid_dataset = ICRGWDataset(data_path, ids_valid, 2)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=1)
    valid_dataloader = DataLoader(valid_dataset, 1, shuffle=None, num_workers=1)

    # Define model
    model = nn.Conv2d(3, 2, 1) # TODO: replace with your model
    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # train model
    bst_dice = 0
    for epoch in range(epochs):
        model.train()
        bar = tqdm(train_dataloader)
        tot_loss = 0
        tot_score = 0
        count = 0
        for X, y in bar:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = ce_loss(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tot_loss += loss.item()
            tot_score += dice_score(pred, y)
            count += 1
            bar.set_postfix(TrainLoss=f'{tot_loss/count:.4f}', TrainDice=f'{tot_score/count:.4f}')

        model.eval()
        bar = tqdm(valid_dataloader)
        tot_score = 0
        count = 0
        for X, y in bar:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            tot_score += dice_score(pred, y)
            count += 1
            bar.set_postfix(ValidDice=f'{tot_score/count:.4f}')

        if tot_score/count > bst_dice:
            bst_dice = tot_score/count
            torch.save(model.state_dict(), 'u-net.pth')
            print("current model saved!")

    # evaluate model on validation set and print results
    model.eval()
    tot_score = 0
    for X, y in valid_dataloader:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        tot_score += dice_score(pred, y)
    print(f"Validation Dice Score: {tot_score/len(valid_dataloader)}")


    # save predictions on test set to csv file suitable for submission
    submission = pd.read_csv('sample_submission.csv', index_col='record_id')

    test_dataset = ICRGWDataset("test/", os.listdir('test'), 2)
    for idx, (X, y) in enumerate(test_dataset):
        X = X.to(device)
        pred = model(X.unsqueeze(0))[:, :, 2:-2, 2:-2] # remove padding
        pred = torch.argmax(pred, dim=1)[0]
        pred = pred.detach().cpu().numpy()
        submission.loc[int(test_dataset.ids[idx]), 'encoded_pixels'] = list_to_string(rle_encode(pred))
    submission.to_csv('submission.csv')
    
