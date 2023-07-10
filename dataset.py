from torch.utils.data import Dataset, IterableDataset
import os
from PIL import Image
import pandas as pd
from torch import nn
import torch


class AnimeCeleb(Dataset):
    def __init__(self, exp_path, rot_path, csv_path, transform=None):
        self.exp_path = exp_path
        self.rot_path = rot_path
        self.transform = transform
        self.csv = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        y_folder_name = self.csv.iloc[index, 1]
        y_png_name = self.csv.iloc[index, 2]
        y_path = os.path.join(self.rot_path, str(y_folder_name).zfill(5), y_png_name)
        exp_path = os.path.join(self.exp_path, str(y_folder_name).zfill(5), y_png_name)
        y_id = self.csv.iloc[index, 0]

        # random choose from rows with the same id
        # print(self.csv.query(f"org_id == {y_id}").sample(n=1)['png_name'].iloc[0])
        x_ = self.csv.query(f"org_id == {y_id}").sample(n=1).astype(str)
        # print(x_)
        x_png_name = x_['png_name'].iloc[0]
        x_folder_name = x_['folder_name'].iloc[0]
        # print(str(x_folder_name).zfill(5))
        x_path = os.path.join(self.rot_path, str(x_folder_name).zfill(5), x_png_name)
        # print(x_path)
        if not os.path.exists(x_path):
            return None
        x = Image.open(x_path).convert("RGB")
        y = Image.open(y_path).convert("RGB")
        exp = Image.open(exp_path).convert("RGB")
        # x = 
        pose = self.csv.iloc[index, 3:]
        pose = torch.tensor(pose)
        pose[-3:] = (pose[-3:]+20)/40
        if self.transform is not None:
            # pass
            x = self.transform(x)
            y = self.transform(y)
            exp = self.transform(exp)

        # print(x.max(), x.min())
        return x, pose, y, exp



class AnimeCelebIter(IterableDataset):
    def __init__(self, exp_path, rot_path, csv_path, transform=None):
        self.exp_path = exp_path
        self.rot_path = rot_path
        self.transform = transform
        self.csv = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.csv)

    def __iter__(self):
        for index in range(len(self.csv)):
            y_folder_name = self.csv.iloc[index, 1]
            y_png_name = self.csv.iloc[index, 2]
            y_path = os.path.join(self.rot_path, str(y_folder_name).zfill(5), y_png_name)
            exp_path = os.path.join(self.exp_path, str(y_folder_name).zfill(5), y_png_name)
            y_id = self.csv.iloc[index, 0]

            # random choose from rows with the same id
            # print(self.csv.query(f"org_id == {y_id}").sample(n=1)['png_name'].iloc[0])
            x_ = self.csv.query(f"org_id == {y_id}").sample(n=1).astype(str)
            # print(x_)
            x_png_name = x_['png_name'].iloc[0]
            x_folder_name = x_['folder_name'].iloc[0]
            # print(str(x_folder_name).zfill(5))
            x_path = os.path.join(self.exp_path, str(x_folder_name).zfill(5), x_png_name)
            # print(x_path)
            if not os.path.exists(x_path):
                continue
            x = Image.open(x_path).convert("RGB")
            y = Image.open(y_path).convert("RGB")
            exp = Image.open(exp_path).convert("RGB")
            # x = 
            pose = self.csv.iloc[index, 3:]
            pose = torch.tensor(pose)
            pose[-3:] = (pose[-3:]+20)/40
            if self.transform is not None:
                # pass
                x = self.transform(x)
                y = self.transform(y)
                exp = self.transform(exp)

            # print(x.max(), x.min())
            yield x, pose, y, exp


        