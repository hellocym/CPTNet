from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd
from torch import nn
import torch


class AnimeCeleb(Dataset):
    def __init__(self, root, csv_path, transform=None):
        self.root = root
        self.transform = transform
        self.imgs = os.listdir(root)
        self.csv = pd.read_csv(csv_path)[:16900]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        y_path = os.path.join(self.root, self.csv.iloc[index, 2])
        y_id = self.csv.iloc[index, 0]
        
        # random choose from rows with the same id
        # print(self.csv.query(f"org_id == {y_id}").sample(n=1)['png_name'].iloc[0])
        x_path = os.path.join(self.root, self.csv.query(f"org_id == {y_id}").sample(n=1)['png_name'].iloc[0])
        x = Image.open(x_path).convert("RGB")
        y = Image.open(y_path).convert("RGB")
        pose = self.csv.iloc[index, 3:]
        pose = torch.tensor(pose)
        pose[-3:] = (pose[-3:]+20)/40
        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)

        # print(x.max(), x.min())
        return x, pose, y



        