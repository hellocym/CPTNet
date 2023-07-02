import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from models import StarGANDiscriminator, CPTNet, initialize_weights


# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Running on {device}')

INIT_LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_STEPS = 600000
IMAGE_SIZE = 256

# Load Data
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],  # mean
            [0.5, 0.5, 0.5],  # std
        ),
    ]
)

# dataset = ImageFolder(root="data", transform=transform)
# loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

G = CPTNet(IMAGE_SIZE, IMAGE_SIZE).to(device)
D = StarGANDiscriminator(IMAGE_SIZE, IMAGE_SIZE).to(device)
initialize_weights(G)
initialize_weights(D)

# Optimizers
optim_G = optim.Adam(G.parameters(), lr=INIT_LEARNING_RATE, betas=(0.5, 0.999))
optim_D = optim.Adam(D.parameters(), lr=INIT_LEARNING_RATE, betas=(0.5, 0.999))

# train for 600000 iterations and linearly decay the learning rate to zero over the last 100000 iterations
lr_scheduler_G = optim.lr_scheduler.LambdaLR(
    optim_G, lr_lambda=lambda step: 1.0 - max(0, step - 500000) / 100000
)
lr_scheduler_D = optim.lr_scheduler.LambdaLR(
    optim_D, lr_lambda=lambda step: 1.0 - max(0, step - 500000) / 100000
)