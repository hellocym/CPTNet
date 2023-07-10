import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from models import StarGANDiscriminator, CPTNet, LossNetwork, initialize_weights
from dataset import AnimeCeleb, AnimeCelebIter
from utils import GANLoss
import os


# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(f'Running on {device}')

INIT_LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_STEPS = 600000
IMAGE_SIZE = 256
LAMBDA_1 = 1000
LAMBDA_2 = 200
FEATURE_SIZE = 20

# Load Data
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     [0.5, 0.5, 0.5],  # mean
        #     [0.5, 0.5, 0.5],  # std
        # ),
    ]
)

# dataset = AnimeCeleb(exp_path='./expression/', rot_path='./rotation/', csv_path='./data/data.csv', transform=transform)
dataset = AnimeCelebIter(exp_path='./expression/', rot_path='./rotation/', csv_path='./data/data.csv', transform=transform)
# loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=True)
# print(len(loader))

G = CPTNet(IMAGE_SIZE, IMAGE_SIZE, FEATURE_SIZE).to(device)
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

# Loss function
criterion = GANLoss('vanilla').to(device)

# Tensorboard
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
writer_fake_exp = SummaryWriter(f"logs/fake_exp")
writer_D = SummaryWriter(f"logs/D")
writer_G = SummaryWriter(f"logs/G")


# Training Loop
step = 0
for epoch in range(NUM_STEPS//BATCH_SIZE):
    for batch_idx, (x, pose, real, real_exp) in enumerate(loader):
        # if x is None:
            # continue
        # print(f'\rbatch:{batch_idx}', end='')
        # pass
        real = real.to(device)
        real_exp = real_exp.to(device)
        pose = pose.to(device)
        x = x.to(device)
        fake_exp, fake = G(x, pose)

        # calc adversarial loss
        # L_adv = E[log(D(G(real, pose)))] - E[log(D(real))] 
        # real_loss = criterion(D(real), torch.zeros_like(D(real)))
        # fake_loss = criterion(D(fake.detach()), torch.ones_like(D(fake)))
        # loss_adv = (real_loss + fake_loss) / 2
        loss_adv = criterion(D(fake.detach()), True)
        # calc content loss
        # L_pair = E[||G(real, pose) - real||], where ||.|| is L1 norm
        loss_pair_ = nn.L1Loss()(fake, real)
        loss_pair_exp = nn.L1Loss()(fake_exp, real_exp)
        loss_pair = (loss_pair_ + loss_pair_exp) / 2
        # calc perceptual loss
        # Only a L1Loss constraint on the generated image and ground truth may cause the image to be blurred.
        # So we adopt the perceptual loss [26] as another constraint.
        # We let the generated image and its corresponding ground truth pass through the pre-trained VGG19 network [27],
        # and extract the features of conv1 1, conv2 1, conv3 1, and conv4 2 layers for L1Loss, and finally weighted summation.
        # L_p = sum_j(E[||VGG_j(G(real, pose)) - VGG_j(real)||]), where ||.|| is L1 norm, and VGG_j is the j-th layer of VGG19
        loss_p = 0
        VGG = torchvision.models.vgg19(pretrained=True)
        VGG = VGG.to(device)
        loss_network = LossNetwork(VGG)
        loss_network.eval()
        # print(loss_network(fake))
        conv_1_1 = nn.L1Loss()(loss_network(fake)[0], loss_network(real)[0])
        conv_2_1 = nn.L1Loss()(loss_network(fake)[1], loss_network(real)[1])
        conv_3_1 = nn.L1Loss()(loss_network(fake)[2], loss_network(real)[2])
        conv_4_2 = nn.L1Loss()(loss_network(fake)[3], loss_network(real)[3])
        loss_p = conv_1_1 + conv_2_1 + conv_3_1 + conv_4_2
            
        # calc full loss
        # L_full = L_adv + lambda_1 * L_pair + lambda_2 * L_p
        loss_full_G = loss_adv + LAMBDA_1 * loss_pair + LAMBDA_2 * loss_p

        # optimize G
        optim_G.zero_grad()
        loss_full_G.backward()
        optim_G.step()

        real_loss = criterion(D(real), True)
        fake_loss = criterion(D(fake.detach()), False)
        loss_full_D = (real_loss + fake_loss) / 2
        
        # optimize D
        optim_D.zero_grad()
        loss_full_D.backward()
        optim_D.step()

        # print losses
        if batch_idx % 1 == 0:
            # pass
            print(
                f"\rEpoch [{epoch}] ,Step {step}/{len(loader)} \
                  Loss D: {loss_full_D:.4f}, Loss G: {loss_full_G:.4f},\
                  Loss G_adv = {loss_adv:.4f}, Loss G_pair = {loss_pair:.4f}, Loss G_p = {loss_p:.4f}\
                  ", end=""
            )
        if batch_idx % 5 == 0:
            with torch.no_grad():
                rand_pose = torch.rand_like(pose)
                fake_exp, fake = G(x, rand_pose)
                # # print(fake_[0].shape)
                img_grid_real = torchvision.utils.make_grid([x[i] for i in range(4)])
                img_grid_fake_exp = torchvision.utils.make_grid([fake_exp[i] for i in range(4)])
                img_grid_fake = torchvision.utils.make_grid([fake[i] for i in range(4)])
                writer_fake_exp.add_image("Fake_Expression", img_grid_fake_exp, global_step=step)
                writer_fake.add_image("Fake_Rotation", img_grid_fake, global_step=step)
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_D.add_scalar("Loss", loss_full_D, global_step=step)
                writer_G.add_scalar("Loss", loss_full_G, global_step=step)
                writer_G.add_scalar("Loss_adv", loss_adv, global_step=step)
                writer_G.add_scalar("Loss_pair", loss_pair, global_step=step)
                writer_G.add_scalar("Loss_p", loss_p, global_step=step)
            step += 1
            # pass
            # update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D.step()

            # save models
            if step % 100 == 0:
                if not os.path.exists('saved_models'):
                    os.makedirs('saved_models')
                torch.save(G.state_dict(), f"saved_models/G_{step}.pth")
                torch.save(D.state_dict(), f"saved_models/D_{step}.pth")

# save final model
torch.save(G.state_dict(), f"saved_models/G_final.pth")
torch.save(D.state_dict(), f"saved_models/D_final.pth")

# close tensorboard writer
writer_real.close()
writer_fake.close()


           