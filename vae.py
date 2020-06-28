import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms

from pytorch_lightning.core import LightningModule

from datasets import Faces

# def vae_loss(rec_x, x, mu, logsigma):
#     KLD = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp(), dim=1)
#     BCE = F.binary_cross_entropy(rec_x.view(x.size(0), -1), x.view(x.size(0), -1), reduction='none').sum(1)
#     return (-KLD + BCE).mean()

def vae_loss(rec_x, x, mu, logsigma):
    #num_features = x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]
    KLD = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())

    x = x.view(x.shape[0], -1)
    rec_x = rec_x.view(rec_x.shape[0], -1)
    #BCE = F.binary_cross_entropy(rec_x, x)
    MSE = F.mse_loss(rec_x, x, reduction='sum')
    #BCE = torch.nn.functional.binary_cross_entropy(rec_x.view(x.shape), x, reduction='none').sum(dim=(1,2,3)).mean()
    #return (KLD + BCE)/2/num_features
    return KLD + MSE

class Encoder(nn.Module):
    def __init__(self, input_shape=(3,45,45), hidden_size=512, latent_dim=100, dropout_rate=0.3):
        super().__init__()
        self.num_channels = input_shape[0]
        self.image_height = input_shape[1]
        self.image_width = input_shape[2]
        #in_features = self.num_channels * self.image_height * self.image_width

        # Encoder
        self.conv = nn.Sequential(
            nn.Conv2d(self.num_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.image_height * self.image_width, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True)
        )
        # Latent vectors mu and sigma
        self.mean = nn.Linear(2048, 2048)
        self.logvar = nn.Linear(2048, 2048)

    def forward(self, x):
        conv = self.conv(x).view(-1, self.image_height * self.image_width)
        fc = self.fc(conv)

        mu = self.mean(fc)
        logvar = self.logvar(fc)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, output_shape, hidden_size=512, latent_dim=100, dropout_rate=0.3):
        super().__init__()
        self.num_channels = output_shape[0]
        self.image_height = output_shape[1]
        self.image_width = output_shape[2]
        #out_features = self.num_channels * self.image_height * self.image_width

        self.fc = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, self.image_height * self.image_width),
            nn.BatchNorm1d(self.image_height * self.image_width),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, self.num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        fc = self.fc(x).view(-1, 16, self.image_height//4, self.image_width//4)
        conv = self.conv(fc).view(-1, self.num_channels, self.image_height, self.image_width)
        return torch.sigmoid(conv)


class VAE(LightningModule):
    def __init__(self, input_shape, dataset_name='mnist'):
        super().__init__()
        self.input_shape = input_shape
        self.dataset_name = dataset_name
        self.encoder = Encoder(input_shape)
        self.decoder = Decoder(input_shape)


        # cache for generated images
        self.generated_images = None
        self.last_images = None

    def sampler(self, mu, logsigma):
        #std = torch.exp(0.5*logsigma)
        if self.training:
            eps = torch.randn_like(logsigma)
            return mu + eps * torch.exp(0.5*logsigma)
        else:
            return mu

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        z = self.sampler(mu, logsigma)
        x = self.decoder(z)
        return x, mu, logsigma

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    ###########
    ## Train ##
    ###########
    def train_dataloader(self):
        if self.dataset_name == 'faces':
            #transforms_ = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.,), (1.,))])
            transforms_ = transforms.Compose([transforms.ToTensor()])
            dataset =  Faces('images/faces/', transforms_=transforms_, img_height=self.input_shape[1], img_width=self.input_shape[2])

            # Train-val splitting:
            indices = list(range(len(dataset)))
            split = int(np.floor(.2 * len(dataset)))

            train_indices = indices[split:]
            train_sampler = SubsetRandomSampler(train_indices)

            return DataLoader(dataset, batch_size=16, sampler=train_sampler)
            
        elif self.dataset_name == 'mnist':
            # transforms_ = transforms.Compose([transforms.ToTensor(),
            #                           transforms.Normalize((0.1307,), (0.3081,))])
            transforms_ = transforms.Compose([transforms.ToTensor()])
            mnist_train = MNIST('images/mnist', train=True, download=True,
                            transform=transforms_)
            return DataLoader(mnist_train, batch_size=16, shuffle=True)
    
    def training_step(self, batch, batch_idx):
        images, _ = batch
        #images = images.view(images.size(0), -1)
        rec_images, mu, logsigma = self(images)
        loss = vae_loss(rec_images, images, mu, logsigma)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
        
    ################
    ## Validation ##
    ################
    def val_dataloader(self):
        if self.dataset_name == 'faces':
            #transforms_ = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.,), (1.,))])
            transforms_ = transforms.Compose([transforms.ToTensor()])
            dataset = Faces('images/faces/', transforms_=transforms_, img_height=self.input_shape[1], img_width=self.input_shape[2])

            # Train-val splitting:
            indices = list(range(len(dataset)))
            split = int(np.floor(.2 * len(dataset)))

            val_indices = indices[:split]
            val_sampler = SubsetRandomSampler(val_indices)

            return DataLoader(dataset, batch_size=16, sampler=val_sampler)
            
        elif self.dataset_name == 'mnist':
            # transforms_ = transforms.Compose([transforms.ToTensor(),
            #                           transforms.Normalize((0.1307,), (0.3081,))])
            transforms_ = transforms.Compose([transforms.ToTensor()])
            mnist_train = MNIST('images/mnist', train=True, download=True,
                            transform=transforms_)
            _, mnist_val = torch.utils.data.random_split(mnist_train, [55000, 5000])
            mnist_val = DataLoader(mnist_val, batch_size=16, shuffle=True)
            return mnist_val
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}

        real_grid = torchvision.utils.make_grid(self.last_images)
        rec_grid = torchvision.utils.make_grid(self.generated_images)
        image_grid = torch.cat((real_grid, rec_grid), 1)
        self.logger.experiment.add_image('recovered_images', image_grid, self.current_epoch)

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        #images = images.view(images.size(0), -1)
        rec_images, mu, logsigma = self(images)  
        loss = vae_loss(rec_images, images, mu, logsigma)

        self.last_images = images[:5]
        self.generated_images = rec_images.view(images.shape)[:5]

        return {'val_loss': loss}