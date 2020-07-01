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

class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim=256):
        super().__init__()

        self.num_channels = input_shape[0]
        self.image_height = input_shape[1]
        self.image_width = input_shape[2]
        
        in_features = self.num_channels * self.image_height * self.image_width
        self.encoder = nn.Sequential(
            nn.Linear(in_features, latent_dim),
            #nn.ReLU(inplace=True),
            #nn.Linear(hidden_size, hidden_size//2),
            #nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, output_shape, latent_dim=256):
        super().__init__()

        self.num_channels = output_shape[0]
        self.image_height = output_shape[1]
        self.image_width = output_shape[2]

        latent_dim = 256
        out_features = self.num_channels * self.image_height * self.image_width
        self.decoder = nn.Sequential(
            #nn.Linear(hidden_size//2, hidden_size),
            #nn.ReLU(inplace=True),
            nn.Linear(latent_dim, out_features)
            #nn.Sigmoid()
        )


    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.decoder(x).view(x.size(0), self.num_channels, self.image_height, self.image_width)


class VanillaAE(LightningModule):
    def __init__(self, input_shape, latent_dim=256, dataset_name='mnist'):
        super().__init__()
        self.dataset_name = dataset_name
        self.input_shape = input_shape

        self.encoder = Encoder(input_shape, latent_dim)
        self.decoder = Decoder(input_shape, latent_dim)

        # cache for generated images
        self.generated_images = None
        self.last_images = None

    def forward(self, x):
        lv = self.encoder(x)
        x = self.decoder(lv)
        return lv, x

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    ###########
    ## Train ##
    ###########
    def train_dataloader(self):
        
        if self.dataset_name == 'faces':
            transforms_ = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.,), (1.,))])
            dataset = Faces('images/faces/', transforms_=transforms_, img_height=self.input_shape[1], img_width=self.input_shape[2])

            # Train-val splitting:
            indices = list(range(len(dataset)))
            split = int(np.floor(.2 * len(dataset)))

            train_indices = indices[split:]
            train_sampler = SubsetRandomSampler(train_indices)

            return DataLoader(dataset, batch_size=16, sampler=train_sampler, drop_last=True)
            
        elif self.dataset_name == 'mnist':
            transforms_ = transforms.Compose([transforms.Resize((self.input_shape[1], self.input_shape[2])),
                                              transforms.ToTensor()]) 
            mnist_train = MNIST('images/mnist', train=True, download=True,
                            transform=transforms_)
            return DataLoader(mnist_train, batch_size=16, drop_last=True)
    
    def training_step(self, batch, batch_idx):
        images, _ = batch
        #images = images.view(images.size(0), -1)
        _, rec_images = self(images)  
        loss = F.mse_loss(rec_images, images)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
        
    ################
    ## Validation ##
    ################
    def val_dataloader(self):

        if self.dataset_name == 'faces':
            transforms_ = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.,), (1.,))])
            dataset = Faces('images/faces/', transforms_=transforms_, img_height=self.input_shape[1], img_width=self.input_shape[2])

            # Train-val splitting:
            indices = list(range(len(dataset)))
            split = int(np.floor(.2 * len(dataset)))

            val_indices = indices[:split]
            val_sampler = SubsetRandomSampler(val_indices)

            return DataLoader(dataset, batch_size=5, sampler=val_sampler, drop_last=True)
            
        elif self.dataset_name == 'mnist':
            transforms_ = transforms.Compose([transforms.Resize((self.input_shape[1], self.input_shape[2])),
                                              transforms.ToTensor()]) 
                                              #transforms.Normalize((0.1307,), (0.3081,))])
            mnist_train = MNIST('images/mnist', train=True, download=True,
                            transform=transforms_)
            _, mnist_val = torch.utils.data.random_split(mnist_train, [55000, 5000])
            mnist_val = DataLoader(mnist_val, batch_size=5, drop_last=True, shuffle=True)
            return mnist_val
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}

        # TODO: Fix this stuff with maybe hparams? It won't work for MNIST.
        #real_grid = torchvision.utils.make_grid(self.last_images.view(self.last_images.size(0), 3, 45, 45))
        #rec_grid = torchvision.utils.make_grid(self.generated_images.view(self.generated_images.size(0), 3, 45, 45))
        real_grid = torchvision.utils.make_grid(self.last_images)
        rec_grid = torchvision.utils.make_grid(self.generated_images)
        image_grid = torch.cat((real_grid, rec_grid), 1)
        self.logger.experiment.add_image('recovered_images', image_grid, self.current_epoch)

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        #images = images.view(images.size(0), -1)
        _, rec_images = self(images)
        loss = F.mse_loss(rec_images, images)
      
        self.last_images = images[:5]
        self.generated_images = rec_images[:5]

        return {'val_loss': loss}