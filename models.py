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
    def __init__(self, input_shape, net_type="fc"):
        super().__init__()

        self.net_type = net_type
        self.num_channels = input_shape[0]
        self.image_height = input_shape[1]
        self.image_width = input_shape[2]

        if net_type == 'fc':
            in_features = self.num_channels * self.image_height * self.image_width
            hidden_size = 512
            self.encoder = nn.Sequential(
                nn.Linear(in_features, hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size//2, hidden_size//4),
                nn.ReLU(inplace=True)
            )
        elif net_type == 'conv':
            self.encoder = nn.Sequential(
                nn.Conv2d(self.num_channels, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 4, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )
        else:
            raise Exception("Wrong net_type")

    def forward(self, x):
        if self.net_type == 'fc':
            x = x.view(x.size(0), -1)
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, output_shape, net_type="fc"):
        super().__init__()

        self.net_type = net_type
        self.num_channels = output_shape[0]
        self.image_height = output_shape[1]
        self.image_width = output_shape[2]

        if net_type == 'fc':
            hidden_size = 512
            out_features = self.num_channels * self.image_height * self.image_width
            self.decoder = nn.Sequential(
                nn.Linear(hidden_size//4, hidden_size//2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size//2, hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, out_features),
                nn.Sigmoid()
            )
        elif net_type == 'conv':
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(4, 16, 2, stride=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(16, self.num_channels, 2, stride=2, output_padding=1),
                #nn.Upsample(size=(self.image_height, self.image_width)),
                nn.Sigmoid()
            )
        else:
            raise Exception("Wrong net_type")

    def forward(self, x):
        if self.net_type == 'fc':
            x = x.view(x.size(0), -1)
        return self.decoder(x).view(x.size(0), self.num_channels, self.image_height, self.image_width)


class VanillaAE(LightningModule):
    def __init__(self, input_shape, net_type='fc', dataset_name='mnist'):
        super().__init__()
        self.dataset_name = dataset_name
        self.encoder = Encoder(input_shape, net_type)
        self.decoder = Decoder(input_shape, net_type)

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
        transforms_ = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])

        if self.dataset_name == 'faces':
            dataset = Faces('images/faces/', transforms_)

            # Train-val splitting:
            indices = list(range(len(dataset)))
            split = int(np.floor(.2 * len(dataset)))

            train_indices = indices[split:]
            train_sampler = SubsetRandomSampler(train_indices)

            return DataLoader(dataset, batch_size=16, sampler=train_sampler)
            
        elif self.dataset_name == 'mnist':
            mnist_train = MNIST('images/mnist', train=True, download=True,
                            transform=transforms_)
            return DataLoader(mnist_train, batch_size=16)
    
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
        transforms_ = transforms.Compose([transforms.ToTensor()])

        if self.dataset_name == 'faces':
            dataset = Faces('images/faces/', transforms_)

            # Train-val splitting:
            indices = list(range(len(dataset)))
            split = int(np.floor(.2 * len(dataset)))

            val_indices = indices[:split]
            val_sampler = SubsetRandomSampler(val_indices)

            return DataLoader(dataset, batch_size=64, sampler=val_sampler)
            
        elif self.dataset_name == 'mnist':
            mnist_train = MNIST('images/mnist', train=True, download=True,
                            transform=transforms_)
            _, mnist_val = torch.utils.data.random_split(mnist_train, [55000, 5000])
            mnist_val = DataLoader(mnist_val, batch_size=64)
            return DataLoader(mnist_val, batch_size=64)
    
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