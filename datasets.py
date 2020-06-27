import glob
import random
import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

from PIL import Image
from imageio import imread


# Move next three funcs to utils.py? 
def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

def imresize(img, size):
    return np.array(Image.fromarray(img).resize(size))

def fetch_dataset(
                    path_to_data='images/faces/', 
                    attrs_name = "lfw_attributes.txt",
                    images_name = "lfw-deepfunneled",
                    dx=80, dy=80,
                    dimx=32, dimy=32
    ):
    
    #download if not exists
    if not os.path.exists(path_to_data + images_name):
        print("images not found, downloading...")
        os.system("wget -P %s http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz -O tmp.tgz" % path_to_data)
        print("extracting...")
        os.system("tar xvzf %s/tmp.tgz && rm images/faces/tmp.tgz" % path_to_data)
        print("done")
        assert os.path.exists(path_to_data + images_name)

    if not os.path.exists(path_to_data + attrs_name):
        print("attributes not found, downloading...")
        os.system("wget -P %s http://www.cs.columbia.edu/CAVE/databases/pubfig/download/%s" % (path_to_data, attrs_name))
        print("done")

    #read attrs
    df_attrs = pd.read_csv("%slfw_attributes.txt" % path_to_data, sep='\t', skiprows=1,) 
    df_attrs = pd.DataFrame(df_attrs.iloc[:,:-1].values, columns = df_attrs.columns[1:])


    #read photos
    photo_ids = []
    for dirpath, _, filenames in os.walk(path_to_data + images_name):
        for fname in filenames:
            if fname.endswith(".jpg"):
                fpath = os.path.join(dirpath,fname)
                photo_id = fname[:-4].replace('_', ' ').split()
                person_id = ' '.join(photo_id[:-1])
                photo_number = int(photo_id[-1])
                photo_ids.append({'person':person_id, 'imagenum':photo_number, 'photo_path':fpath})

    photo_ids = pd.DataFrame(photo_ids)
    # print(photo_ids)
    #mass-merge
    #(photos now have same order as attributes)
    df = pd.merge(df_attrs, photo_ids, on=('person','imagenum'))

    assert len(df)==len(df_attrs), "lost some data when merging dataframes"

    # print(df.shape)
    #image preprocessing
    all_photos = df['photo_path'].apply(imread)\
                                .apply(lambda img: img[dy:-dy, dx:-dx])\
                                .apply(lambda img: imresize(img, [dimx, dimy]))

    all_photos = np.stack(all_photos.values).astype('uint8')
    all_attrs = df.drop(["photo_path","person","imagenum"], axis=1)

    return all_photos, all_attrs



class Faces(Dataset):

    def __init__(self, path_to_data, img_height=45, img_width=45, transforms_=None):
        data = fetch_dataset(path_to_data=path_to_data, dimx=img_height, dimy=img_width)

        self.transform = transforms_

        self.images = data[0]
        self.attributes = data[1]

    def __getitem__(self, index):
        index = index % len(self.images)
        
        image = Image.fromarray(self.images[index].astype('uint8'), 'RGB')
        attributes = self.attributes.iloc[index].to_dict()
        
        if image.mode != "RGB":
            image = to_rgb(image)

        if self.transform != None:
            image = self.transform(image)

        return image, attributes

    def __len__(self):
        return len(self.images)