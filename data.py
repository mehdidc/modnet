import torch.utils.data as data
from PIL import Image
import torch
from torchvision.datasets import folder
from collections import defaultdict
import numpy as np
import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from sklearn.preprocessing import LabelEncoder
from torchvision.datasets import ImageFolder as _ImageFolder


class ImageFolderDataset(_ImageFolder):

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        label = torch.zeros(len(self.classes))
        label[target] = 1
        return sample, label, path


class ImageFilenamesDataset:

    def __init__(self, filenames, transform):
        self.filenames = filenames
        self.transform = transform

    def __getitem__(self, i):
        orig_im = default_loader(self.filenames[i])
        im = self.transform(orig_im)
        return orig_im, im, self.filenames[i]

    def __len__(self):
        return len(self.filenames)


class BalancedSample:

    def __init__(self, dataset, seed=None, nb=None, neg_classes=['background']):
        self.rng = np.random.RandomState(seed)
        self.dataset = dataset
        self.classes = dataset.classes
        class_indices = defaultdict(list)
        nb_neg = 0
        nb_pos = 0
        for i, (_, targets) in enumerate(dataset.samples):
            if len(targets) == 1 and targets[0] in neg_classes:
                nb_neg += 1
            else:
                nb_pos += 1
            for target in targets:
                class_indices[target].append(i)
        if nb is None:
            nb = nb_pos
        self.nb = nb
        self.class_indices = class_indices
        self.transform = dataset.transform

    def __getitem__(self, idx):
        cl = self.rng.choice(list(self.class_indices.keys()))
        idx = self.rng.choice(self.class_indices[cl])
        return self.dataset[idx]

    def __len__(self):
        return self.nb


class BalancedSampleTest:

    def __init__(self, dataset, neg_classes=['background'], nb_neg=None):
        self.dataset = dataset
        self.classes = dataset.classes
        class_indices = defaultdict(list)
        self.transform = dataset.transform

        pos_samples = []
        neg_samples = []
        for i, (_, targets) in enumerate(dataset.samples):
            if len(targets) == 1 and targets[0] in neg_classes:
                neg_samples.append(i)
            else:
                pos_samples.append(i)
                for target in targets:
                    class_indices[target].append(i)
        if nb_neg is None:
            nb_neg = len(pos_samples)
        neg_samples = neg_samples[0:nb_neg]
        self.samples = pos_samples + neg_samples

    def __getitem__(self, idx):
        idx_on_dataset = self.samples[idx]
        return self.dataset[idx_on_dataset]

    def __len__(self):
        return len(self.samples)


class CSVImageDataset:

    def __init__(self,
                 data_frame,
                 transform,
                 images_folder='images',
                 label_col='label',
                 image_id_col='image_id',
                 label_encoder=None,
                 ext='jpg',
                 seed=42):
        if not label_encoder:
            label_encoder = LabelEncoder()
            label_encoder.fit([
                label for labels in data_frame[label_col].values 
                for label in labels.split(';')])
        self.data_frame = data_frame
        self.label_encoder = label_encoder
        self.transform = transform
        self.label_col = label_col
        self.image_id_col = image_id_col
        self.images_folder = images_folder
        self.rng = np.random.RandomState(seed)
        self.classes = label_encoder.classes_
        self.ext = ext
        self.samples = [
                (image_id, labels.split(';'))
                for image_id, labels in data_frame[[image_id_col, label_col]].values
        ]

    def __getitem__(self, idx):
        image_id, labels = self.samples[idx]
        labels = self.label_encoder.transform(labels)
        label = torch.zeros(len(self.classes))
        for l in labels:
            label[l] = 1
        filename = os.path.join(self.images_folder, '{}.{}'.format(image_id, self.ext))
        img = folder.default_loader(filename)
        img = self.transform(img)
        return img, label, filename

    def __len__(self):
        return len(self.samples)


class SubSample:

    def __init__(self, dataset, nb):
        nb = min(len(dataset), nb)
        self.dataset = dataset
        self.nb = nb
        self.transform = dataset.transform
        self.classes = dataset.classes
        self.samples = dataset.samples
        self.transform = dataset.transform

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        return self.nb
