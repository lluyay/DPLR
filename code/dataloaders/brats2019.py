import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler


class BraTS2019(Dataset):
    """ BraTS2019 Dataset """

    def __init__(self, base_dir=None, split='train', num=None, transform=None, pseudo_label_path=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        self.has_pseudo_label = []
        self.pseudo_label_path = pseudo_label_path
        self.split = split

        train_path = '/code/data/BraTS2019/train.txt'
        test_path = '/code/data/BraTS2019/val.txt'

        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        self.has_pseudo_label = np.zeros(len(self.image_list), dtype=bool)
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File("/data/YayuanLu/BraTS19/data/{}.h5".format(image_name), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label.astype(np.uint8)}
        if self.split == "train":
            if self.has_pseudo_label[idx] == True:
                h5f = h5py.File(self.pseudo_label_path + "/{}.h5".format(image_name), "r")
                pseudo_label1 = h5f["pseudo_label1"][:]
                pseudo_label2 = h5f["pseudo_label2"][:]
            else:
                pseudo_label1 = np.full_like(label, 6).astype(np.uint8)
                pseudo_label2 = np.full_like(label, 6).astype(np.uint8)
                h5f = h5py.File(self.pseudo_label_path + "/{}.h5".format(image_name), "w")
                h5f.create_dataset('pseudo_label1', data=pseudo_label1)
                h5f.create_dataset('pseudo_label2', data=pseudo_label2)
                h5f.close()
            sample["pseudo_label1"] = pseudo_label1
            sample["pseudo_label2"] = pseudo_label2
            k = -1
            axis = -1
            w1 = -1
            h1 = -1
            d1 = -1
            aug = np.array([int(k), int(axis), int(w1), int(h1), int(d1)])
            sample["aug"] = aug
        if self.transform:
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample
    
    def update_labels(self, idx, aug, pseudo_label1_crop, pseudo_label2_crop, pseudo_label_path):
        for i, item in enumerate(idx):
            if self.has_pseudo_label[item] == False:
                self.has_pseudo_label[item] = True
            case = self.image_list[item]

            k = aug[i][0]
            axis = aug[i][1]
            w1 = aug[i][2]
            h1 = aug[i][2]
            d1 = aug[i][3]
            h5f = h5py.File(pseudo_label_path + "/{}.h5".format(case), "r")
            pseudo_label1 = h5f["pseudo_label1"][:]
            pseudo_label2 = h5f["pseudo_label2"][:]
            h5f.close()
            data1 = pseudo_label1_crop[i].cpu().numpy()
            data2 = pseudo_label2_crop[i].cpu().numpy()
            if axis != -1:
                data1 = np.flip(data1, axis=axis)
                data2 = np.flip(data2, axis=axis)
            if k != -1:
                k = 4 - k
                data1 = np.rot90(data1, k)
                data2 = np.rot90(data2, k)
            if w1 != -1:
                pseudo_label1[w1:w1 + 96, h1:h1 + 96, d1:d1 + 96] = data1
                pseudo_label2[w1:w1 + 96, h1:h1 + 96, d1:d1 + 96] = data2
            h5f = h5py.File(pseudo_label_path + "/{}.h5".format(case), "w")
            h5f.create_dataset('pseudo_label1', data=pseudo_label1)
            h5f.create_dataset('pseudo_label2', data=pseudo_label2)
            h5f.close()


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label, pseudo_label1, pseudo_label2, aug = sample["image"], sample["label"], sample["pseudo_label1"], sample["pseudo_label2"], sample["aug"]
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            pseudo_label1 = np.pad(pseudo_label1, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            pseudo_label2 = np.pad(pseudo_label2, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)],
                             mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])
        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        pseudo_label1 = pseudo_label1[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        pseudo_label2 = pseudo_label2[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'pseudo_label1': pseudo_label1, 'pseudo_label2' : pseudo_label2, 'sdf': sdf}
        else:
            return {'image': image, 'label': label, 'pseudo_label1': pseudo_label1, 'pseudo_label2' : pseudo_label2, "aug": aug}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label, pseudo_label1, pseudo_label2, aug = sample["image"], sample["label"], sample["pseudo_label1"], sample["pseudo_label2"], sample["aug"]
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        pseudo_label1 = np.rot90(pseudo_label1, k)
        pseudo_label2 = np.rot90(pseudo_label2, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        pseudo_label1 = np.flip(pseudo_label1, axis=axis).copy()
        pseudo_label2 = np.flip(pseudo_label2, axis=axis).copy()
        aug[0] = k
        aug[1] = axis
        return {'image': image, 'label': label, 'pseudo_label1': pseudo_label1, 'pseudo_label2' : pseudo_label2, "aug" : aug}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(
            image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros(
            (self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label, 'onehot_label': onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(
            1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(), 'pseudo_label1': torch.from_numpy(sample['pseudo_label1']).long(), 'pseudo_label2': torch.from_numpy(sample['pseudo_label2']).long(), "aug": sample["aug"]}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)