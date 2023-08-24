import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import os

class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.data = []
        self.targets = []

        for i in range(len(data_list)):
            self.data.append(data_list[i][0])
            self.targets.append(data_list[i][1])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

    def get_list(self):
        list = []
        for i in range(self.__len__()):
            list.append([self.data[i], int(self.targets[i])])

        return list


def get_train_dataset(DATASET):
    if DATASET == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        return datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    elif DATASET == 'CIFAR-10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        return datasets.CIFAR10(root='./data/CIFAR-10', train=True, download=True, transform=transform)
    else:
        raise NotImplementedError


def get_test_dataset(DATASET):
    if DATASET == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        return datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif DATASET == 'CIFAR-10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        return datasets.CIFAR10(root='./data/CIFAR-10', train=False, download=True, transform=transform)
    else:
        raise NotImplementedError


def save_and_create_training_set(DATASET, sample_size, label_noise_ratio, dataset_path):
    clean_dataset_path = os.path.join(dataset_path, 'subset-clean.pth')

    if not os.path.exists(clean_dataset_path):
        print('Saving Clean Dataset...')
        train_dataset = get_train_dataset(DATASET)
        train_dataset = torch.utils.data.Subset(train_dataset, indices=np.arange(sample_size))

        torch.save(list(train_dataset), clean_dataset_path)

    if label_noise_ratio == 0:
        print('Loading Clean Dataset...')
        return torch.load(clean_dataset_path)

    elif label_noise_ratio > 0:
        noisy_dataset_path = os.path.join(dataset_path, 'subset-noise-%d%%.pth' % (100 * label_noise_ratio))

        if not os.path.exists(noisy_dataset_path):
            print('Generating Noisy Dataset...')
            train_dataset_2_list = torch.load(clean_dataset_path)
            train_dataset_2 = ListDataset(train_dataset_2_list)

            label_noise_transform = transforms.Lambda(lambda y: torch.tensor(np.random.randint(0, 10)))
            num_noisy_samples = int(label_noise_ratio * len(train_dataset_2))

            noisy_indices = np.random.choice(len(train_dataset_2), num_noisy_samples, replace=False)
            for idx in noisy_indices:
                train_dataset_2.targets[idx] = label_noise_transform(train_dataset_2.targets[idx])

            print("%d Label Noise added to Train Data;\n" % (label_noise_ratio * 100))

            torch.save(train_dataset_2.get_list(),
                       os.path.join(dataset_path, 'subset-noise-%d%%.pth' % (100 * label_noise_ratio)))

        print('Loading Noisy Dataset...')
        return torch.load(noisy_dataset_path)

