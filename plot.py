import matplotlib.pyplot as plt

import models
import torch
import os

def load_model(checkpoint_path, dataset, hidden_unit):
    if dataset == 'MNIST':
        model = models.Simple_FC(hidden_unit)
    elif dataset == 'CIFAR-10':
        model = models.FiveLayerCNN(hidden_unit)
    elif dataset == 'ResNet18':
        model = models.ResNet18(hidden_unit)
    else:
        raise NotImplementedError

    checkpoint = torch.load(os.path.join(checkpoint_path, 'Model_State_Dict_%d.pth' % hidden_unit))
    model.load_state_dict(checkpoint['net'])

    return model

if __name__ == '__main__':
    model = 'SimpleFC'
    parameters = []
    hidden_units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 25, 30, 35, 40, 45, 50, 55, 60, 70,
                    80, 90, 100, 120, 150, 200, 400, 600, 800, 1000]
    directory = f'assets/MNIST-SimpleFC/N=4000-3D/TEST-6/Epoch=4000-noise-20-model-0-sgd/ckpt'
    for n in hidden_units:
        model = load_model(directory, 'MNIST', n)
        parameters.append(sum(p.numel() for p in model.parameters()))

    fig = plt.figure(figsize=(5, 5))
    plt.title('Fully Connected Neural Network')
    plt.xlabel('Layer Width (k)')
    plt.ylabel('Number of Parameters (P)')
    plt.yscale('log')
    plt.plot(hidden_units, parameters)
    plt.grid()

    plt.savefig('images/SimpleFC.png')

    model = 'CNN'
    parameters = []
    hidden_units = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
    directory = f'assets/CIFAR-10-CNN/N=50000-3D/TEST-2/GS=100K-noise-20-model-0-sgd/ckpt'
    for n in hidden_units:
        model = load_model(directory, 'CIFAR-10', n)
        parameters.append(sum(p.numel() for p in model.parameters()))

    fig = plt.figure(figsize=(5, 5))
    plt.title('Five-layer CNNs')
    plt.xlabel('Layer Width (k)')
    plt.ylabel('Number of Parameters (P)')
    plt.yscale('log')
    plt.plot(hidden_units, parameters)
    plt.grid()

    plt.savefig('images/CNNs.png')


    model = 'ResNet18s'
    parameters = []
    hidden_units = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
    directory = f'assets/CIFAR-10-ResNet18/N=50000-3D/TEST-2/GS=100K-noise-20-model-0-sgd/ckpt'
    for n in hidden_units:
        model = load_model(directory, 'CIFAR-10', n)
        parameters.append(sum(p.numel() for p in model.parameters()))

    fig = plt.figure(figsize=(5, 5))
    plt.title('ResNet18s')
    plt.xlabel('Layer Width (k)')
    plt.ylabel('Number of Parameters (P)')
    plt.yscale('log')
    plt.plot(hidden_units, parameters)
    plt.grid()

    plt.savefig('images/ResNet18s.png')
