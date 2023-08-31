import torch
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import csv
import os

from datetime import datetime
import numpy as np
import random

import models
import datasets


# ------------------------------------------------------------------------------------------


# Training Settings
DATASET = 'MNIST'
MODEL = 'SimpleFC'
N_SAMPLES = 4000

#DATASET = 'CIFAR-10'
#MODEL = 'CNN'
#N_SAMPLES = 50000

TEST_GROUP = 0
TEST_NUMBERS = [0]
label_noise_ratio = 0.0

BATCH_SIZE = 128
save_model = True

test_gap = 10 * 1000

# ------------------------------------------------------------------------------------------


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

# Return the train_dataloader and test_dataloader of MINST
def get_train_and_test_dataloader(dataset_path):
    if label_noise_ratio > 0:
        train_dataset = datasets.save_and_create_training_set(DATASET, sample_size=N_SAMPLES,
                label_noise_ratio=label_noise_ratio, dataset_path=dataset_path)
    elif label_noise_ratio == 0:
        train_dataset = datasets.get_train_dataset(DATASET=DATASET)
    else:
        raise NotImplementedError

    train_dataloader = DataLoaderX(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    test_dataset = datasets.get_test_dataset(DATASET=DATASET)

    test_dataloader = DataLoaderX(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print(f'Load {DATASET} dataset success;')

    return train_dataloader, test_dataloader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ------------------------------------------------------------------------------------------


# Set the neural network model to be used
def get_model(hidden_unit, device):
    if DATASET == 'MNIST':
        model = models.Simple_FC(hidden_unit)
    elif DATASET == 'CIFAR-10':
        model = models.FiveLayerCNN(hidden_unit)
    elif DATASET == 'ResNet18':
        model = models.ResNet18(hidden_unit)

    model = model.to(device)

    print(f"\n{MODEL} Model with %d hidden neurons successfully generated;" % hidden_unit)

    print('Number of parameters: %d' % sum(p.numel() for p in model.parameters()))

    return model


# ------------------------------------------------------------------------------------------


def model_save(model, epoch, test_accuracy, checkpoint_path):
    state = {
        'net': model.state_dict(),
        'acc': test_accuracy,
        'epoch': epoch,
    }

    torch.save(state, os.path.join(checkpoint_path, 'Model_State_Dict_%d.pth' % hidden_unit))
    print("Torch saved successfully!\n")

def status_save(n_hidden_units, epoch, parameters, train_loss, train_acc, test_loss, test_acc, lr, time,
                dictionary_path):
    print("Hidden Neurons : %d ; Parameters : %d ; Train Loss : %f ; Train Acc : %.3f ; Test Loss : %f ; "
          "Test Acc : %.3f\n" % (n_hidden_units, parameters, train_loss, train_acc, test_loss, test_acc))

    print('Writing to a csv file...')
    dictionary = {'Hidden Neurons': hidden_unit, 'Epoch': epoch, 'Parameters': parameters,
                  'Train Loss': train_loss, 'Train Accuracy': train_acc, 'Test Loss': test_loss, 'Test Accuracy': test_acc,
                  'Learning Rate': lr, 'Time Cost': time}

    with open(dictionary_path, "a", newline="") as fp:
        # Create a writer object
        writer = csv.DictWriter(fp, fieldnames=dictionary.keys())

        # Write the data rows
        writer.writerow(dictionary)
        print('Done writing to a csv file\n')


# ------------------------------------------------------------------------------------------


# Train and Evalute the model
def train_and_evaluate_model(model, device, train_dataloader, test_dataloader, optimizer, criterion, dictionary_path, checkpoint_path):
    start_time = datetime.now()

    parameters = sum(p.numel() for p in model.parameters())
    n_hidden_units = model.n_hidden_units

    for epoch in range(1, 4000 + 1):
        # Model Training
        model.train()
        cumulative_loss, correct, total, idx = 0.0, 0, 0, 0

        for idx, (inputs, labels) in enumerate(train_dataloader):
            labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            cumulative_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.argmax(1)).sum().item()

        train_loss = cumulative_loss / (idx + 1)
        train_acc = correct / total

        lr = optimizer.param_groups[0]['lr']

        print("Epoch : %d ; Train Loss : %f ; Train Acc : %.3f ; Learning Rate : %f" %
              (epoch, train_loss, train_acc, lr))

        # Test Model after every 10K Gradient Steps
        if epoch % 400 == 0:
            # Test Model
            model.eval()
            cumulative_loss, correct, total, idx = 0.0, 0, 0, 0

            with torch.no_grad():
                for idx, (inputs, labels) in enumerate(test_dataloader):
                    labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    cumulative_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.argmax(1)).sum().item()

            test_loss = cumulative_loss / (idx + 1)
            test_acc = correct / total

            curr_time = datetime.now()
            time = (curr_time - start_time).seconds / 60

            status_save(n_hidden_units, epoch, parameters, train_loss, train_acc, test_loss, test_acc, lr, time,
                            dictionary_path=dictionary_path)

    if save_model:
        model_save(model, test_acc, epoch, checkpoint_path=checkpoint_path)

    return


# ------------------------------------------------------------------------------------------


if __name__ == '__main__':
    # Initialization of used device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    print('Using device : ', torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))

    # Initialization of hidden units
    if MODEL == 'SimpleFC':
        hidden_units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 25, 30, 35, 40, 45, 50, 55, 60, 70,
                        80, 90, 100, 120, 150, 200, 400, 600, 800, 1000]
    elif MODEL == 'CNN' or MODEL == 'ResNet18':
        hidden_units = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
    else:
        raise NotImplementedError

    # Main Program
    for test_number in TEST_NUMBERS:
        # Setup seed for reproduction
        setup_seed(20 + test_number)

        # Define the roots and paths
        directory = f"assets/{DATASET}-{MODEL}/N=%d-3d/TEST-%d/Epoch=4000-noise-%d-model-%d-adam" \
                    % (N_SAMPLES, TEST_GROUP, label_noise_ratio * 100, test_number)

        dictionary_path = os.path.join(directory, "dictionary.csv")
        dataset_path = os.path.join(directory, 'dataset')
        checkpoint_path = os.path.join(directory, "ckpt")

        if not os.path.isdir(directory):
            os.mkdir(directory)
        if not os.path.isdir(dataset_path):
            os.mkdir(dataset_path)
        if save_model and not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)

        # Initialize Status Dictionary
        dictionary = {'Hidden Neurons': 0, 'Epoch': 0, 'Parameters': 0, 'Train Loss': 0,
                      'Train Accuracy': 0, 'Test Loss': 0, 'Test Accuracy': 0, 'Learning Rate': 0, 'Time Cost': 0}

        with open(dictionary_path, "a", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=dictionary.keys())
            writer.writeheader()

        # Get the training and testing data of specific sample size
        train_dataloader, test_dataloader = get_train_and_test_dataloader(dataset_path=dataset_path)

        # Main Training Unit
        for hidden_unit in hidden_units:
            # Generate the model with specific number of hidden_unit
            model = get_model(hidden_unit, device)

            # Set the optimizer and criterion
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()
            criterion = criterion.to(device)

            # Train and evaluate the model
            train_and_evaluate_model(model, device, train_dataloader, test_dataloader, optimizer, criterion,
                                        dictionary_path=dictionary_path, checkpoint_path=checkpoint_path)
