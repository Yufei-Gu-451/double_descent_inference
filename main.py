import torch
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import csv
import os

from datetime import datetime
import numpy as np
import random
import argparse

import models
import datasets


# ------------------------------------------------------------------------------------------


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

# Return the train_dataloader and test_dataloader of MINST
def get_train_and_test_dataloader(args, dataset_path):
    train_dataset = datasets.load_test_dataset_from_file(label_noise_ratio=args.noise_ratio, dataset_path=dataset_path)

    train_dataloader = DataLoaderX(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    test_dataset = datasets.get_test_dataset(DATASET=args.dataset)

    test_dataloader = DataLoaderX(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f'Load {args.dataset} dataset success;')

    return train_dataloader, test_dataloader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ------------------------------------------------------------------------------------------


# Set the neural network model to be used
def get_model(dataset, model_name, hidden_unit, device):
    if dataset == 'MNIST':
        model = models.Simple_FC(hidden_unit)
    elif dataset == 'CIFAR-10':
        model = models.FiveLayerCNN(hidden_unit)
    elif dataset == 'ResNet18':
        model = models.ResNet18(hidden_unit)
    else:
        raise NotImplementedError

    model = model.to(device)

    print(f"\n{model_name} Model with %d hidden neurons successfully generated;" % hidden_unit)

    print('Number of parameters: %d' % sum(p.numel() for p in model.parameters()))

    return model


# ------------------------------------------------------------------------------------------


def model_save(model, gradient_step, test_accuracy, checkpoint_path):
    state = {
        'net': model.state_dict(),
        'acc': test_accuracy,
        'gradient_step': gradient_step,
    }

    torch.save(state, os.path.join(checkpoint_path, 'Model_State_Dict_%d.pth' % hidden_unit))
    print("Torch saved successfully!\n")

def status_save(n_hidden_units, gradient_step, parameters, train_loss, train_acc, test_loss, test_acc, lr,
                time, curr_time, dictionary_path):
    print("Hidden Neurons : %d ; Parameters : %d ; Train Loss : %.3f ; Train Acc : %.3f ; Test Loss : %.3f ; "
          "Test Acc : %.3f\n" % (n_hidden_units, parameters, train_loss, train_acc, test_loss, test_acc))

    print('Writing to a csv file...')
    dictionary = {'Hidden Neurons': hidden_unit, 'Gradient Steps': gradient_step, 'Parameters': parameters,
                  'Train Loss': train_loss, 'Train Accuracy': train_acc, 'Test Loss': test_loss, 'Test Accuracy': test_acc,
                  'Learning Rate': lr, 'Time Cost': time, 'Date-Time': curr_time}

    with open(dictionary_path, "a", newline="") as fp:
        # Create a writer object
        writer = csv.DictWriter(fp, fieldnames=dictionary.keys())

        # Write the data rows
        writer.writerow(dictionary)
        print('Done writing to a csv file\n')


# ------------------------------------------------------------------------------------------


# Train and Evalute the model
def train_and_evaluate_model(model, device, args, train_dataloader, test_dataloader, optimizer, criterion, dictionary_path, checkpoint_path):
    start_time = datetime.now()

    parameters = sum(p.numel() for p in model.parameters())
    n_hidden_units = model.n_hidden_units

    gradient_step, gs_count, test_acc = 0, 0, 0

    while gradient_step <= args.gradient_step:
        # Model Training
        model.train()
        cumulative_loss, correct, total, idx = 0.0, 0, 0, 0

        for idx, (inputs, labels) in enumerate(train_dataloader):
            gradient_step += 1
            if gradient_step % 512 == 0:
                optimizer.param_groups[0]['lr'] = args.lr / pow(gradient_step // 512, 0.5)

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

        print("Gradient Step : %d(K) ; Train Loss : %f ; Train Acc : %.3f ; Learning Rate : %f" %
              (gradient_step // 1000, train_loss, train_acc, lr))

        # Test Model after every 10K Gradient Steps
        if gradient_step // args.test_gap > gs_count:
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
            gs_count = gradient_step // args.test_gap

            status_save(n_hidden_units, gradient_step, parameters, train_loss, train_acc, test_loss, test_acc, lr,
                        time, curr_time, dictionary_path=dictionary_path)

        model_save(model, test_acc, gradient_step, checkpoint_path=checkpoint_path)

    return


# ------------------------------------------------------------------------------------------



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Double Descent Experiment')
    parser.add_argument('--dataset', choices=['MNIST', 'CIFAR-10'], type=str, help='dataset')
    parser.add_argument('--sample_size', type=int, help='number of samples used as training data')
    parser.add_argument('--noise_ratio', type=float, help='label noise ratio')
    parser.add_argument('--model', choices=['SimpleFC', 'CNN', 'ResNet18'], type=str, help='neural network architecture')

    parser.add_argument('--group', type=int, help='TEST GROUP')
    parser.add_argument('--start', type=int, help='starting number of test number')
    parser.add_argument('--end', type=int, help='ending number of test number')

    parser.add_argument('--hidden_units', action='append', type=int, help='hidden units / layer width')

    # parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')

    parser.add_argument('--gradient_step', default=500 * 1000, type=int, help='gradient steps used in experiment')
    parser.add_argument('--test-gap', default=10 * 1000, type=int, help='gradient step gap to test the model')
    parser.add_argument('--opt', default='sgd', type=str, help='use which optimizer. SGD or Adam')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    # parser.add_argument('-momentum', default=0.0, type=float, help='momentum for SGD')

    args = parser.parse_args()
    print(args)

    # Initialization of used device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    print('Using device : ', torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))

    # Main Program
    for test_number in range(args.start, args.end + 1):
        # Setup seed for reproduction
        setup_seed(20 + test_number)

        # Define the roots and paths
        directory = f"assets/{args.dataset}-{args.model}/N=%d-3d/TEST-%d/GS=%dK-noise-%d-model-%d-sgd" \
                % (args.sample_size, args.group, args.gradient_step // 1000, args.noise_ratio * 100, test_number)

        dataset_path = os.path.join(directory, 'dataset')
        checkpoint_path = os.path.join(directory, "ckpt")

        if not os.path.isdir(directory):
            os.mkdir(directory)
        if not os.path.isdir(dataset_path):
            os.mkdir(dataset_path)
        if not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)

        # Get the training and testing data of specific sample size
        train_dataloader, test_dataloader = get_train_and_test_dataloader(args=args, dataset_path=dataset_path)

        # Main Training Unit
        print('Hidden Units : ', args.hidden_units)
        for hidden_unit in args.hidden_units:
            # Generate the model with specific number of hidden_unit
            model = get_model(args.dataset, args.model, hidden_unit, device)

            # Set the optimizer and criterion
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
            criterion = torch.nn.CrossEntropyLoss()
            criterion = criterion.to(device)

            # Setup the dictionary file for n_hidden_unit
            dictionary_path = os.path.join(directory, "dictionary_%d.csv" % hidden_unit)

            # Initialize Status Dictionary
            dictionary = {'Hidden Neurons': 0, 'Gradient Steps': 0, 'Parameters': 0, 'Train Loss': 0, 'Train Accuracy': 0,
                          'Test Loss': 0, 'Test Accuracy': 0, 'Learning Rate': 0, 'Time Cost': 0, 'Date-Time': 0}

            with open(dictionary_path, "a", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=dictionary.keys())
                writer.writeheader()

            # Train and evaluate the model
            train_and_evaluate_model(model, device, args, train_dataloader, test_dataloader, optimizer, criterion,
                                        dictionary_path=dictionary_path, checkpoint_path=checkpoint_path)
