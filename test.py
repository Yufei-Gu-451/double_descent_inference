import torch
import torchvision.datasets as datasets
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import argparse
import csv
import os

import main2 as main
import datasets
import models


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


def get_clean_noisy_dataloader_cifar(dataset_path, sample_size, noise_ratio, batch_size):
    # Load Clean and Noisy Dataset
    org_train_dataset = torch.load(os.path.join(dataset_path, 'clean-dataset.pth'))
    noisy_train_dataset = torch.load(os.path.join(dataset_path, f'noise-dataset-{int(noise_ratio * 100)}%.pth'))

    org_train_dataset = datasets.ListDataset(list(org_train_dataset))
    noisy_train_dataset = datasets.ListDataset(list(noisy_train_dataset))

    # Spilt the Training set to the ones with clean labels and the ones with random (noisy) labels
    clean_label_list, noisy_label_list_c, noisy_label_list_n = [], [], []

    for i in range(len(org_train_dataset)):
        data = org_train_dataset[i][0].numpy()

        if org_train_dataset[i][1] != noisy_train_dataset[i][1]:
            noisy_label_list_c.append((data, org_train_dataset[i][1]))
            noisy_label_list_n.append((data, noisy_train_dataset[i][1]))
        else:
            clean_label_list.append((data, org_train_dataset[i][1]))

    clean_dataset = datasets.ListDataset(clean_label_list)
    noisy_dataset_c = datasets.ListDataset(noisy_label_list_c)
    noisy_dataset_n = datasets.ListDataset(noisy_label_list_n)

    # Create Clean and Noisy Training Dataloader
    clean_label_dataloader = main.DataLoaderX(clean_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=0, pin_memory=True)
    noisy_label_dataloader_c = main.DataLoaderX(noisy_dataset_c, batch_size=batch_size, shuffle=False,
                                                num_workers=0, pin_memory=True)
    noisy_label_dataloader_n = main.DataLoaderX(noisy_dataset_n, batch_size=batch_size, shuffle=False,
                                                num_workers=0, pin_memory=True)

    return clean_label_dataloader, noisy_label_dataloader_c, noisy_label_dataloader_n, len(noisy_dataset_c)


def get_clean_noisy_dataloader_mnist(dataset_path, sample_size, noise_ratio, batch_size):
    # Load Clean and Noisy Dataset
    org_train_dataset = torch.load(os.path.join(dataset_path, 'clean-dataset.pth'))
    noisy_train_dataset = torch.load(os.path.join(dataset_path, f'noise-dataset-{int(noise_ratio * 100)}%.pth'))

    # Filter Cleand and Noisy Data Index
    clean_index, noisy_index = [], []
    for i in range(4000):
        if org_train_dataset.dataset.targets[i] == noisy_train_dataset.dataset.targets[i]:
            clean_index.append(i)
        else:
            noisy_index.append(i)

    # Filter Cleand and Noisy Data
    clean_data = org_train_dataset.dataset.data[clean_index]
    clean_label = org_train_dataset.dataset.targets[clean_index]

    noisy_data = noisy_train_dataset.dataset.data[noisy_index]
    noisy_label_n = noisy_train_dataset.dataset.targets[noisy_index]
    noisy_label_c = org_train_dataset.dataset.targets[noisy_index]

    # Create Clean and Noisy Training Dataset
    clean_dataset = datasets.ImageDataset(clean_data, clean_label)
    noisy_dataset_n = datasets.ImageDataset(noisy_data, noisy_label_n)
    noisy_dataset_c = datasets.ImageDataset(noisy_data, noisy_label_c)

    # Create Clean and Noisy Training Dataloader
    clean_label_dataloader = main.DataLoaderX(clean_dataset, batch_size=batch_size, shuffle=False,
                                                      num_workers=0, pin_memory=True)
    noisy_label_dataloader_c = main.DataLoaderX(noisy_dataset_c, batch_size=batch_size, shuffle=False,
                                                      num_workers=0, pin_memory=True)
    noisy_label_dataloader_n = main.DataLoaderX(noisy_dataset_n, batch_size=batch_size, shuffle=False,
                                                  num_workers=0, pin_memory=True)

    return clean_label_dataloader, noisy_label_dataloader_c, noisy_label_dataloader_n, len(noisy_dataset_c)


def get_hidden_features(dataset, model, dataloader):
    # Obtain the hidden features
    data, hidden_features, predicts, true_labels = [], [], [], []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloader):
            hidden_feature = model(inputs.to(torch.float32), path='half1')
            outputs = model(hidden_feature, path='half2')

            for input in inputs:
                input = input.cpu().detach().numpy()
                data.append(input)

            for hf in hidden_feature:
                hf = hf.cpu().detach().numpy()
                hidden_features.append(hf)

            for output in outputs:
                predict = output.cpu().detach().numpy().argmax()
                predicts.append(predict)

            for label in labels:
                true_labels.append(label)

    # Define image_size and feature size by Dataset
    if dataset == 'MNIST':
        image_size = 28 * 28
        feature_size = model.n_hidden_units
    elif dataset == 'CIFAR-10':
        image_size = 32 * 32 * 3
        feature_size = model.n_hidden_units * 8
    else:
        raise NotImplementedError

    # Reshape all numpy arrays
    data = np.array(data).reshape(len(true_labels), image_size)
    hidden_features = np.array(hidden_features).reshape(len(true_labels), feature_size)
    predicts = np.array(predicts).reshape(len(true_labels), )
    true_labels = np.array(true_labels).reshape(len(true_labels), )

    return data, hidden_features, predicts, true_labels


def test(model, test_dataloader):
    model.eval()
    cumulative_loss, correct, total = 0.0, 0, 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_dataloader):
            labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            cumulative_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.argmax(1)).sum().item()

    test_loss = cumulative_loss / len(test_dataloader)
    test_acc = correct/total

    return test_loss, test_acc


def knn_prediction_test(directory, hidden_units, args):
    print('\nKNN Prediction Test\n')
    dataset_path = os.path.join(directory, 'dataset')

    # Load cleand and noisy dataloader
    if args.dataset == 'MNIST':
        clean_label_dataloader, noisy_label_dataloader_c, noisy_label_dataloader_n, n_noisy_data = \
            get_clean_noisy_dataloader_mnist(dataset_path, sample_size=args.sample_size, noise_ratio=args.noise_ratio, batch_size=args.batch_size)
    elif args.dataset == 'CIFAR-10':
        clean_label_dataloader, noisy_label_dataloader_c, noisy_label_dataloader_n, n_noisy_data = \
            get_clean_noisy_dataloader_cifar(dataset_path, sample_size=args.sample_size, noise_ratio=args.noise_ratio,
                                             batch_size=args.batch_size)
    else:
        raise NotImplementedError

    knn_5_accuracy_list = []

    for n in hidden_units:
        # Initialize model with pretrained weights
        checkpoint_path = os.path.join(directory, "ckpt")
        model = load_model(checkpoint_path, dataset=args.dataset, hidden_unit=n)
        model.eval()

        # Obtain the hidden features of the clean data set
        data, hidden_features, predicts, labels = get_hidden_features(args.dataset, model, clean_label_dataloader)
        data_2, hidden_features_2, predicts_2, labels_2 = get_hidden_features(args.dataset, model, noisy_label_dataloader_c)

        knn_5 = KNeighborsClassifier(n_neighbors=5, metric='cosine')
        knn_5.fit(hidden_features, labels)

        correct = sum(knn_5.predict(hidden_features_2) == labels_2)
        knn_5_accuracy_list.append(correct / n_noisy_data)
        print('Test No = %d ; Hidden Units = %d ; Correct = %d ; k = 5' % (test_number, n, correct))

    return knn_5_accuracy_list


def decision_boundary_test(args, directory):
    print('\nDecision Boundary Test\n')
    decision_boundary_distance = []

    for test_number in [args.test_number_start, args.test_number_end + 1]:
        dataset_path = os.path.join(directory, 'dataset')

        # Load cleand and noisy dataloader
        if args.dataset == 'MNIST':
            clean_label_dataloader, noisy_label_dataloader_c, noisy_label_dataloader_n, n_noisy_data = \
                get_clean_noisy_dataloader_mnist(dataset_path, sample_size=args.sample_size,
                                                 noise_ratio=args.noise_ratio, batch_size=args.batch_size)
        elif args.dataset == 'CIFAR-10':
            clean_label_dataloader, noisy_label_dataloader_c, noisy_label_dataloader_n, n_noisy_data = \
                get_clean_noisy_dataloader_cifar(dataset_path, sample_size=args.sample_size,
                                                 noise_ratio=args.noise_ratio,
                                                 batch_size=args.batch_size)
        else:
            raise NotImplementedError

        decision_boundary_distance.append([])

        for n in args.hidden_units:
            # Initialize model with pretrained weights
            checkpoint_path = os.path.join(directory, "ckpt")
            model = load_model(checkpoint_path, dataset=args.dataset, hidden_unit=n)
            model.eval()

            # Obtain the hidden features of the clean data set
            data, hidden_features, predicts, labels = get_hidden_features(args.dataset, model, clean_label_dataloader)
            data_3, hidden_features_3, predicts_3, labels_3 = get_hidden_features(args.dataset, model, noisy_label_dataloader_n)

            knn = KNeighborsClassifier(n_neighbors=10, metric='cosine')
            knn.fit(hidden_features, labels)
            neigh_dist, neigh_ind = knn.kneighbors(hidden_features_3)

            db_distance_list = []
            for noise_idx in range(n_noisy_data):
                for neigh_idx in neigh_ind[noise_idx]:
                    vector = hidden_features[neigh_idx] - hidden_features_3[noise_idx]
                    distance = vector
                    lr = 1

                    for k in range(10):
                        output = model(torch.from_numpy(hidden_features_3[noise_idx] + distance), path='half2')
                        predicted = output.argmax()

                        if predicted.eq(labels_3[noise_idx]):
                            distance = distance + vector * lr
                        else:
                            lr = lr / 2
                            distance = distance - vector * lr

                    if np.linalg.norm(vector) == 0:
                        break
                    n = np.linalg.norm(distance) / np.linalg.norm(vector)
                    db_distance_list.append(n)

            decision_boundary_distance[-1].append(np.mean(np.array(db_distance_list)))
            print(n, decision_boundary_distance[-1])

    decision_boundary_distance = np.mean(np.array(decision_boundary_distance), axis=0)

    return decision_boundary_distance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Double Descent Experiment')
    parser.add_argument('-d', '--dataset', choices=['MNIST', 'CIFAR-10'], type=str, help='dataset')
    parser.add_argument('-N', '--sample_size', type=int, help='number of samples used as training data')
    parser.add_argument('-p', '--noise_ratio', type=float, help='label noise ratio')
    parser.add_argument('-m', '--model', choices=['SimpleFC', 'CNN', 'ResNet18'], type=str,
                        help='neural network architecture')

    parser.add_argument('-g', '--group', type=int, help='TEST GROUP')
    parser.add_argument('-s', '--start', type=int, help='starting number of test number')
    parser.add_argument('-e', '--end', type=int, help='ending number of test number')

    #parser.add_argument('--hidden_units', action='append', type=int, help='hidden units / layer width')

    # parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')

    parser.add_argument('--epochs', type=int, help='epochs of training time')
    parser.add_argument('--test-gap', default=10 * 1000, type=int, help='gradient step gap to test the model')
    parser.add_argument('--opt', default='sgd', type=str, help='use which optimizer. SGD or Adam')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    # parser.add_argument('-momentum', default=0.0, type=float, help='momentum for SGD')

    args = parser.parse_args()
    print(args)

    if args.model == 'SimpleFC':
        hidden_units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 25, 30, 35, 40, 45, 50, 55, 60, 70,
                        80, 90, 100, 120, 150, 200, 400, 600, 800, 1000]
    elif args.model == 'CNN' or args.model == 'ResNet18':
        hidden_units = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
    else:
        raise NotImplementedError

    # Initialization of used device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    print('Using device : ', torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))

    train_accuracy, test_accuracy, train_losses, test_losses = [], [], [], []
    knn_test, knn_5_accuracy_list = True, []

    for test_number in range(args.start, args.end + 1):
        # Define the roots and paths
        directory = f"assets/{args.dataset}-{args.model}/N=%d-3d/TEST-%d/Epoch=%d-noise-%d-model-%d-sgd" \
                % (args.sample_size, args.group, args.epochs, args.noise_ratio * 100, test_number)

        train_accuracy.append([])
        test_accuracy.append([])
        train_losses.append([])
        test_losses.append([])

        for hidden_unit in hidden_units:
        # Get Parameters and dataset Losses
            dictionary_path = os.path.join(directory, "dictionary/dictionary_%d.csv" % hidden_unit)

            with open(dictionary_path, "r", newline="") as infile:
                reader = csv.DictReader(infile)
                for row in reader:
                    if int(row['Epochs']) == args.epochs:
                        train_accuracy[-1].append(float(row['Train Accuracy']))
                        test_accuracy[-1].append(float(row['Test Accuracy']))
                        train_losses[-1].append(float(row['Train Loss']))
                        test_losses[-1].append(float(row['Test Loss']))

        # Run KNN Test
        if knn_test and args.noise_ratio > 0:
           knn_5_accuracy_list.append(knn_prediction_test(directory, hidden_units, args))

    train_accuracy = np.mean(np.array(train_accuracy), axis=0)
    test_accuracy = np.mean(np.array(test_accuracy), axis=0)
    train_losses = np.mean(np.array(train_losses), axis=0)
    test_losses = np.mean(np.array(test_losses), axis=0)

    if knn_test and args.noise_ratio > 0:
        knn_5_accuracy_list = np.mean(np.array(knn_5_accuracy_list), axis=0)

    # Plot the Diagram
    scale_function = (lambda x: x ** (1 / 4), lambda x: x ** 4)

    fig, (ax1, ax3) = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))
    ax1.set_title(f'Experiment Results on {args.dataset} (N=%d, p=%d%%)' % (args.sample_size, args.noise_ratio * 100))

    if args.dataset == 'MNIST':
        ax1.set_xscale('function', functions=scale_function)
        ax1.set_xticks([1, 5, 15, 40, 100, 250, 500, 1000])
        ax3.set_xscale('function', functions=scale_function)
        ax3.set_xticks([1, 5, 15, 40, 100, 250, 500, 1000])

        ax3.set_ylim([0.0, 1.75])
    elif args.dataset == 'CIFAR-10':
        ax1.set_xticks([1, 10, 20, 30, 40, 50, 60, 64])
        ax3.set_xticks([1, 10, 20, 30, 40, 50, 60, 64])

        ax3.set_ylim([0.0, 3.0])
    else:
        raise NotImplementedError

    if args.model == 'SimpleFC':
        ax1.set_xlabel('Number of Hidden Neurons (N)')
        ax3.set_xlabel('Number of Hidden Neurons (N)')
    elif args.model == 'CNN' or args.model == 'ResNet18':
        ax1.set_xlabel('Convolutional Layer Width (K)')
        ax3.set_xlabel('Convolutional Layer Width (K)')
    else:
        raise NotImplementedError

    # Subplot 1
    ln1 = ax1.plot(hidden_units, train_accuracy, label='Train Accuracy', color='red')
    ln2 = ax1.plot(hidden_units, test_accuracy, label='Test Accuracy', color='blue')
    ax1.set_ylabel('Accuracy (100%)')
    ax1.set_ylim([0, 1.05])

    if knn_test and args.noise_ratio > 0:
        ax2 = ax1.twinx()
        ln3 = ax2.plot(hidden_units, knn_5_accuracy_list, label='Prediction Accuracy', color='cyan')
        ax2.set_ylabel('KNN Label Accuracy (100%)')
        ax2.set_ylim([0, 1.05])

        lns = ln1 + ln2 + ln3
    else:
        lns = ln1 + ln2

    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='lower right')
    ax1.grid()

    # Subplot 2
    ln6 = ax3.plot(hidden_units, train_losses, label='Train Losses', color='red')
    ln7 = ax3.plot(hidden_units, test_losses, label='Test Losses', color='blue')
    ax3.set_ylabel('Cross Entropy Loss')

    lns = ln6 + ln7
    labs = [l.get_label() for l in lns]
    ax3.legend(lns, labs, loc='upper right')
    ax3.grid()

    # Plot Title and Save
    plt.savefig(f'assets/{args.dataset}-{args.model}/N=%d-3d/TEST-%d/{args.dataset}-{args.model}-p=%d.png' %
                (args.sample_size, args.group, args.noise_ratio))