import torch
import torchvision.datasets as datasets
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

import main_arg
import datasets
import models

MODEL = 'CNN'
DATASET = 'CIFAR-10'

TEST_GROUP = 2
TEST_NUMBERS = [0]
label_noise_ratio = 0.2

if DATASET == 'MNIST':
    hidden_units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90,
                    100, 120, 150, 200, 400, 600, 800, 1000]
    N_SAMPLES = 4000
elif DATASET == 'CIFAR-10':
    hidden_units = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
    N_SAMPLES = 50000
else:
    raise NotImplementedError

GRADIENT_STEP = 100 * 1000
BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_train_losses(dictionary_path):
    with open(dictionary_path, "r", newline="") as infile:
        reader = csv.DictReader(infile)
        train_losses = []

        for row in reader:
            if int(row['Gradient Steps']) >= GRADIENT_STEP:
                train_losses.append(float(row['Train Loss']))

        return train_losses

def get_test_losses(dictionary_path):
    with open(dictionary_path, "r", newline="") as infile:
        reader = csv.DictReader(infile)
        test_losses = []

        for row in reader:
            if int(row['Gradient Steps']) >= GRADIENT_STEP:
                test_losses.append(float(row['Test Loss']))

        return test_losses

def get_test_accuracy(dictionary_path):
    with open(dictionary_path, "r", newline="") as infile:
        reader = csv.DictReader(infile)
        test_accuracy = []

        for row in reader:
            if int(row['Gradient Steps']) >= GRADIENT_STEP:
                test_accuracy.append(float(row['Test Accuracy']))

        return test_accuracy

def get_train_accuracy(dictionary_path):
    with open(dictionary_path, "r", newline="") as infile:
        reader = csv.DictReader(infile)
        train_accuracy = []

        for row in reader:
            if int(row['Gradient Steps']) >= GRADIENT_STEP:
                train_accuracy.append(float(row['Train Accuracy']))

        return train_accuracy

def get_parameters(dictionary_path):
    with open(dictionary_path, "r", newline="") as infile:
        reader = csv.DictReader(infile)
        parameters = []

        for row in reader:
            if int(row['Gradient Steps']) >= GRADIENT_STEP:
                parameters.append(int(row['Parameters']) // 1000)

        return parameters

def load_dataset(dataset_path):
    org_train_dataset = torch.load(os.path.join(dataset_path, 'subset-clean.pth'))

    if label_noise_ratio > 0:
        noisy_train_dataset = torch.load(os.path.join(dataset_path, f'subset-noise-{int(label_noise_ratio * 100)}%.pth'))
    elif label_noise_ratio == 0:
        noisy_train_dataset = torch.load(os.path.join(dataset_path, 'subset-clean.pth'))
    else:
        raise NotImplementedError

    assert (len(org_train_dataset) == len(noisy_train_dataset))

    return org_train_dataset, noisy_train_dataset


def load_model(checkpoint_path, hidden_unit):
    if DATASET == 'MNIST':
        model = models.Simple_FC(hidden_unit)
    elif DATASET == 'CIFAR-10':
        model = models.FiveLayerCNN(hidden_unit)
    elif DATASET == 'ResNet18':
        model = models.ResNet18(hidden_unit)
    else:
        raise NotImplementedError

    checkpoint = torch.load(os.path.join(checkpoint_path, 'Model_State_Dict_%d.pth' % hidden_unit))
    model.load_state_dict(checkpoint['net'])

    return model


def get_clean_noisy_dataloader(dataset_path):
    # Load the two dataset
    org_train_dataset, noisy_train_dataset = load_dataset(dataset_path)

    # Spilt the Training set to the ones with clean labels and the ones with random (noisy) labels
    clean_label_list, noisy_label_list_c, noisy_label_list_n = [], [], []

    for i in range(len(org_train_dataset)):
        data = org_train_dataset[i][0].numpy()

        if org_train_dataset[i][1] != noisy_train_dataset[i][1]:
            noisy_label_list_c.append((data, org_train_dataset[i][1]))
            noisy_label_list_n.append((data, noisy_train_dataset[i][1]))
        else:
            clean_label_list.append((data, org_train_dataset[i][1]))

    clean_label_dataset = datasets.ListDataset(clean_label_list)
    noisy_label_dataset_c = datasets.ListDataset(noisy_label_list_c)
    noisy_label_dataset_n = datasets.ListDataset(noisy_label_list_n)

    clean_label_dataloader = main.DataLoaderX(clean_label_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                      num_workers=0, pin_memory=True)
    noisy_label_dataloader_c = main.DataLoaderX(noisy_label_dataset_c, batch_size=BATCH_SIZE, shuffle=False,
                                                      num_workers=0, pin_memory=True)
    noisy_label_dataloader_n = main.DataLoaderX(noisy_label_dataset_n, batch_size=BATCH_SIZE, shuffle=False,
                                                  num_workers=0, pin_memory=True)

    return clean_label_dataloader, noisy_label_dataloader_c, noisy_label_dataloader_n, len(noisy_label_list_c)

def get_hidden_features(model, dataloader):
    # Obtain the hidden features
    data, hidden_features, predicts, true_labels = [], [], [], []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloader):
            hidden_feature = model(inputs, path='half1')
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

    if DATASET == 'MNIST':
        image_size = 28 * 28
        feature_size = model.n_hidden_units
    elif DATASET == 'CIFAR-10':
        image_size = 32 * 32 * 3
        feature_size = model.n_hidden_units * 8
    else:
        raise NotImplementedError

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


def knn_prediction_test(directory):
    print('\nKNN Prediction Test\n')
    dataset_path = os.path.join(directory, 'dataset')
    clean_label_dataloader, noisy_label_dataloader_c, noisy_label_dataloader_n, n_noisy_data = \
        get_clean_noisy_dataloader(dataset_path)

    knn_5_accuracy_list = []

    for n in hidden_units:
        # Initialize model with pretrained weights
        checkpoint_path = os.path.join(directory, "ckpt")
        model = load_model(checkpoint_path, n)
        model.eval()

        # Obtain the hidden features of the clean data set
        data, hidden_features, predicts, labels = get_hidden_features(model, clean_label_dataloader)
        data_2, hidden_features_2, predicts_2, labels_2 = get_hidden_features(model, noisy_label_dataloader_c)

        knn_5 = KNeighborsClassifier(n_neighbors=5, metric='cosine')
        knn_5.fit(hidden_features, labels)

        correct = sum(knn_5.predict(hidden_features_2) == labels_2)
        knn_5_accuracy_list.append(correct / n_noisy_data)
        print('Test No = %d ; Hidden Units = %d ; Correct = %d ; k = 5' % (test_number, n, correct))

    return knn_5_accuracy_list


def decision_boundary_test(directory):
    print('\nDecision Boundary Test\n')
    decision_boundary_distance = []

    for test_number in TEST_NUMBERS:
        dataset_path = os.path.join(directory, 'dataset')
        clean_label_dataloader, noisy_label_dataloader_c, noisy_label_dataloader_n, n_noisy_data = \
            get_clean_noisy_dataloader(dataset_path)

        decision_boundary_distance.append([])

        for n in hidden_units:
            # Initialize model with pretrained weights
            checkpoint_path = os.path.join(directory, "ckpt")
            model = load_model(checkpoint_path, n)
            model.eval()

            # Obtain the hidden features of the clean data set
            data, hidden_features, predicts, labels = get_hidden_features(model, clean_label_dataloader)
            data_3, hidden_features_3, predicts_3, labels_3 = get_hidden_features(model, noisy_label_dataloader_n)

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
            print(i, decision_boundary_distance[-1])

    decision_boundary_distance = np.mean(np.array(decision_boundary_distance), axis=0)

    with open(f"assets/{DATASET}/N=%d-3D/TEST-%d/decision_boundary_distance.csv" % (N_SAMPLES, TEST_GROUP), 'w') as f:
        #f.write('decesion_boundary_distance\n')

        for item in decision_boundary_distance:
            f.write('%lf\n' % item)


if __name__ == '__main__':
    train_accuracy, test_accuracy, train_losses, test_losses = [], [], [], []
    knn_5_accuracy_list = []

    for i, test_number in enumerate(TEST_NUMBERS):
        directory = f"assets/{DATASET}-{MODEL}/N=%d-3D/TEST-%d/GS=%dK-noise-%d-model-%d" % (
            N_SAMPLES, TEST_GROUP, GRADIENT_STEP // 1000, label_noise_ratio * 100, test_number)

        # Get Parameters and dataset Losses
        dictionary_path = os.path.join(directory, "dictionary.csv")
        train_accuracy.append(get_train_accuracy(dictionary_path))
        test_accuracy.append(get_test_accuracy(dictionary_path))
        train_losses.append(get_train_losses(dictionary_path))
        test_losses.append(get_test_losses(dictionary_path))

        # Run KNN Test
        if label_noise_ratio > 0:
            knn_5_accuracy_list.append(knn_prediction_test(directory))

    train_accuracy = np.mean(np.array(train_accuracy), axis=0)
    test_accuracy = np.mean(np.array(test_accuracy), axis=0)
    train_losses = np.mean(np.array(train_losses), axis=0)
    test_losses = np.mean(np.array(test_losses), axis=0)

    if label_noise_ratio > 0:
        knn_5_accuracy_list = np.mean(np.array(knn_5_accuracy_list), axis=0)


    # Plot the Diagram
    scale_function = (lambda x: x ** (1 / 4), lambda x: x ** 4)

    fig, (ax1, ax3) = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))

    if DATASET == 'MNIST':
        ax1.set_xscale('function', functions=scale_function)
        ax1.set_xticks([1, 5, 15, 40, 100, 250, 500, 1000])
        ax3.set_xscale('function', functions=scale_function)
        ax3.set_xticks([1, 5, 15, 40, 100, 250, 500, 1000])
    elif DATASET == 'CIFAR-10':
        ax1.set_xticks([1, 10, 20, 30, 40, 50, 60, 64])
        ax3.set_xticks([1, 10, 20, 30, 40, 50, 60, 64])
    else:
        raise NotImplementedError

    if MODEL == 'SimpleFC':
        ax1.set_xlabel('Number of Hidden Neurons (N)')
        ax3.set_xlabel('Number of Hidden Neurons (N)')
    elif MODEL == 'CNN' or MODEL == 'ResNet18':
        ax1.set_xlabel('Convolutional Layer Width (K)')
        ax3.set_xlabel('Convolutional Layer Width (K)')
    else:
        raise NotImplementedError

    # Subplot 1
    ln1 = ax1.plot(hidden_units, train_accuracy, label='Train Accuracy', color='red')
    ln2 = ax1.plot(hidden_units, test_accuracy, label='Test Accuracy', color='blue')
    ax1.set_ylabel('Accuracy (100%)')
    ax1.set_ylim([0, 1.05])

    if label_noise_ratio > 0:
        ax2 = ax1.twinx()
        ln3 = ax2.plot(hidden_units, knn_5_accuracy_list, label='KNN Prediction Accuracy (k = 5)', color='cyan')
        ax2.set_ylabel('KNN Label Accuracy (100%)')
        ax2.set_ylim([0, 1.05])

        lns = ln1 + ln2 + ln3
    elif label_noise_ratio == 0:
        lns = ln1 + ln2
    else:
        raise NotImplementedError

    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    ax1.grid()

    # Subplot 2
    ln6 = ax3.plot(hidden_units, train_losses, label='Train Losses', color='red')
    ln7 = ax3.plot(hidden_units, test_losses, label='Test Losses', color='blue')
    ax3.set_ylabel('Cross Entropy Loss')

    lns = ln6 + ln7
    labs = [l.get_label() for l in lns]
    ax3.legend(lns, labs, loc=0)
    ax3.grid()

    # Plot Title and Save
    plt.title(f'Experiment Results on {DATASET} (N=%d, p=%d%%)' % (N_SAMPLES, label_noise_ratio * 100))
    plt.savefig(f'assets/{DATASET}-{MODEL}/N=%d-3D/TEST-%d/GS=%dK-noise-%d-ER.png' %
                (N_SAMPLES, TEST_GROUP, GRADIENT_STEP // 1000, label_noise_ratio * 100))
