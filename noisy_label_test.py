import sklearn.decomposition
import torch
import torchvision.datasets as datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

import dd_exp
import datasets
import models

DATASET = 'MNIST'

if DATASET == 'MNIST':
    #hidden_units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80,
    #                90, 100, 120, 150, 200, 400, 600, 800, 1000]
    hidden_units = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100,
                    120, 150, 200, 400, 600, 800, 1000]
elif DATASET == 'CIFAR-10':
    hidden_units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]

N_EPOCHS = 4000
N_SAMPLES = 4000
BATCH_SIZE = 64

TEST_GROUP = 1
TEST_NUMBERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

label_noise_ratio = 0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def decomposition_by_SVD(mat, k):
    # Singular Value Decomposition (SVD)
    U, S, V = np.linalg.svd(mat)

    # Choose top k important singular values
    Uk = U[:, :k]
    Sk = np.diag(S[0:k])
    Vk = V[:k, :]

    # recover the image
    imgMat_new = Uk @ Sk @ Vk
    return Uk

def get_test_losses(dictionary_path):
    with open(dictionary_path, "r", newline="") as infile:
        reader = csv.DictReader(infile)
        test_losses = []

        for row in reader:
            if int(row['Epoch']) == N_EPOCHS:
                test_losses.append(float(row['Test Loss']))

        return test_losses

def get_test_accuracy(dictionary_path):
    with open(dictionary_path, "r", newline="") as infile:
        reader = csv.DictReader(infile)
        test_accuracy = []

        for row in reader:
            if int(row['Epoch']) == N_EPOCHS:
                test_accuracy.append(float(row['Test Accuracy']))

        return test_accuracy

def get_train_accuracy(dictionary_path):
    with open(dictionary_path, "r", newline="") as infile:
        reader = csv.DictReader(infile)
        train_accuracy = []

        for row in reader:
            if int(row['Epoch']) == N_EPOCHS:
                train_accuracy.append(float(row['Train Accuracy']))

        return train_accuracy

def get_parameters(dictionary_path):
    with open(dictionary_path, "r", newline="") as infile:
        reader = csv.DictReader(infile)
        parameters = []

        for row in reader:
            if int(row['Epoch']) == N_EPOCHS:
                parameters.append(int(row['Parameters']) // 1000)

        return parameters

def load_dataset(dataset_path):
    org_train_dataset = torch.load(os.path.join(dataset_path, 'subset-clean.pth'))
    noisy_train_dataset = torch.load(os.path.join(dataset_path, 'subset-noise-20%.pth'))

    assert (len(org_train_dataset) == len(noisy_train_dataset))

    return org_train_dataset, noisy_train_dataset

def load_model(checkpoint_path, hidden_unit):
    if DATASET == 'MNIST':
        model = models.Simple_FC(hidden_unit)
    elif DATASET == 'CIFAR-10':
        model = models.FiveLayerCNN(hidden_unit)

    checkpoint = torch.load(os.path.join(checkpoint_path, 'Simple_FC_%d.pth' % hidden_unit))
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

    clean_label_dataloader = dd_exp.DataLoaderX(clean_label_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                      num_workers=0, pin_memory=True)
    noisy_label_dataloader_c = dd_exp.DataLoaderX(noisy_label_dataset_c, batch_size=BATCH_SIZE, shuffle=False,
                                                      num_workers=0, pin_memory=True)
    noisy_label_dataloader_n = dd_exp.DataLoaderX(noisy_label_dataset_n, batch_size=BATCH_SIZE, shuffle=False,
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


def knn_prediction_test(k):
    print('\nKNN Prediction Test: k = %d\n' % k)
    knn_accuracy_list, train_accuracy, test_accuracy, test_losses = [], [], [], []
    clean_train_accuracy, noisy_train_accuracy = [], []

    for i, test_number in enumerate(TEST_NUMBERS):
        directory = f"assets/{DATASET}/N=%d-3D/TEST-%d/epoch=%d-noise-%d-model-%d" % (
        N_SAMPLES, TEST_GROUP, N_EPOCHS, label_noise_ratio * 100, test_number)
        dataset_path = os.path.join(directory, 'dataset')
        clean_label_dataloader, noisy_label_dataloader_c, noisy_label_dataloader_n, n_noisy_data = \
            get_clean_noisy_dataloader(dataset_path)

        knn_accuracy_list.append([])
        clean_train_accuracy.append([])
        noisy_train_accuracy.append([])

        for n in hidden_units:
            # Initialize model with pretrained weights
            checkpoint_path = os.path.join(directory, "ckpt")
            model = load_model(checkpoint_path, n)
            model.eval()

            # Test the clean training data
            clean_train_loss, clean_train_acc = test(model, clean_label_dataloader)
            clean_train_accuracy[-1].append(clean_train_acc)
            noisy_train_loss, noisy_train_acc = test(model, noisy_label_dataloader_n)
            noisy_train_accuracy[-1].append(noisy_train_acc)

            # Obtain the hidden features of the clean data set
            data, hidden_features, predicts, labels = get_hidden_features(model, clean_label_dataloader)
            data_2, hidden_features_2, predicts_2, labels_2 = get_hidden_features(model, noisy_label_dataloader_c)

            knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
            knn.fit(hidden_features, labels)

            correct = sum(knn.predict(hidden_features_2) == labels_2)
            knn_accuracy_list[-1].append(correct / n_noisy_data)
            print('Test No = %d ; Hidden Units = %d ; Correct = %d' % (test_number, n, correct))

        # Get Parameters and dataset Losses
        dictionary_path = os.path.join(directory, "dictionary.csv")
        train_accuracy.append(get_train_accuracy(dictionary_path))
        test_accuracy.append(get_test_accuracy(dictionary_path))
        test_losses.append(get_test_losses(dictionary_path))

    knn_accuracy_list = np.mean(np.array(knn_accuracy_list), axis=0)
    train_accuracy = np.mean(np.array(train_accuracy), axis=0)
    test_accuracy = np.mean(np.array(test_accuracy), axis=0)
    test_losses = np.mean(np.array(test_losses), axis=0)

    clean_train_accuracy = np.mean(np.array(clean_train_accuracy), axis=0)
    noisy_train_accuracy = np.mean(np.array(noisy_train_accuracy), axis=0)

    # Plot the Diagram
    plt.figure(figsize=(10, 7))
    ax = plt.axes()

    if DATASET == 'MNIST':
        scale_function = (lambda x: x ** (1 / 3), lambda x: x ** 3)
        ax.set_xscale('function', functions=scale_function)

    ln1 = ax.plot(hidden_units, knn_accuracy_list, marker='o', label='KNN Prediction Accuracy (Noisy Training Data)',
                  color='red')
    ln2 = ax.plot(hidden_units, train_accuracy, marker='o', label='Train Accuracy (Full Train Set)', color='orange')
    ln3 = ax.plot(hidden_units, test_accuracy, marker='o', label='Test Accuracy (Full Test Set)', color='green')
    ln5 = ax.plot(hidden_units, clean_train_accuracy, marker='o', label='Train Accuracy (Clean Training Data)',
                  color='yellow')
    ln6 = ax.plot(hidden_units, noisy_train_accuracy, marker='o', label='Train Accuracy (Noisy Training Data)',
                  color='yellow')
    ax.set_xlabel('Number of Hidden Neurons (N)')
    ax.set_ylabel('Accuracy Rate')

    ax2 = ax.twinx()
    ln4 = ax2.plot(hidden_units, test_losses, marker='o', label='Test Losses (Full Test Set)', color='blue')
    ax2.set_ylabel('Cross Entropy Loss')
    ax2.set_ylim([0, 2.75])

    if DATASET == 'MNIST':
        plt.xticks([1, 5, 15, 40, 100, 250, 500, 1000])

    lns = ln1 + ln2 + ln3 + ln4 + ln5 + ln6
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    ax.grid()

    plt.title('%d-KNN Feature Extraction Test' % k)
    plt.savefig(f'assets/{DATASET}/N=%d-3D/TEST-%d/%d-NN Feature Extraction Test.png'
                % (N_SAMPLES, TEST_GROUP, k))


def decision_boundary_test():
    print('\nDecision Boundary Test\n')
    train_accuracy, test_accuracy, test_losses = [], [], []
    decision_boundary_distance = []

    for i, test_number in enumerate(TEST_NUMBERS):
        directory = f"assets/{DATASET}/N=%d-3D/TEST-%d/epoch=%d-noise-%d-model-%d" % (
            N_SAMPLES, TEST_GROUP, N_EPOCHS, label_noise_ratio * 100, test_number)
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

                    for k in range(50):
                        output = model(torch.from_numpy(hidden_features_3[noise_idx] + distance), path='half2')
                        predicted = output.argmax()

                        if predicted.eq(labels_3[noise_idx]):
                            distance = distance + vector * lr
                            #print(True, np.linalg.norm(distance), np.linalg.norm(vector))
                        else:
                            lr = lr / 2
                            distance = distance - vector * lr
                            #print(False, np.linalg.norm(distance), np.linalg.norm(vector))

                    if np.linalg.norm(vector) == 0:
                        break
                    n = np.linalg.norm(distance) / np.linalg.norm(vector)
                    db_distance_list.append(n)

            decision_boundary_distance[-1].append(np.mean(np.array(db_distance_list)))
            print(i, decision_boundary_distance[-1])

        # Get Parameters and dataset Losses
        dictionary_path = os.path.join(directory, "dictionary.csv")
        train_accuracy.append(get_train_accuracy(dictionary_path))
        test_accuracy.append(get_test_accuracy(dictionary_path))
        test_losses.append(get_test_losses(dictionary_path))

    train_accuracy = np.mean(np.array(train_accuracy), axis=0)
    test_accuracy = np.mean(np.array(test_accuracy), axis=0)
    test_losses = np.mean(np.array(test_losses), axis=0)
    decision_boundary_distance = np.mean(np.array(decision_boundary_distance), axis=0)

    # Plot the Diagram
    plt.figure(figsize=(10, 7))
    ax = plt.axes()

    if DATASET == 'MNIST':
        scale_function = (lambda x: x ** (1 / 3), lambda x: x ** 3)
        ax.set_xscale('function', functions=scale_function)

    ln1 = ax.plot(hidden_units, decision_boundary_distance, marker='o',
                      label='Decision Boundary Distance', color='red')
    ln2 = ax.plot(hidden_units, train_accuracy, marker='o', label='Train Accuracy (Full Train Set)', color='orange')
    ln3 = ax.plot(hidden_units, test_accuracy, marker='o', label='Test Accuracy (Full Test Set)', color='green')
    ax.set_xlabel('Number of Hidden Neurons (N)')
    ax.set_ylabel('Accuracy Rate')

    ax2 = ax.twinx()
    ln4 = ax2.plot(hidden_units, test_losses, marker='o', label='Test Losses (Full Test Set)', color='blue')
    ax2.set_ylabel('Cross Entropy Loss')
    ax2.set_ylim([0, 2.75])

    if DATASET == 'MNIST':
        plt.xticks([1, 5, 15, 40, 100, 250, 500, 1000])

    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    ax.grid()

    plt.title('Decision Boundary Distance Test')
    plt.savefig(f'assets/{DATASET}/N=%d-3D/TEST-%d/Decision Boundary Distance Test.png' % (N_SAMPLES, TEST_GROUP))

if __name__ == '__main__':
    knn_prediction_test(1)
    knn_prediction_test(5)
    knn_prediction_test(10)

    decision_boundary_test()
