import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import models
import dd_exp
import datasets as datasets_

N_EPOCHS = 100
N_SAMPLES = 50000
BATCH_SIZE = 64


def train_layer(model, device, dataloader, optimizer, criterion):
    model.train()
    cumulative_loss, correct, total = 0.0, 0, 0

    feature_dataset = []

    for idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        hidden_features = model(inputs, path='half1')
        outputs = model(hidden_features, path='half2')

        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
        loss = criterion(outputs, one_hot_labels)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        cumulative_loss = cumulative_loss + loss.item()
        _, predicted = outputs.max(1)
        total = total + one_hot_labels.size(0)
        correct = correct + predicted.eq(one_hot_labels.argmax(1)).sum().item()

        for i in range(len(labels)):
            feature_dataset.append([hidden_features[i], labels[i]])

    train_loss = cumulative_loss / len(dataloader)
    train_acc = correct / total

    return model, feature_dataset, train_loss, train_acc


def train_classifier(classifier, device, dataloader, optimizer, criterion):
    classifier.train()
    cumulative_loss, correct, total = 0.0, 0, 0

    for idx, (inputs, labels) in enumerate(feature_dataloader_1):
        labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = classifier.forward(inputs)
        loss = criterion(outputs, labels)

        optimizer3.zero_grad()
        loss.backward(retain_graph=True)
        optimizer3.step()

        cumulative_loss = cumulative_loss + loss.item()
        _, predicted = outputs.max(1)
        total = total + labels.size(0)
        correct = correct + predicted.eq(labels.argmax(1)).sum().item()

    train_loss = cumulative_loss / len(feature_dataloader_1)
    train_acc = correct / total

    return classifier, train_loss, train_acc



# Model testing
def test(layer1, classifier, device, test_dataloader):
    layer1.eval()
    #layer2.eval()
    classifier.eval()
    cumulative_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_dataloader):
            labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
            inputs = inputs.to(device)
            labels = labels.to(device)

            hidden_features_1 = layer1(inputs, path='half1')
            #hidden_features_2 = layer2(hidden_features_1, path='half1')
            outputs = classifier(hidden_features_1)
            loss = criterion(outputs, labels)

            cumulative_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.argmax(1)).sum().item()

    test_loss = cumulative_loss / len(test_dataloader)
    test_acc = correct/total

    return test_loss, test_acc


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    train_dataset = torch.utils.data.Subset(train_dataset, indices=np.arange(N_SAMPLES))

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_dataloader = dd_exp.DataLoaderX(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                          pin_memory=True)

    test_dataloader = dd_exp.DataLoaderX(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                         pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device : ', torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))

    torch.backends.cudnn.benchmark = True

    # Initialize the models
    layer1 = models.LL_Layer(784, 40)
    #layer2 = models.LL_Layer(40, 40)
    classifier = models.LL_Classifier(40)

    layer1.to(device)
    #layer2.to(device)
    classifier.to(device)

    # Set the optimizer
    optimizer1 = torch.optim.SGD(layer1.parameters(), lr=0.05)
    #optimizer2 = torch.optim.SGD(layer2.parameters(), lr=0.1)
    optimizer3 = torch.optim.SGD(classifier.parameters(), lr=0.05)

    # Set the criterion
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    for epoch in range(N_EPOCHS):
        layer1, feature_dataset_1, train_loss, train_acc = train_layer(layer1, device, train_dataloader, optimizer1, criterion)
        print("Epoch = %d ; Layer_1 : Train Loss = %f ; Train Acc = %.3f" % (epoch, train_loss, train_acc))

        feature_dataset_1 = datasets_.ListDataset(feature_dataset_1)
        print("Convert to ListDataset!")
        feature_dataloader_1 = dd_exp.DataLoaderX(feature_dataset_1, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                                  pin_memory=False)
        print("Convert to DataLoaderX!")

        '''
        layer2, feature_dataset_2, train_loss, train_acc = train_layer(layer2, device, feature_dataloader_1, optimizer2, criterion)
        print("Epoch = %d ; Layer_2 : Train Loss = %f ; Train Acc = %.3f" % (epoch, train_loss, train_acc))

        feature_dataset_2 = datasets_.ListDataset(feature_dataset_2)
        print("Convert to ListDataset!")
        feature_dataloader_2 = dd_exp.DataLoaderX(feature_dataset_2, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                                  pin_memory=False)
        print("Convert to DataLoaderX!")
        '''

        classifier, train_loss, train_acc = train_classifier(classifier, device, feature_dataloader_1, optimizer3, criterion)

        print("Epoch = %d ; Classifier : Train Loss = %f ; Train Acc = %.3f" % (epoch, train_loss, train_acc))

        test_loss, test_acc = test(layer1, classifier, device, test_dataloader)

        # Print training and evaluation outcome
        print("Train Loss : %f ; Train Acc : %.3f ; Test Loss : %f ; Test Acc : %.3f\n"
              % (train_loss, train_acc, test_loss, test_acc))



        optimizer1.param_groups[0]['lr'] = 0.05 / pow(1 + epoch, 0.5)
        #optimizer2.param_groups[0]['lr'] = 0.1 / pow(1 + epoch, 0.5)
        optimizer3.param_groups[0]['lr'] = 0.05 / pow(1 + epoch, 0.5)
        print("\nLearning Rate : ", optimizer1.param_groups[0]['lr'])


'''
Train Loss : 0.086070 ; Train Acc : 0.976 ; Test Loss : 0.108997 ; Test Acc : 0.967
Learning Rate :  0.005
'''