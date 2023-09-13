import torch
import torch.nn as nn
import torch.nn.functional as F

# ResNet18 ----------------------------------------------------------------------------------
'''
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, k, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = k

        self.conv1 = nn.Conv2d(3, k, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(k)
        self.layer1 = self._make_layer(block, k, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * k, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * k, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * k, num_blocks[3], stride=2)
        self.linear = nn.Linear(8 * k * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, path='all'):
        if path == 'all':
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        elif path == 'half1':
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
        elif path == 'half2':
            out = x
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        else:
            raise NotImplementedError

        return out

def ResNet18(k):
    return ResNet(BasicBlock, [2, 2, 2, 2], k=k)



# Standard CNN ----------------------------------------------------------------------------------

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), x.size(1))

class FiveLayerCNN(nn.Module):
    def __init__(self, k):
        self.n_hidden_units = k

        super(FiveLayerCNN, self).__init__()

        # Layer 0
        self.conv1 = nn.Conv2d(3, k, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(k)
        self.relu1 = nn.ReLU()

        # Layer 1
        self.conv2 = nn.Conv2d(k, 2 * k, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(2 * k)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        # Layer 2
        self.conv3 = nn.Conv2d(2 * k, 4 * k, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(4 * k)
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        # Layer 3
        self.conv4 = nn.Conv2d(4 * k, 8 * k, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(8 * k)
        self.relu4 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)

        # Layer 4
        self.pool4 = nn.MaxPool2d(4)
        self.flatten = Flatten()
        self.fc = nn.Linear(8 * k, 10, bias=True)

    def forward(self, x, path='all'):
        if path == 'all':
            x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
            x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        elif path == 'half1':
            x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
            x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
            x = x.view(x.size(0), -1)
        elif path == 'half2':
            x = self.fc(x)
        else:
            raise NotImplementedError

        return x


# Simple FC ----------------------------------------------------------------------------------
class Simple_FC(nn.Module):
    def __init__(self, n_hidden_units):
        self.n_hidden_units = n_hidden_units

        super(Simple_FC, self).__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, n_hidden_units),
            nn.ReLU()
        )

        self.classifier = nn.Linear(n_hidden_units, 10)

        if self.n_hidden_units == 1:
            torch.nn.init.xavier_uniform_(self.features[1].weight, gain=1.0)
            torch.nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)
        else:
            torch.nn.init.normal_(self.features[1].weight, mean=0.0, std=0.1)
            torch.nn.init.normal_(self.classifier.weight, mean=0.0, std=0.1)


    def forward(self, x, path='all'):
        if path == 'all':
            out = self.features(x)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
        elif path == 'half1':
            out = self.features(x)
            out = out.view(out.size(0), -1)
        elif path == 'half2':
            out = self.classifier(x)
        else:
            raise NotImplementedError

        return out


class LL_Layer(nn.Module):
    def __init__(self, n1, n2):

        super(LL_Layer, self).__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n1, n2),
            nn.ReLU()
        )

        self.classifier = nn.Linear(n2, 10)
        self.classifier.requires_grad = False

        torch.nn.init.normal_(self.features[1].weight, mean=0.0, std=0.1)
        torch.nn.init.normal_(self.classifier.weight, mean=0.0, std=0.1)

    def forward_half1(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def forward_half2(self, x):
        out = self.classifier(x)
        return out

    def forward(self, x, path='all'):
        if path == 'all':
            x = self.forward_half1(x)
            x = self.forward_half2(x)
        elif path == 'half1':
            x = self.forward_half1(x)
        elif path == 'half2':
            x = self.forward_half2(x)
        else:
            raise NotImplementedError

        return x

class LL_Classifier(nn.Module):
    def __init__(self, n):
        super(LL_Classifier, self).__init__()
        self.classifier = nn.Linear(n, 10)

        torch.nn.init.normal_(self.classifier.weight, mean=0.0, std=0.1)

    def forward(self, x):
        return self.classifier(x)
