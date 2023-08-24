import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import csv
import os


#DATASET = 'MNIST'
#N_SAMPLES = 4000

DATASET = 'CIFAR-10'
N_SAMPLES = 50000

TEST_GROUP = 2
TEST_NUMBER = 0
label_noise_ratio = 0.2

GRADIDENT_STEP = 100000
BATCH_SIZE = 128
learning_rate = 0.1
save_model = True

directory = f"assets/{DATASET}/N=%d-3d/TEST-%d/GS=%dK-noise-%d-model-%d" \
                    % (N_SAMPLES, TEST_GROUP, GRADIDENT_STEP // 1000, label_noise_ratio * 100, TEST_NUMBER)

dictionary_path = os.path.join(directory, "dictionary.csv")
plots_path = os.path.join(directory, 'plots')

if not os.path.isdir(plots_path):
    os.mkdir(plots_path)

index, hidden_units, parameters, gradient_steps = [], [], [], [],
train_losses, train_accs, test_losses, test_accs = [], [], [], []

with open(dictionary_path, "r", newline="") as infile:
    # Create a reader object
    reader = csv.DictReader(infile)

    for row in reader:
        if len(gradient_steps) == 0 or int(row['Gradient Steps']) < gradient_steps[-1][-1]:
            hidden_units.append([])
            parameters.append([])
            gradient_steps.append(([]))
            train_losses.append([])
            train_accs.append([])
            test_losses.append([])
            test_accs.append([])

        hidden_units[-1].append(int(row['Hidden Neurons']))
        parameters[-1].append(int(row['Parameters']))
        gradient_steps[-1].append(int(row['Gradient Steps']))
        train_losses[-1].append(float(row['Train Loss']))
        train_accs[-1].append(float(row['Train Accuracy']))
        test_losses[-1].append(float(row['Test Loss']))
        test_accs[-1].append(float(row['Test Accuracy']))

hidden_units = np.array(hidden_units)
parameters = np.array(parameters)
gradient_steps = np.array(gradient_steps)
train_losses = np.array(train_losses)
train_accs = np.array(train_accs)
test_losses = np.array(test_losses)
test_accs = np.array(test_accs)

print(hidden_units)
print(gradient_steps)


my_col = cm.jet(test_accs/np.amin(test_accs))
scale_function = (lambda x: x ** (1 / 4), lambda x: x ** 4)

fig2 = plt.figure(figsize=(15, 10))
ax = plt.axes(projection='3d')
ax.plot_surface(hidden_units, gradient_steps // 1000, train_losses, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#ax.set_xscale('function', functions=scale_function)
ax.set_zlabel('Train Losses')
plt.xlabel('Number of Hidden Units (H)')
plt.ylabel('Number of Gradient Steps')
plt.title(f'Double Descent on {DATASET} (N = %d)' % N_SAMPLES)
plt.savefig(os.path.join(plots_path, 'Train_Losses-Hidden_Units-3D.png'))

fig3 = plt.figure(figsize=(15, 10))
ax2 = plt.axes(projection='3d')
ax2.plot_surface(hidden_units, gradient_steps // 1000, test_losses, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#x2.set_xscale('function', functions=scale_function)
ax2.set_zlabel('Test Losses')
plt.xlabel('Number of Hidden Units (H)')
plt.ylabel('Number of Gradient Steps')
plt.title(f'Double Descent on {DATASET} (N = %d)' % N_SAMPLES)
plt.savefig(os.path.join(plots_path, 'Test_Losses-Hidden_Units-3D.png'))
