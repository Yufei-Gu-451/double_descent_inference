import argparse
import datasets
import os

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

    parser.add_argument('--epochs', type=int, help='epochs of training time')

    args = parser.parse_args()
    print(args)

    if not os.path.isdir(f"assets"):
        os.mkdir(f"assets")
    if not os.path.isdir(f"assets/{args.dataset}-{args.model}"):
        os.mkdir(f"assets/{args.dataset}-{args.model}")
    if not os.path.isdir(f"assets/{args.dataset}-{args.model}/N=%d-3d" % args.sample_size):
        os.mkdir(f"assets/{args.dataset}-{args.model}/N=%d-3d" % args.sample_size)
    if not os.path.isdir(f"assets/{args.dataset}-{args.model}/N=%d-3d/TEST-%d" % (args.sample_size, args.group)):
        os.mkdir(f"assets/{args.dataset}-{args.model}/N=%d-3d/TEST-%d" % (args.sample_size, args.group))

    for test_number in range(args.start, args.end + 1):
        directory = f"assets/{args.dataset}-{args.model}/N=%d-3d/TEST-%d/Epoch=%d-noise-%d-model-%d-sgd" \
                    % (args.sample_size, args.group, args.epochs, args.noise_ratio * 100, test_number)

        if not os.path.isdir(directory):
            os.mkdir(directory)

        dataset_path = os.path.join(directory, 'dataset')
        checkpoint_path = os.path.join(directory, "ckpt")
        dictionary_path = os.path.join(directory, 'dictionary')

        if not os.path.isdir(dataset_path):
            os.mkdir(dataset_path)
        if not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)
        if not os.path.isdir(dictionary_path):
            os.mkdir(dictionary_path)

        datasets.generate_train_dataset(dataset=args.dataset, sample_size=args.sample_size,
                                            label_noise_ratio=args.noise_ratio, dataset_path=dataset_path)

        print('Dataset Generated for test number %d' % test_number)
