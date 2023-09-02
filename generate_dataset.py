import argparse
import datasets
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Double Descent Experiment')
    parser.add_argument('--dataset', choices=['MNIST', 'CIFAR-10'], type=str, help='dataset')
    parser.add_argument('--sample_size', type=int, help='number of samples used as training data')
    parser.add_argument('--noise_ratio', type=float, help='label noise ratio')
    parser.add_argument('--model', choices=['SimpleFC', 'CNN', 'ResNet18'], type=str,
                        help='neural network architecture')

    parser.add_argument('--group', type=int, help='TEST GROUP')
    parser.add_argument('--start', type=int, help='starting number of test number')
    parser.add_argument('--end', type=int, help='ending number of test number')

    args = parser.parse_args()
    print(args)


    for test_number in range(args.start, args.end + 1):
        directory = f"assets/{args.dataset}-{args.model}/N=%d-3d/TEST-%d/GS=%dK-noise-%d-model-%d-sgd" \
                % (args.sample_size, args.group, args.gradient_step // 1000, args.noise_ratio * 100, test_number)

        dataset_path = os.path.join(directory, 'dataset')

        if not os.path.isdir(dataset_path):
            os.mkdir(dataset_path)

        datasets.generate_train_dataset(dataset=args.dataset, sample_size=args.sample_size,
                                            label_noise_ratio=args.noise_ratio, dataset_path=dataset_path)

        print('Dataset Generated for test number %d' % test_number)
