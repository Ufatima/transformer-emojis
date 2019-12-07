import os
import random
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Split dataset into train dev and test sets")
    parser.add_argument(
        "--input", type=str, help="Path to input dataset", required=True
    )
    parser.add_argument(
        "--output", type=str, help="Where to save the splitted dataset.", required=True
    )
    parser.add_argument(
        "--dev_size",
        type=int,
        help="Size of the dev set (train = total - dev - test)",
        required=True,
    )
    parser.add_argument(
        "--test_size",
        type=int,
        help="Size of the test set (train = total - dev - test)",
        required=True,
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for the random generator. Will not shuffle if not provided",
        required=False,
    )

    args = parser.parse_args()

    print("Loading data...")
    with open(args.input, "r") as f:
        data = f.readlines()

    if args.random_seed:
        print("Shuffling data...")
        random.seed(args.random_seed)
        random.shuffle(data)

    print("Splitting data..")
    dev_set = data[: args.dev_size]
    test_set = data[args.dev_size : args.dev_size + args.test_size]
    train_set = data[args.dev_size + args.test_size :]

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    datasets = [
        {"type": "dev", "data": dev_set},
        {"type": "test", "data": test_set},
        {"type": "train", "data": train_set},
    ]

    print("Saving data..")
    for dataset in datasets:
        out_path = os.path.join(args.output, dataset["type"] + ".tsv")
        print(f"Saving data to {out_path}")
        with open(out_path, "w+") as f:
            for datum in dataset["data"]:
                f.write(datum)
