from split_dataset import create_tf_dataset

if __name__ == "__main__":
    train_ds, valid_ds, test_ds = create_tf_dataset()

    print("Train")
    for x, y in train_ds.take(1):
        print(x.shape, y)

    print("Valid")
    for x, y in valid_ds.take(1):
        print(x.shape, y)

    print("Test")
    for x, y in test_ds.take(1):
        print(x.shape, y)