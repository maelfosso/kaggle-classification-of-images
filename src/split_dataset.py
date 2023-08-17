import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df = pd.read_csv("dataset/train.csv")

    x = df.Image
    y = df.Class

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x, y, test_size=0.1, random_state=170892, stratify=y
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_full, y_train_full,
        test_size=0.15, random_state=170892, stratify=y_train_full
    )

    df_train = pd.DataFrame({"Image": x_train, "Class": y_train})
    df_valid = pd.DataFrame({"Image": x_valid, "Class": y_valid})
    df_test = pd.DataFrame({"Image": x_test, "Class": y_test})

    df_train.to_csv("./inputs/train.csv", index=False)
    df_valid.to_csv("./inputs/valid.csv", index=False)
    df_test.to_csv("./inputs/test.csv", index=False)

