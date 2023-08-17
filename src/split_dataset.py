import pandas as pd
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split

def split_dataset():
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

def load_and_preproces_image(x):
    image_path = os.path.join(os.getcwd(), "dataset/Train Images", x.numpy().decode("utf-8"))
    image_raw = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_raw)
    image = tf.image.resize(image, [235, 80]) # resize to max height and max width
    image = image / 255.
    return image

def dataframe_to_tf_dataset(df):
    ds = tf.data.Dataset.from_tensor_slices((df.Image, df.Class))

    n_readers = 5
    ds = ds.interleave(
        lambda x, y: tf.data.Dataset.from_tensors(
            (tf.py_function(load_and_preproces_image, [x], [tf.float32]), y)
        ),
        cycle_length=n_readers,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return ds

def create_tf_dataset():
    df_train = pd.read_csv('./inputs/train.csv')
    df_valid = pd.read_csv('./inputs/valid.csv')
    df_test = pd.read_csv('./inputs/test.csv')

    train_ds = dataframe_to_tf_dataset(df_train)
    valid_ds = dataframe_to_tf_dataset(df_valid)
    test_ds = dataframe_to_tf_dataset(df_test)

    return train_ds, valid_ds, test_ds

if __name__ == "__main__":
    split_dataset()

