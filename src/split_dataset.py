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

def load_and_preproces_image(x, y):
    image_raw = tf.io.read_file(x)
    image = tf.image.decode_image(image_raw, expand_animations = False, channels=3)
    image = tf.image.resize(image, [235, 80])
    image = tf.cast(image, tf.float32) / 255.
    return tf.data.Dataset.from_tensors((image, y))

def dataframe_to_tf_dataset(df):
    ds = tf.data.Dataset.from_tensor_slices((
        df.Image.apply(lambda i: os.path.join(os.getcwd(), "dataset/Train Images", i)), # x.numpy().decode("utf-8"))),
        df.Class.replace({"Food": 0, "Attire": 1, "misc": 2, "Decorationandsignage": 3})
    ))

    n_readers = 5
    ds = ds.interleave(
        lambda x, y: load_and_preproces_image(x, y),
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

