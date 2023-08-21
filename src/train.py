import tensorflow as tf
import time
from split_dataset import create_tf_dataset

def data_preprocessing_layers():
    return tf.keras.Sequential([
        tf.keras.layers.Resizing(height=224, width=224, crop_to_aspect_ratio=True),
        tf.keras.layers.Lambda(tf.keras.applications.resnet50.preprocess_input)
    ])

def build_model(trainable_layers=None):
    n_classes = 4

    base_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False)
    if trainable_layers is None:
        base_model.trainable = False
    else:
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = True

    avg = tf.keras.layers.GlobalAvgPool2D()(base_model.output)
    outputs = tf.keras.layers.Dense(
        n_classes,
        activation="softmax"
    )(avg)
    model = tf.keras.Model(inputs=base_model.inputs, outputs=outputs)
    # print(model.summary())

    return model

def fit(model, train_ds, valid_ds, batch_size=32, epochs=10):
    train_ds = train_ds.shuffle(100, seed=170892).batch(batch_size=batch_size).prefetch(1)

    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    # Metrics
    train_f1_score_metric = tf.keras.metrics.F1Score(average="weighted")
    valid_f1_score_metric = tf.keras.metrics.F1Score(average="weighted")

    for epoch in range(epochs):
        print(f"\nStart of epoch {epoch}/{epochs}")
        start_time = time.time()

        for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)

                loss_value = loss_fn(y_batch_train, logits)
                train_f1_score_metric.update_state(y_batch_train, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)

            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if step % 10 == 0:
                print(f"(for one batch) At step {step}. Training loss: {float(loss_value)}. Training F1score: {float()}")
                print(f"Seen so far: {(step + 1) * batch_size}")

        # Display Training metric at the end of an epoch
        train_f1_score = train_f1_score_metric.result()
        train_f1_score_metric.reset_states()

        # Run a validation loop at the end of each epoch
        for x_batch_valid, y_batch_valid in valid_ds:
            valid_logits = model(x_batch_valid, training=False)
            valid_f1_score_metric.update_state(y_batch_valid, valid_logits)
        valid_f1_score = valid_f1_score_metric.result()
        valid_f1_score_metric.reset_states()

        print(f"Training f1 score over epoch: {float(train_f1_score)}. Validation v1 score over epoch: {float(valid_f1_score)}")
        print(f"Time taken: {time.time() - start_time}\n----------------------------------\n")

def run():
    tf.keras.backend.clear_session()
    tf.random.set_seed(170892)

    # getting the data
    train_ds, valid_ds, test_ds = create_tf_dataset()

    # define hyperparameter
    batch_size = 32

    # preprocess the data
    preprocess = data_preprocessing_layers()
    train_set = train_ds.map(lambda X, y: (preprocess(X), y))
    valid_set = valid_ds.map(lambda X, y: (preprocess(X), y))
    test_set = test_ds.map(lambda X, y: (preprocess(X), y))

    # model
    model = build_model(trainable_layers=5)

    # compiling the model
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    # model.compile(
    #     loss="sparse_categorical_crossentropy", # tf.keras.losses.SparseCategoricalCrossentropy(),
    #     optimizer=optimizer,
    #     metrics=["accuracy"] # tf.keras.metrics.F1Score()
    # )
    #
    # # fitting
    # model.fit(
    #     train_set,
    #     validation_data=valid_set,
    #     epochs=10
    # )
    fit(
        model,
        train_set,
        valid_set,
        epochs=10
    )


if __name__ == "__main__":
    run()
    # build_model(trainable_layers=10)
    # train_ds, valid_ds, test_ds = create_tf_dataset()
    #
    # print("Train")
    # for x, y in train_ds.take(1):
    #     print(x.shape, y)
    #
    # print("Valid")
    # for x, y in valid_ds.take(1):
    #     print(x.shape, y)
    #
    # print("Test")
    # for x, y in test_ds.take(1):
    #     print(x.shape, y)