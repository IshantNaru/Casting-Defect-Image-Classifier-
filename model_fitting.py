import numpy as np
from tensorflow import keras
from keras import Sequential, layers
from keras.models import load_model
from keras.optimizers import adam_v2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing import image


def modelBuilding():
    # Building the convolutional model

    cast = Sequential([
        layers.Conv2D(32, (3, 3), input_shape=(300, 300, 1), activation='relu', padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=2),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=2),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=2),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), strides=2),
        layers.Flatten(),
        layers.Dense(300, activation='relu'),
        layers.Dropout(rate=0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compilation with loss function, optimizers and eval metrics
    cast.compile(loss='binary_crossentropy',
                 optimizer=adam_v2.Adam(learning_rate=1e-5),
                 metrics=['accuracy'])
    print("Compilation was successful...")
    return cast


def modelFitting(train_gen, val_gen):
    cnn = modelBuilding()
    print(cnn.summary())

    # Creating Checkpoints
    checkpoint_cb = ModelCheckpoint("Cast_Def.h5",
                                    save_best_only=True)
    early_stopping_cb = EarlyStopping(min_delta=0.00001,
                                      patience=8,
                                      restore_best_weights=True)
    print("Checkpoints successfully created...")

    try:
        history = cnn.fit(train_gen,
                          epochs=30,
                          steps_per_epoch=52,
                          validation_data=val_gen,
                          validation_steps=30,
                          callbacks=[checkpoint_cb, early_stopping_cb],
                          verbose=1)
        return history
    except Exception as e:
        print(e)


def trained_model(image_name):
    # loading the trained model
    model1 = load_model("Cast_Def.h5")

    test_image = image.load_img(image_name, color_mode='grayscale', target_size=(300, 300))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model1.predict(test_image)

    if np.round(result)[0][0] == 1:
        prediction = 'Casting Defect Present'
        return [{"image": prediction}]
    else:
        prediction = 'Casting Defect Absent'
        return [{"image": prediction}]
