import splitfolders as sf
from keras.preprocessing.image import ImageDataGenerator as IDG


def splittingfolders(path, output):
    try:
        sf.ratio(input=path, output=output, seed=1337, ratio=(0.6, 0.2, 0.2))
    except Exception as e:
        print(e)
    else:
        print("Successfully split the folders")


def imagepreprocess(train):
    # Creating Image Data Generator train and test objects
    train_datagen = IDG(rescale=1. / 255,
                        zoom_range=0.15,
                        brightness_range=[0.8, 1.0],
                        shear_range=0.1,
                        validation_split=0.22)
    test_datagen = IDG(rescale=1. / 255)

    # Creating train and validation generators
    try:
        train_generator = train_datagen.flow_from_directory(train,
                                                            color_mode='grayscale',
                                                            target_size=(300, 300),
                                                            batch_size=100,
                                                            classes={'ok_front': 0, 'def_front': 1},
                                                            class_mode='binary',
                                                            seed=45,
                                                            subset='training')

        validation_generator = test_datagen.flow_from_directory(train,
                                                                color_mode='grayscale',
                                                                target_size=(300, 300),
                                                                batch_size=50,
                                                                classes={'ok_front': 0, 'def_front': 1},
                                                                class_mode='binary',
                                                                seed=45,
                                                                subset='validation')
        return train_generator, validation_generator

    except Exception as e:
        print("Some error loading the images")
