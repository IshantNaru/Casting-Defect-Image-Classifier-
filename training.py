import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import imagepreprocess
from model_fitting import modelFitting
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator as IDG

train = "C:/Users/Ishant Naru/Desktop/Project2/casting_data/casting_data/train"
test = "C:/Users/Ishant Naru/Desktop/Project2/casting_data/casting_data/test"

# Loading the images and processing them
train_generator, validation_generator = imagepreprocess(train)

# training the model
# try:
#     history = modelFitting(train_generator, validation_generator)
# except Exception as e:
#     print(e)

model1 = load_model("Cast_Def.h5")

test_generator = IDG(1. / 255).flow_from_directory(test,
                                                   color_mode='grayscale',
                                                   target_size=(300, 300),
                                                   batch_size=72,
                                                   classes={'ok_front': 0, 'def_front': 1},
                                                   class_mode='binary',
                                                   seed=45)

model1.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])

# Evaluating the model performance on test dataset
model1.evaluate(test_generator, steps=10)

# hist_df = pd.DataFrame(history.history)
# #Displaying the results
# try:
#     pd.DataFrame(hist.history).plot(figsize=(10,7))
#     plt.grid(True)
#     plt.gca().set_ylim(0,1)
#     plt.show()
# except Exception as e:
#     print(e)
