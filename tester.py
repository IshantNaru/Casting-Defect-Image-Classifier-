from keras.preprocessing.image import ImageDataGenerator as IDG
from keras.models import load_model
import numpy as np
import pandas as pd

base_folder = "C:/Users/Ishant Naru/Desktop/Project2/Pred"

pred_generator = IDG(rescale=1. / 255).flow_from_directory(base_folder,
                                                           batch_size=2,
                                                           target_size=(300, 300),
                                                           color_mode='grayscale',
                                                           class_mode=None,
                                                           interpolation='nearest',
                                                           shuffle=False)

model1 = load_model("Cast_Def.h5")

result = model1.predict(pred_generator, steps=8)
print(result)

cl = np.round(result)
class_list = cl.tolist()
filenames = pred_generator.filenames

df = pd.DataFrame({"Name": filenames, "predictions": result.tolist(), "class": class_list})
print(df)
# image2 = base_folder + "/Cancer (7).jpg"
# image3 = base_folder + "/Cancer (12).jpg"
# image4 = base_folder + "/Not Cancer  (9).jpg"
# image5 = base_folder + "/Not Cancer  (27).jpg"
# image6 = base_folder + "/Not Cancer  (31).jpg"
#
# images = [image1, image2, image3, image4, image5, image6]
#
# for image in images:
#     classifier = model(image_file=image)
#     result = classifier.trained_model()
#     print(result)
