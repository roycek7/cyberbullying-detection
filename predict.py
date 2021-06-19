import numpy as np
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

input_size = (512, 512)
nb_samples = 1005


def feature_extractor_pred(img):
    conv_base = VGG19(weights='imagenet', include_top=False)
    return conv_base.predict(img)


def load_image(img_path):
    img = image.load_img(img_path, target_size=input_size)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    return img_tensor


# Predict Image
pred_feature = np.reshape(feature_extractor_pred(
    load_image('C:/Users/s4625266/Dropbox/UCE Corals/19.GBR.25/PB083463.JPG')), (1, 256, 256, 2))

model = load_model('C:/Users/s4625266/PycharmProjects/coral/log/coral_classification.h5')

prediction = model.predict(pred_feature)

print(np.argmax(prediction[0]) + nb_samples, np.argmax(prediction[1])
      + nb_samples, np.argmax(prediction[2]) + nb_samples, np.argmax(prediction[3]) +
      nb_samples, np.argmax(prediction[4]) + nb_samples, np.argmax(prediction[5]) + nb_samples)
