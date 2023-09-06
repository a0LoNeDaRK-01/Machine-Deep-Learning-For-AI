import numpy as np
from PIL import Image
from IPython.display import Image as show_img
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResnetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input,decode_predictions

img = Image.open("submarine.jpg").resize((299,299))
img = np.array(img)
print(img.shape)
print(img.ndim)

img = img.reshape(-1,299,299,3)
print(img.shape)
print(img.ndim)

img = preprocess_input(img)

incresV2_model = InceptionResnetV2(weights="imagenet",classes=1000)
print(incresV2_model.summary())
print(type(incresV2_model))

show_img(filename="submarine.jpg")

preds = incresV2_model.predict(img)
print("Predicted Categories: ",decode_predictions(preds,top=2)[0])


