import cv2
import numpy as np
from keras.api.preprocessing.image import img_to_array
from keras.api.applications.mobilenet_v2 import preprocess_input

def get_face_embedding(model, face_img):
    try:
        if face_img is None or face_img.size == 0:
            return None

        face_img = cv2.resize(face_img, (224, 224))
        face_img = img_to_array(face_img)
        face_img = np.expand_dims(face_img, axis=0)
        face_img = preprocess_input(face_img)

        embedding = model.predict(face_img, verbose=0)[0]
        return embedding
    except Exception as e:
        print("Error embedding face:", e)
        return None
