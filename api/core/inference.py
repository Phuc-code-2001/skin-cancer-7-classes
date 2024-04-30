import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np

from .explaination import create_shap, get_shap_url

IMG_SIZE = 224

model : keras.Model = keras.models.load_model(r'.\api\cnn-models\skin-cancer-7-classes_CustomModel_ph2_model.h5')
model.trainable = False
model.summary()

classes_dict = {
    'bkl': "benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses)", 
    'nv': "melanocytic nevi",
    'df': "dermatofibroma", 
    'mel': "melanoma", 
    'vasc': "vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)", 
    'bcc': "basal cell carcinoma", 
    'akiec': "Actinic keratoses and intraepithelial carcinoma / Bowen's disease"
}

classes_group_dict = {
    'dangerous': ['mel', 'akiec'],
    'non-dangerous': ['bkl', 'nv', 'df', 'vasc', 'bcc']
}

classes = [
    'bkl', 
    'nv', 
    'df', 
    'mel', 
    'vasc', 
    'bcc', 
    'akiec'
]
num_classes = len(classes)

def predict(file):

    bytes = file.read()
    img = cv2.imdecode(np.frombuffer(bytes , np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    x = np.expand_dims(img, axis=0)
    probs = model.predict(x)[0]
    
    shap_id = create_shap(model, x, "unknown", classes)
    
    results = {
        'class_dict': classes_dict,
        'label': classes[np.argmax(probs)],
        'confidence': np.max(probs),
        'probs': { classes[i]: prob for i, prob in enumerate(probs) },
        'shap_id': shap_id
    }
    
    return results