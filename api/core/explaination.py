import shap
from matplotlib import pyplot as plt
from secrets import token_hex
import threading, os
from django.conf import settings

INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 1
media_url = settings.MEDIA_URL
media_root = settings.MEDIA_ROOT

shap_output = dict()
thread_manager = dict() 

def create_shap_async(id, model, x, ground_truth, classes):
    
    # define a masker that is used to mask out partitions of the input image.
    masker = shap.maskers.Image("blur(128,128)", INPUT_SHAPE)
    
    # create an explainer with model and image masker
    explainer = shap.Explainer(model, masker, output_names=classes)
    shap_values = explainer(x, max_evals=512, batch_size=BATCH_SIZE, outputs=shap.Explanation.argsort.flip[:len(classes)])
    
    # output with shap values
    shap.image_plot(shap_values, true_labels=[ground_truth], labelpad=10, show=False)
    foldpath = media_root
    os.makedirs(foldpath, exist_ok=True)
    savepath = os.path.join(foldpath, f'{id}.png')
    plt.savefig(savepath, dpi=72, bbox_inches='tight')
    shap_output[id] = f'{media_url}/{id}.png'
    

def create_shap(model, x, ground_truth, classes):
    
    id = token_hex(16)
    thread = threading.Thread(target=create_shap_async, args=(id, model, x, ground_truth, classes))
    thread_manager[id] = thread
    thread.start()
    return id

def get_shap_url(id):
    return shap_output.get(id)