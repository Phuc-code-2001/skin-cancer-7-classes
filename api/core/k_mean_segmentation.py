from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import cv2, os

from django.conf import settings

media_url = settings.MEDIA_URL
media_root = settings.MEDIA_ROOT

def create_segmented_mask(id, img, n_clusters=2):
    
    w, h, channel = img.shape
    X = np.reshape(img, (w * h, channel))
    
    # Kmeans clustering with 3 clusters
    kmeans = KMeans(n_clusters, random_state=0, n_init='auto').fit(X)

    pred_label = kmeans.predict(X)
    pred_label = np.reshape(pred_label, (h, w))

    mask = np.zeros(shape=(h, w, channel))
    mask_colors = [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.5),
        (1.0, 1.0, 1.0),
    ]

    # Display image clustering
    fg, ax = plt.subplots(1, 2, figsize = (10, 5))
    
    ax[0].imshow(img / 255.0)
    ax[0].set_title(f"Original Image")
    ax[0].axis('off')
    
    for i in range(n_clusters):
        mask[pred_label == i] = mask_colors[i % len(mask_colors)]
        
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
     
    ax[1].imshow(mask)
    ax[1].set_title(f'Segmented Mask')
    ax[1].axis('off')
    
    foldpath = media_root
    os.makedirs(foldpath, exist_ok=True)
    savepath = os.path.join(foldpath, f'{id}-seg.png')
    fg.savefig(savepath, dpi=72, bbox_inches='tight')
    url = fr'{media_url}{id}-seg.png'
    
    return url
    