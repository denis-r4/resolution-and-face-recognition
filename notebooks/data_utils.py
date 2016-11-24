import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize


def prep_image(path):
    img = plt.imread(path)
    # ('nearest', 'bilinear', 'bicubic')
    img = imresize(img, (224, 224), interp='bilinear')

    # Shuffle axes to (3, 224, 224)
    img = np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)
    # Convert to BGR
    img = img[::-1, :, :]

    if img.shape[0] == 4:
        img = img[1:4, :, :]
    # Add extra axes
    img = img[np.newaxis, :]
    return img
