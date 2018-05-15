import numpy as np;
import os;
import nibabel as nib;
import matplotlib.pyplot as plt;
import zipfile;
from io import BytesIO;
import scipy as sp;
import time;
import tensorflow as tf;
# from tensorflow.contrib.data import Iterator;
from tensorflow.contrib.layers.python.layers import regularizers;




def load_image_data(path):
  numFolders = len(os.listdir(path)) - 1;
  retVal = np.zeros((numFolders, 240 * 240 * 155));
  #WITH all 5 images, we want the above to be numFolders x 5 x 240 x 240 x 155
  curFolder = 0;

  for folder in os.listdir(path):
    if folder == ".DS_Store":
      continue;
    folderpath = path + "/" + folder;

    curImg = 0;
    
    allFiles = (os.listdir(folderpath));

    for filename in sorted(allFiles):
      if "t1ce" not in filename:
        continue;
      file = os.path.join(folderpath, filename);
      img = nib.load(file);
      img_data = img.get_data();
      retVal[curFolder] = img_data.reshape((1, 240 * 240 * 155));
    curFolder += 1;
  return retVal;


def show_slices(slices):
	fig, axes = plt.subplots(1, len(slices))
	for i, slice in enumerate(slices):
		axes[i].imshow(slice.T, cmap="gray", origin="lower")



path = "LGG";
imagesInLGG = load_image_data(path);
path = "HGG";

slice_0 = imagesInLGG[0, 0, 120, :, :];
slice_1 = imagesInLGG[0, 0, :, 120, :];
slice_2 = imagesInLGG[0, 0, :, :, 77];

show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center Slices for EPI Image")
plt.show();


