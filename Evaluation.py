folder_path = "/Users/luc/Desktop/Aerobotics_Survey/good/1428-7911-2564-2017-11-23!"
import glob

prediction = glob.glob(folder_path+"/predictionSVM.tif")[0]
ground_truth = glob.glob(folder_path+"/*mask.tif")[0]

from skimage import io
prediction = io.imread(prediction).reshape(-1)
ground_truth = io.imread(ground_truth).reshape(-1)

import numpy as np

from sklearn.metrics import confusion_matrix

true_neg, false_pos, false_neg, true_pos = confusion_matrix(ground_truth, prediction).ravel()

print("true_neg, false_pos, false_neg, true_pos")
print((true_neg, false_pos, false_neg, true_pos))
