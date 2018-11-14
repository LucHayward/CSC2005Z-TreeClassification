from sklearn.ensemble import RandomForestClassifier
import input

import numpy as np
import matplotlib

import matplotlib.pyplot as plt

# X = input.get_samples("/Users/luc/Desktop/247-2528-1613-2017-07-31/247-2528-1613-2017-07-31.tif")
# Y = input.get_classes("/Users/luc/Desktop/247-2528-1613-2017-07-31/247-2528-1613-2017-07-31_mask.tif")
X, Y = input.get_scans("/Users/luc/Desktop/247-2528-1613-2017-07-31")
inputimage = X.copy()

startRow, endRow, startCol, endCol = 990, 1090, 2030, 2130

# Preprocessing and visualisation
# plt.figure(1)
# plt.imshow(Y[startRow:endRow, startCol:endCol])

# plt.figure(2)
# plt.imshow(X[startRow:endRow, startCol:endCol, :3])

#
Y = Y[startRow:endRow, startCol:endCol]
X = X[startRow:endRow, startCol:endCol, :]

shapeY, shapeX, shapeInputImage = Y.shape, X.shape, inputimage[:, :, 0].shape

X = X.reshape(-1, 4)
Y = Y.reshape(-1)
Y = np.clip(Y, 0, 1)

import time

start_time = time.time()
rf_model = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
rf_model.fit(X, Y)

print("Trained model")
print("Time to train: " + str(time.time() - start_time) + "seconds for " + str(
    (endCol - startCol) * (endRow - startRow)) + " samples")


def predict_mask(folder_path):
    inputimage, fuckoff = input.get_scans(folder_path)
    shapeInputImage = inputimage[:, :, 0].shape

    start_time = time.time()

    Z = rf_model.predict(inputimage.reshape(-1, 4))

    # print("Finished predicting")
    print("Time to predict: " + str(time.time() - start_time) + " seconds for a " + str(shapeInputImage) + " scan")
    print(folder_path)
    # plt.figure(3)
    # plt.imshow(Z.reshape(shapeInputImage))
    # plt.imshow(Z.reshape(shapeY))

    # plt.show()

    from PIL import Image
    from scipy.ndimage import median_filter
    Z = Z.reshape(shapeInputImage)
    Z = median_filter(Z*255, size=10)
    # print(np.unique(Z))
    np.round(Z, 0)
    # print(np.unique(Z))
    im = Image.fromarray((Z), 'L')
    im.save("prediction.tif")


