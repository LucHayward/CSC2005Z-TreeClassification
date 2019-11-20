from sklearn.ensemble import RandomForestClassifier
import input

import numpy as np

X, Y = input.get_scans("/Users/luc/Desktop/247-2528-1613-2017-07-31")
inputimage = X.copy()

startRow, endRow, startCol, endCol = 990, 1090, 2030, 2130


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
    inputimage, _ = input.get_scans(folder_path)
    shapeInputImage = inputimage[:, :, 0].shape

    start_time = time.time()

    Z = rf_model.predict(inputimage.reshape(-1, 4))

    # print("Finished predicting")
    print("Time to predict: " + str(time.time() - start_time) + " seconds for a " + str(shapeInputImage) + " scan")
    print(folder_path)

    from PIL import Image
    from scipy.ndimage import median_filter
    Z = Z.reshape(shapeInputImage)
    Z = median_filter(Z*255, size=10)
    # print(np.unique(Z))
    np.round(Z, 0)
    # print(np.unique(Z))
    im = Image.fromarray((Z), 'L')
    im.save("predictionSVM.tif")


