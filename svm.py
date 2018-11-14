# Quickly add random forests
# Include NDVI Data
# Split into small groups and then calculate on the center pixel
# Sample lots of areas based on the mask, gen random samples to place centers at and make sure some are transition areas and some fully inside/outside.
import matplotlib

matplotlib.use('macosx')

import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt

mask_truth = plt.imread("/Users/luc/Desktop/247-2528-1613-2017-07-31/247-2528-1613-2017-07-31_mask.tif")

# Reads in as red,green,blue,alpha as expected
scan = plt.imread("/Users/luc/Desktop/247-2528-1613-2017-07-31/247-2528-1613-2017-07-31.tif")

# Remove alpha
scan = scan[:, :, :3]

X = scan
Y = mask_truth

plt.figure(1)
plt.imshow(Y[990:1090, 2030:2130])

plt.figure(2)
plt.imshow(X[990:1090, 2030:2130, :])

#
Y = Y[990:1090, 2030:2130]
X = X[990:1090, 2030:2130, :]

shape1 = Y.shape

X = X.reshape(-1, 3)
Y = Y.reshape(-1)
Y = np.clip(Y, 0, 1)

C = 1.0  # SVM regularization parameter

svc = svm.SVC(kernel='linear', C=C).fit(X, Y)
Z = svc.predict(X)

plt.figure(3)
plt.imshow(Z.reshape(shape1))

plt.show()
