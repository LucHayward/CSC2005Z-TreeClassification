import matplotlib

matplotlib.use('macosx')

from sklearn.ensemble import RandomForestClassifier
import input

import numpy as np
from sklearn import svm, datasets

import matplotlib.pyplot as plt

# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

X = input.get_samples("/Users/luc/Desktop/247-2528-1613-2017-07-31/247-2528-1613-2017-07-31.tif")
Y = input.get_classes("/Users/luc/Desktop/247-2528-1613-2017-07-31/247-2528-1613-2017-07-31_mask.tif")

# Preprocessing and visualisation
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

rf_model = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=3)
rf_model.fit(X, Y)

Z = rf_model.predict(X)

plt.figure(3)
plt.imshow(Z.reshape(shape1))

plt.show()

# for numFeatures in range(1, 5):
#     samples = X[:, :numFeatures]
#
#     rf_model = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=3)
#     rf_model.fit(samples, y)
#
#     Z = rf_model.predict(samples)
#     print('accuracy:', numFeatures, '::', sum(Z == y) / len(y))
#     print('--------------------------')
