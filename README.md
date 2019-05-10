# CSC2005Z-TreeClassification
Second year research project (CSC2005Z at UCT): Tree presence classification using ML models
This project explores the effectiveness of Support Vector machine and Random Forest classifiers for tree detection at the per pixel level using python.

Working with drone scan data provided by Aerobotics, a comparison was performed of the value to use SVM and Random Forest classifiers to create binary masks of tree locations in a scan using RGB and NDVI data.

# Abstract:
The abundance of relatively low-cost consumer drones on
the market has opened up new opportunities for high quality
aerial scans to be captured. Analysing these scans by hand is
both time-consuming and error prone. This report examines
the viability of using machine learning models to generate
bitmasks representing the location of trees in an image using
a combination of colour photography and near infrared data.
Support Vector Machines and Random Forest classifiers
were chosen as they are popular for use in binary
classification scenarios such as this. A combination of
analytical tests and qualitative comparisons against
manually generated ground truths were used to evaluate the
effectiveness of the classifiers. It was found that the
Random Forest classifier produced the most accurate results
without compromising on training time for the model.

# Conclusion:
Initial visual comparisons
suggested that both models were able to outperform the
typical standard that had been deemed acceptable for the
hand created masks.
Further experimental tests showed that the initial visual
perceptions were correct and that both SVM and Random
Forest classifiers produced excellent results. By comparing
the results of both classifiers, it is clear that in the current
use case, both models produce masks which are
quantitatively similar to within a margin of error. However,
the Random Forest model trained faster and produced high
quality results in significantly shorter amounts of time. This
makes it a far better solution as the rate of data collection
and the resolution of that data is only likely to increase in
the future.
