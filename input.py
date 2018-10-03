import matplotlib.pyplot as plt


def get_classes(file_location):
    return plt.imread(file_location)


def get_samples(file_location):
    scan = plt.imread(file_location)  # Read file in as RGBA
    return scan[:, :, :3]  # Remove alpha/depth channel
