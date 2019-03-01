import os
import numpy as np
import csv
import re
import pandas as pd
from sklearn.model_selection import train_test_split


# Returns a dictionary of labels (keys) and the number of images that label has (values)
def getImageOccurrence(csvFile, minNumImgsPerLabel):
    images = dict()
    df = pd.read_csv(csvFile)
    # Iterate through each row in the CSV
    for _, row in df.iterrows():
        # If the entry in the CSV has the specified minimum # of images per label, add it to the dict
        if row['images'] >= minNumImgsPerLabel:
            images[row['name']] = row['images']
    return images


# Creates a CSV that tracks how many images there are for each label
def createLFWCropCSV(directory):
    face_dict = dict()

    for filename in os.listdir(directory + "/faces"):
        # Split the filename to get just the name
        split_pos = filename.rfind("_")
        name = filename[:split_pos]
        # Either add the label to the dict or increment the label's value
        if name in face_dict:
            face_dict[name] = face_dict[name] + 1
        else:
            face_dict[name] = 1

    # Write or overwrite the CSV
    with open(directory + '/people.csv', 'w', newline="\n", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(("name", "images"))
        w.writerows(face_dict.items())


# Load a .pgm image (simple greyscale image format) into a numpy array
def loadPGMImage(filename):
    # Read the file
    with open(filename, 'rb') as f:
        buffer = f.read()

    try:
        # Try to parse a valid header
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)

    # If the file buffer doesn't fit the specified header and image size, attempt to correct the header length
    # The correction assumes the image size is correct
    if len(header) + (int(width) * int(height)) > len(buffer):
        print("Slicing PGM header of " + filename + " for correction.")
        corrected_size = len(buffer) - (int(width) * int(height))
        header = header[:corrected_size]

    # Create a numpy array from the buffer and shape it into the image size
    return np.frombuffer(buffer,
                         dtype='u1' if int(maxval) < 256 else ">" + 'u2',
                         count=int(width) * int(height),
                         offset=len(header)).reshape((int(height), int(width)))


# Load a .ppm image (simple RGB image format) into a numpy array
def loadPPMImage(filename):
    # Read the file
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        # Try to parse a valid header
        header, width, height, maxval = re.search(
            b"(^P6\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PPM file: '%s'" % filename)

    # If the file buffer doesn't fit the specified header and image size, attempt to correct the header length
    # The correction assumes the image size is correct
    if len(header) + (int(width) * int(height)) > len(buffer):
        print("Slicing PPM header of " + filename + " for correction.")
        corrected_size = len(buffer) - (int(width) * int(height))
        header = header[:corrected_size]

    # Create a numpy array from the buffer
    image_matrix = np.frombuffer(buffer,
                                 dtype=np.dtype(np.uint8) if int(maxval) < 256 else ">" + 'u2',
                                 count=int(width) * int(height) * 3,
                                 offset=len(header))

    # Extract the RGB components into different layers
    # Get every 3rd pixel starting from pixel 0 (Red), 1 (Green), and 2 (Blue)
    shaped_matrix = np.zeros((int(width), int(height), 3), dtype=np.dtype(np.uint8))
    shaped_matrix[:, :, 0] = image_matrix[0::3].reshape(64, 64)
    shaped_matrix[:, :, 1] = image_matrix[1::3].reshape(64, 64)
    shaped_matrix[:, :, 2] = image_matrix[2::3].reshape(64, 64)

    return shaped_matrix


# Get an array of PGM/PPM images and their labels, given a dict of the labels and the number of images each has
def getPXMImages(img_occ, directory, type="pgm"):
    loaded_imgs = []
    labels = []
    img_occs = list(img_occ.items())

    for label, num in img_occs:
        # Generate filenames for each image
        for img_num in range(1, num + 1):
            filename = "{}_{:04d}.{}".format(label, img_num, type)
            # Load the file with the approach function
            if type == "pgm":
                loaded_imgs.append(loadPGMImage(directory + "/faces/" + filename))
            elif type == "ppm":
                loaded_imgs.append(loadPPMImage(directory + "/faces/" + filename))
            # Keep track of the label
            labels.append(label)

    return np.array(loaded_imgs), np.array(labels)


# Get datasets from LFWCrop data
def getLFWCropData(minNumImgsPerLabel, RGB=False, split=None):
    # Set for either the greyscale data or the RGB data
    directory = "lfwcrop_grey"
    type = "pgm"
    if RGB:
        directory = "lfwcrop_color"
        type = "ppm"

    # Generate the image occurrence CSV if it doesn't exist for the data yet
    if not os.path.isfile('/people.csv'):
        createLFWCropCSV(directory)

    # Get a subset of the data that has a minimum number of images for each label
    img_occ = getImageOccurrence(directory + '/people.csv', float(minNumImgsPerLabel))

    # Load all of the images and labels
    X, y = getPXMImages(img_occ, directory, type)

    # Either return as is, or optionally split the data into a training and test set
    if split is None:
        return X, y
    else:
        return train_test_split(X, y, test_size=1 - split)


# Get the number of unique classes in a subset of the data that has a minimum number of images for each label
def getNumClasses(minNumImgsPerLabel, RGB=False):
    # Set to the correct dataset
    directory = "lfwcrop_grey"
    if RGB:
        directory = "lfwcrop_color"

    # Generate the image occurrence CSV if it doesn't exist for the data yet
    if not os.path.isfile('/people.csv'):
        createLFWCropCSV(directory)

    # Return the number of labels (keys) in the data subset
    return len(list(getImageOccurrence(directory + '/people.csv', float(minNumImgsPerLabel)).keys()))