from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from loadData import getLFWCropData

def run(RGB_BOOL=False, MIN_IMG_PER_LABEL=20, TRAIN_RATIO=0.75):
    print(__doc__)


    def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
        """Helper function to plot a gallery of portraits"""
        plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            plt.title(titles[i], size=12)
            plt.xticks(())
            plt.yticks(())


    def title(y_pred, y_test, i):
        pred_name = y_pred[i].rsplit(' ', 1)[-1]
        true_name = y_test[i].rsplit(' ', 1)[-1]
        return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # Get the datasets
    X_train, X_test, y_train, y_test = getLFWCropData(MIN_IMG_PER_LABEL, RGB_BOOL, TRAIN_RATIO)

    # Init the data and variables
    h = X_train.shape[2]
    w = X_train.shape[1]

    X_train = X_train.reshape((X_train.shape[0], 4096))
    X_test = X_test.reshape((X_test.shape[0], 4096))

    n_samples = len(y_train)

    target_names = np.unique(y_test)
    n_classes = target_names.shape[0]

    # Show a sample of the training images
    plot_gallery(X_train, y_train, h, w)

    plt.show()

    # #############################################################################
    # Compute a PCA (eigenfaces) on the face dataset
    n_components = 150

    print("Extracting the top %d eigenfaces from %d faces"
        % (n_components, X_train.shape[0]))
    t0 = time()
    pca = PCA(n_components=n_components, svd_solver='randomized',
            whiten=True).fit(X_train)
    print("done in %0.3fs" % (time() - t0))

    eigenfaces = pca.components_.reshape((n_components, h, w))

    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))


    # #############################################################################
    # Train a SVM classification model
    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                    param_grid, cv=5)
    clf = clf.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)


    # #############################################################################
    # Quantitative evaluation of the model quality on the test set
    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    print("done in %0.3fs" % (time() - t0))

    print(classification_report(y_test, y_pred, target_names=target_names))


    # #############################################################################
    # Qualitative evaluation of the predictions using matplotlib

    prediction_titles = [title(y_pred, y_test, i)
                        for i in range(y_pred.shape[0])]

    plot_gallery(X_test, prediction_titles, h, w)

    # plot the gallery of the most significative eigenfaces

    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w)

    plt.show()

run() 