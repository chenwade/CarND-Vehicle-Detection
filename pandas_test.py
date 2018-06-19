import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import glob
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from scipy.ndimage.measurements import label



def search_best_combination1(classifiers, color_spaces, spatial_features, hist_features, hog_features):

    # resolve the features

    use_spaital = spatial_features['use']
    spatial_size_set = spatial_features['size']

    use_hist_set = hist_features['use']
    hist_bins_set = hist_features['bin_num']

    hog_use_set = hog_features['use']
    hog_orient_set = hog_features['orient'] = (8, 9)
    hog_pix_per_cell_set = hog_features['pix_per_cell']
    hog_cell_per_block_set = hog_features['cell_per_block']
    hog_channel_set = hog_features['hog_channel']


    df = pd.DataFrame(columns=['color_space',
                        'spatial_feat',
                        'spatial_size',
                        'hist_feat',
                        'hist_bins',
                        'hog_feat',
                        'orient',
                        'pix_per_cell',
                        'cell_per_block',
                        'hog_channel',
                        'clf_name',
                        'train_time',
                        'accuracy'])


    for color_space in color_spaces:
        for spatial_feat in use_spaital:
            for spatial_size in spatial_size_set:
                for hist_feat in use_hist_set:
                    for hist_bins in hist_bins_set:
                        for hog_feat in hog_use_set:
                            for orient in hog_orient_set:
                                for pix_per_cell in hog_pix_per_cell_set:
                                    for cell_per_block in hog_cell_per_block_set:
                                        for hog_channel in hog_channel_set:
                                            if spatial_feat or hist_feat or hog_feat:
                                                for clf_name in classifiers.keys():
                                                    # collect data into csv

                                                    accuracy = 0.985
                                                    train_time = 1.05
                                                    print(color_space, "spitial:", spatial_feat, spatial_size,
                                                          "hist:", hist_feat, hist_bins, "hog:", hog_feat, orient, pix_per_cell, cell_per_block, hog_channel, clf_name,
                                                          "accuracy = %3.4f, training time = %3.3f " % (
                                                    accuracy, train_time))

                                                    new_series = pd.Series(
                                                        [color_space, spatial_feat, spatial_size, hist_feat, hist_bins,
                                                         hog_feat, orient, pix_per_cell, cell_per_block, hog_channel,
                                                         clf_name, train_time, accuracy], index=['color_space',
                                                                                                 'spatial_feat',
                                                                                                 'spatial_size',
                                                                                                 'hist_feat',
                                                                                                 'hist_bins',
                                                                                                 'hog_feat',
                                                                                                 'orient',
                                                                                                 'pix_per_cell',
                                                                                                 'cell_per_block',
                                                                                                 'hog_channel',
                                                                                                 'clf_name',
                                                                                                 'train_time',
                                                                                                 'accuracy'])
                                                    print(new_series)
                                                    df = df.append(new_series, ignore_index=True)
                                                    print(df)




                                                    """
                                                    df.append({'color_space': color_space,
                                                                'spatial_feat': spatial_feat,
                                                                'spatial_size': spatial_size,
                                                                'hist_feat': hist_feat,
                                                                'hist_bins': hist_bins,
                                                               'hog_feat': hog_feat,
                                                               'orient': orient,
                                                               'pix_per_cell': pix_per_cell,
                                                              'cell_per_block': cell_per_block,
                                                               'hog_channel': hog_channel,
                                                                'classifier': clf_name,
                                                                 'training_time': train_time,
                                                                 'accuracy': accuracy})

                                                    df.append([color_space, spatial_feat, spatial_size, hist_feat, hist_bins, hog_feat, orient, pix_per_cell,
                                                               cell_per_block, hog_channel, clf_name, train_time, accuracy], ignore_index=True)
                                                               
                                                    """
                                        df.to_csv('test.csv')



def set_features_classifiers_sets1():

    color_spaces = ('RGB', 'HLS', 'LUV', 'HSV', 'YCrCb')

    spatial_features = dict()
    spatial_features['use'] = (False, True)
    spatial_features['size'] = ((16, 16), (32, 32), (64, 64))

    hist_features = dict()
    hist_features['use'] = (False, True)
    hist_features['bin_num'] = (16, 32, 64)

    hog_features = dict()
    hog_features['use'] = (False, True)
    hog_features['orient'] = (8, 9)
    hog_features['pix_per_cell'] = (8, 16)
    hog_features['cell_per_block'] = (2, 4)
    hog_features['hog_channel'] = (0, 1, 2, 'ALL')

    classifiers = dict()
    nb = GaussianNB()
    dt = DecisionTreeClassifier()
    SVC_1_linear = SVC(C=1.0, kernel='linear')
    SVC_10_linear = SVC(C=10.0, kernel='linear')
    SVC_1_rbf = SVC(C=1.0, kernel='rbf')
    SVC_10_rbf = SVC(C=10.0, kernel='rbf')
    classifiers['nb'] = nb
    classifiers['dt'] = dt
    classifiers['SVC_1_linear'] = SVC_1_linear
    classifiers['SVC_10_linear'] = SVC_10_linear
    classifiers['SVC_1_rbf'] = SVC_1_rbf
    classifiers['SVC_10_rbf'] = SVC_10_rbf

    return classifiers, color_spaces, spatial_features, hist_features, hog_features


def foo(s):
    return 10 / int(s)

def bar(s):
    return foo(s) * 2

def exception_test():
    for i in range(5, -5, -1):
        try:
            print(bar(i))
        except Exception as e:
            doc = open('error.txt', mode='a')
            print('Error:', e, file=doc)
            doc.close()
            continue



if __name__ == "__main__":


    X = np.array([[1 ,2, 3, np.nan, 5, 6, np.nan], [1 ,2, 3, 4, 5, 6, 7], [1, 2, 3, np.nan, 5, 6, np.nan], [1, 2, 3, 8, 5, 6, 11]])
    y = np.array([[0], [1], [1], [0]])
    # filter bad value in X
    nans_index = np.where(np.isnan(X))
    row_index = list(set(nans_index[0]))
    X = np.delete(X, row_index, axis=0)



    exception_test()


    classifiers, color_spaces, spatial_features, hist_features, hog_features = set_features_classifiers_sets1()
    search_best_combination1(classifiers, color_spaces, spatial_features, hist_features, hog_features)