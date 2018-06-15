import numpy as np
import cv2

import pickle
import time
import os
import pandas as pd


from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class ImageFeature(object):
    def __init__(self):
        self.features = None
        # the scaler to standardize features by removing the mean and scaling to unit variance
        self.features_scaler = None
        self.cspace = None

    def extract_feature(self, image):
        pass

    def set_scalers(self, feature_scaler):
        # set the feature scaler
        self.features_scaler = feature_scaler

    def color_convert(self, image, cspace=None):
        if cspace is None:
            cspace = self.cspace

        assert cspace == 'RGB' or cspace == 'HSV' or cspace == 'HLS' or cspace == 'YUV' or cspace == 'LUV' or cspace == 'YCrCb', 'unknown color space'
        # apply color conversion if other than 'RGB'
        if self.cspace != 'RGB':
            if self.cspace == 'HSV':
                ctrans_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif self.cspace == 'LUV':
                ctrans_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif self.cspace == 'HLS':
                ctrans_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif self.cspace == 'YUV':
                ctrans_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif self.cspace == 'YCrCb':
                ctrans_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            else:
                raise Exception("color space is %s" % cspace)
        else:
            ctrans_image = np.copy(image)

        return ctrans_image


class SpatialFeature(ImageFeature):
    """
        features: spatial binning of color
    """
    def __init__(self, cspace='RGB', size=(32, 32)):
        ImageFeature.__init__(self)
        self.cspace = cspace
        self.size = size

    def extract_scaled_feature(self, image, vis=False, feature_vec=True):
        feature = self.extract_feature(image).reshape(1, -1)
        scaled_feature = self.features_scaler.transform(feature)
        return scaled_feature

    # Define a function to compute binned color features
    def extract_feature(self, image):
        # apply color conversion if other than 'RGB'
        image = self.color_convert(image)
        # Use cv2.resize().ravel() to create the feature vector
        self.features = cv2.resize(image, self.size).ravel()
        return self.features

    def search_sub_windows_feat(self, image, window=64, step=16):
        # apply color conversion if other than 'RGB'
        image = self.color_convert(image)
        nxsteps = (image.shape[1] - window) // step + 1
        nysteps = (image.shape[0] - window) // step + 1
        sub_features_list = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                xleft = xb * step
                ytop = yb * step
                # Extract the image patch
                subimg = cv2.resize(image[ytop:ytop + window, xleft:xleft + window], (64, 64))
                sub_features = self.extract_feature(subimg)
                sub_features_list.append(sub_features)

        scaled_sub_feature_list = self.features_scaler.transform(sub_features_list)
        return scaled_sub_feature_list

    def get_debug_info(self):
        debug_info = 'spatial: size=' + str(self.size)
        return debug_info

class ColorHistFeature(ImageFeature):
    """
        features: histogram of color
    """
    def __init__(self, cspace='RGB', nbins=32, bins_range=(0, 256)):
        ImageFeature.__init__(self)
        self.cspace = cspace
        self.nbins = nbins
        self.bins_range = bins_range

    def extract_scaled_feature(self, image, vis=False, feature_vec=True):
        feature = self.extract_feature(image).reshape(1, -1)
        scaled_feature = self.features_scaler.transform(feature)
        return scaled_feature

    # Define a function to compute binned color features
    def extract_feature(self, image):
        # apply color conversion if other than 'RGB'
        image = self.color_convert(image)

        features = []
        if len(image.shape) > 2:
            channel_num = image.shape[2]
            # Compute the histogram of the color channels separately
            for channel in range(channel_num):
                hist_feat, bin_edges = np.histogram(image[:, :, channel], bins=self.nbins, range=self.bins_range)
                features.append(hist_feat)
            # Concatenate the histograms into a single feature vector
            self.features = np.concatenate(features)
        # single channel image
        else:
            self.features = np.histogram(image, bins=self.nbins, range=self.bins_range)
        # Return the individual histograms, bin_centers and feature vector
        return self.features

    def search_sub_windows_feat(self, image, window=64, step=16):
        # apply color conversion if other than 'RGB'
        image = self.color_convert(image)
        nxsteps = (image.shape[1] - window) // step + 1
        nysteps = (image.shape[0] - window) // step + 1
        sub_features_list = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                xleft = xb * step
                ytop = yb * step
                # Extract the image patch
                subimg = cv2.resize(image[ytop:ytop + window, xleft:xleft + window], (64, 64))
                sub_features = self.extract_feature(subimg)
                sub_features_list.append(sub_features)

        scaled_sub_feature_list = self.features_scaler.transform(sub_features_list)
        return scaled_sub_feature_list

    def get_debug_info(self):
        debug_info = 'hist: nbins=' + str(self.nbins)
        return debug_info

class HogFeature(ImageFeature):
    """
        features: histogram of oriented gradient (care about texture, not color)
    """
    def __init__(self, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL'):
        ImageFeature.__init__(self)
        self.cspace = cspace
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel

    def extract_scaled_feature(self, image, vis=False, feature_vec=True):
        feature = self.extract_feature(image, vis, feature_vec).reshape(1, -1)
        scaled_feature = self.features_scaler.transform(feature)
        return scaled_feature

    def extract_feature(self, image, vis=False, feature_vec=True):

        # apply color conversion if other than 'RGB'
        image = self.color_convert(image)

        # Call get_hog_features() with vis=False, feature_vec=True
        if self.hog_channel == 'ALL':
            assert len(image.shape) > 2, "input image is single channel, we need more than 3 channel image " \
                                         "to compute the 'ALL' hog feature"
            hog_features = []
            channel_num = image.shape[2]
            for channel in range(channel_num):
                hog_features.append(self.__get_hog_features(image[:, :, channel], vis=False, feature_vec=True))
            self.features = np.ravel(hog_features)
        else:
            self.features = self.__get_hog_features(image[:, :, self.hog_channel], vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        return self.features

    # Define a function to return HOG features and visualization
    def __get_hog_features(self, img, vis=False, feature_vec=True):
        # Call with two outputs if vis==True
        if vis is True:
            features, hog_image = hog(img, orientations=self.orient,
                                      pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                      block_norm='L2-Hys',
                                      cells_per_block=(self.cell_per_block, self.cell_per_block),
                                      transform_sqrt=True,
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(img, orientations=self.orient,
                           pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                           cells_per_block=(self.cell_per_block, self.cell_per_block),
                           block_norm='L2-Hys',
                           transform_sqrt=True,
                           visualise=vis, feature_vector=feature_vec)
            return features

    def search_sub_windows_feat(self, image, window=64, step=16):
        # apply color conversion if other than 'RGB'
        image = self.color_convert(image)

        nxblocks = (image.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (image.shape[0] // self.pix_per_cell) - self.cell_per_block + 1
        #nfeat_per_block = self.orient * self.cell_per_block ** 2
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step = step // self.pix_per_cell  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

        if self.hog_channel == 'ALL':
            assert len(image.shape) > 2, "input image is single channel, we need more than 3 channel image " \
                                         "to compute the 'ALL' hog feature"
            features = []
            for channel in range(image.shape[2]):
                features.append(self.__get_hog_features(image[:, :, channel], vis=False, feature_vec=False))
        else:
            features = self.__get_hog_features(image[:, :, self.hog_channel], vis=False, feature_vec=False)

        sub_features_list = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step

                if self.hog_channel == 'ALL':
                    sub_hog_features = []
                    for channel in range(image.shape[2]):
                        sub_hog_features.append(features[channel][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel())
                    sub_hog_features = np.hstack(sub_hog_features)
                else:
                    sub_hog_features = features[ypos:ypos + nblocks_per_window,
                                   xpos:xpos + nblocks_per_window].ravel()
                sub_features_list.append(sub_hog_features)

        scaled_sub_feature_list = self.features_scaler.transform(sub_features_list)
        return scaled_sub_feature_list

    def get_debug_info(self):
        debug_info = 'hog: ori=' + str(self.orient) + ' ppc=' + str(self.pix_per_cell) + ' cpb=' + str(
            self.cell_per_block) + ' chan:' + str(self.hog_channel)
        return debug_info


def construct_train_test_dataset(car_features_list, notcar_features_list):
    """
        construct the train and test dataset from the feature set
        Noticed that we need to normalize(scale) each kind of features respectively and then combined them into together
        """
    # the of feature
    features_num = len(car_features_list)
    train_data = None
    test_data = None
    feature_scalers = []
    rand_state = np.random.randint(0, 100)
    for index in range(features_num):
        # for each kind of feature vector, we split it and scaled it
        car_features = car_features_list[index]
        notcar_features = notcar_features_list[index]
        features = car_features + notcar_features
        # Define the labels vector
        labels = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
        # Split up data into randomized training and test feature
        train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, test_size=0.2, random_state=rand_state)

        # Fit each type of features a scaler
        feat_scaler = StandardScaler().fit(train_features)
        feature_scalers.append(feat_scaler)
        # Apply the scaler to train and test feature
        scaled_train_features = feat_scaler.transform(train_features)
        scaled_test_features = feat_scaler.transform(test_features)
        # combine scaled features to final dataset
        if train_data is None:
            train_data = scaled_train_features
            test_data = scaled_test_features
        else:
            train_data = np.hstack((train_data, scaled_train_features))
            test_data = np.hstack((test_data, scaled_test_features))

        """
        # filter bad value in X
        nans_index = np.where(np.isnan(X))
        row_index = list(set(nans_index[0]))
        X = np.delete(X, row_index, axis=0)
        y = np.delete(y, row_index, axis=0)
        """

    return train_data, test_data, train_labels, test_labels, feature_scalers


# Define a function to extract features from a list of feature managers
def extract_features(imgs, feature_managers, cspace='RGB'):
    # Create a list to append feature vectors
    features_list = []
    for feature_mgr in feature_managers:
        features_list.append([])

    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # extract image feature by using different methods
        for index, feature_mgr in enumerate(feature_managers):
            features_list[index].append(feature_mgr.extract_feature(image))

    # Return list of feature vectors
    return features_list


def set_features_classifiers_sets():
    # color_spaces = ['RGB', 'HSV', 'HLS', 'LUV', 'YCrCb']
    color_spaces = ['RGB', 'HSV', 'LUV']

    spatial_features = dict()
    spatial_features['use'] = [False, True]
    # spatial_features['size'] = [(16, 16), (32, 32), (64, 64)]
    spatial_features['size'] = [(16, 16)]

    hist_features = dict()
    hist_features['use'] = (False, True)
    # hist_features['bin_num'] = (16, 32, 64)
    hist_features['bin_num'] = [16]
    hog_features = dict()
    hog_features['use'] = (False, True)
    hog_features['orient'] = [9]
    # hog_features['orient'] = [8, 9]
    hog_features['pix_per_cell'] = [8, 16]
    hog_features['cell_per_block'] = [2]
    hog_features['hog_channel'] = [0, 1, 2, 'ALL']

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


def search_best_combination(cars, notcars, classifiers, color_spaces, spatial_features, hist_features, hog_features):
    # resolve the features from input

    use_spaital = spatial_features['use']
    spatial_size_set = spatial_features['size']

    use_hist_set = hist_features['use']
    hist_bins_set = hist_features['bin_num']

    hog_use_set = hog_features['use']
    hog_orient_set = hog_features['orient']
    hog_pix_per_cell_set = hog_features['pix_per_cell']
    hog_cell_per_block_set = hog_features['cell_per_block']
    hog_channel_set = hog_features['hog_channel']

    # create a empty data frame
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
                                            # at least extract one feature
                                            if spatial_feat or hist_feat or hog_feat:

                                                # init needed image feature manager
                                                feature_mgrs = []
                                                if spatial_feat is True:
                                                    sp_mgr = SpatialFeature(spatial_size)
                                                    feature_mgrs.append(sp_mgr)
                                                if hist_feat is True:
                                                    color_hist_mgr = ColorHistFeature(hist_bins)
                                                    feature_mgrs.append(color_hist_mgr)
                                                if hog_feat is True:
                                                    hog_mgr = HogFeature(orient, pix_per_cell, cell_per_block,
                                                                         hog_channel)
                                                    feature_mgrs.append(hog_mgr)

                                                """
                                                       Extract the images features from image dataset and construct the feature dataset based on the feature 
                                                       parameters setting. 
                                                       Once we finished extracting the feature dataset, we save it to the 'features_set' folder 
                                                       so that we don't need do it next time
                                                       """
                                                # define feature_dataset path
                                                folder_name = 'features_set/'
                                                suffix = str(color_space) + '_spatial-' + str(spatial_feat) + str(
                                                    spatial_size) + '_hist-' + str(
                                                    hist_feat) + str(hist_bins) + '_hog-' + str(hog_feat) + str(
                                                    orient) + str(pix_per_cell) + \
                                                         str(cell_per_block) + str(hog_channel) + '.p'
                                                save_path = folder_name + suffix

                                                if os.path.isfile(save_path):
                                                    # if feature dataset exist, load it from pickle
                                                    save_dict = pickle.load(open(save_path, "rb"))
                                                    car_features_list = save_dict['car_features_list']
                                                    notcar_features_list = save_dict['notcar_features_list']
                                                    feature_length = save_dict['feature_length']
                                                    extract_time = save_dict['time']
                                                else:
                                                    # if the dataset doesn't exist, create the feature the dataset and store it to a pickle
                                                    t1 = time.time()
                                                    car_features_list = extract_features(cars, feature_mgrs,
                                                                                         color_space)
                                                    notcar_features_list = extract_features(notcars, feature_mgrs,
                                                                                            color_space)
                                                    t2 = time.time()
                                                    extract_time = round(t2 - t1, 2)

                                                    feature_length = 0
                                                    for i in range(len(feature_mgrs)):
                                                        feature_length += len(car_features_list[i][0])

                                                    # store the dataset into the .p
                                                    save_dict = {'car_features_list': car_features_list,
                                                                 'notcar_features_list': notcar_features_list,
                                                                 'feature_length': feature_length, 'time': extract_time}
                                                    with open(save_path, 'wb') as f:
                                                        pickle.dump(save_dict, f)

                                                print('Using:', color_space, 'spatial_feat', spatial_feat, 'hist_feat',
                                                      hist_feat, 'hog_feat', hog_feat)

                                                print('Using:', spatial_size, 'spatial_size', hist_bins, 'hist bin',
                                                      orient, 'orientations', pix_per_cell, 'pixels per cell and',
                                                      cell_per_block, 'cells per block', 'hog_channel:', hog_channel)

                                                print("feature length: %d, feature extract time: %5.2f" % (
                                                    feature_length, extract_time))

                                                train_data, test_data, train_labels, test_labels, feature_scalers = construct_train_test_dataset(
                                                    car_features_list,
                                                    notcar_features_list)

                                                """
                                                Define a model and train it.
                                                Once we finished training the model, we save it to the 'models' folder 
                                                    so that we don't need do it next time

                                                """
                                                folder_name = 'models/'
                                                for clf_name in classifiers:

                                                    model_path = folder_name + clf_name + suffix
                                                    if os.path.isfile(model_path):
                                                        save_dict = pickle.load(open(model_path, "rb"))
                                                        clf = save_dict['classifier']
                                                        train_time = save_dict['train_time']
                                                        accuracy = save_dict['accuracy']

                                                    else:
                                                        try:
                                                            # Use a classifier
                                                            clf = classifiers[clf_name]
                                                            # Check the training time for the SVC
                                                            t = time.time()
                                                            clf.fit(train_data, train_labels)
                                                            t2 = time.time()
                                                            train_time = round(t2 - t, 2)
                                                            accuracy = round(clf.score(test_data, test_labels), 4)

                                                            # save the model info
                                                            save_dict = {'classifier': clf, 'accuracy': accuracy,
                                                                         'train_time': train_time}
                                                            with open(model_path, 'wb') as f:
                                                                pickle.dump(save_dict, f)

                                                            # create a series to record current training information

                                                        except Exception as e:
                                                            # if exception happened in the training process, move to next training
                                                            doc = open('error.txt', mode='a')

                                                            print(color_space, "spitial:", spatial_feat, spatial_size,
                                                                  "hist:", hist_feat, hist_bins, "hog:", hog_feat,
                                                                  orient,
                                                                  pix_per_cell, cell_per_block, hog_channel, clf_name,
                                                                  '----Error:', e, file=doc)
                                                            doc.close()
                                                            continue

                                                    # Check the score of the Classifier
                                                    print(train_time, 'Seconds to train ', clf_name, 'Test Accuracy = ',
                                                          accuracy)

                                                    new_series = pd.Series(
                                                        [color_space, spatial_feat, spatial_size, hist_feat,
                                                         hist_bins,
                                                         hog_feat, orient, pix_per_cell, cell_per_block,
                                                         hog_channel,
                                                         clf_name, train_time, accuracy],
                                                        index=['color_space',
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

                                                    # append the series to the data frame
                                                    df = df.append(new_series, ignore_index=True)
                                                df.to_csv('feature_selection.csv')

