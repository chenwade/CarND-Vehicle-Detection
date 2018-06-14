import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import glob
import os
import pickle
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from scipy.ndimage.measurements import label

# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split


import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog


from feature import *

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


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

        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                ctrans_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                ctrans_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                ctrans_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                ctrans_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                ctrans_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            ctrans_image = np.copy(image)

        # extract image feature by using different methods
        for index, feature_mgr in enumerate(feature_managers):
            features_list[index].append(feature_mgr.extract_feature(ctrans_image))

    # Return list of feature vectors
    return features_list


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


def set_features_classifiers_sets():

    #color_spaces = ['RGB', 'HSV', 'HLS', 'LUV', 'YCrCb']
    color_spaces = ['RGB', 'HSV', 'LUV']

    spatial_features = dict()
    spatial_features['use'] = [False, True]
    #spatial_features['size'] = [(16, 16), (32, 32), (64, 64)]
    spatial_features['size'] = [(16, 16)]

    hist_features = dict()
    hist_features['use'] = (False, True)
    #hist_features['bin_num'] = (16, 32, 64)
    hist_features['bin_num'] = [16]
    hog_features = dict()
    hog_features['use'] = (False, True)
    hog_features['orient'] = [9]
    #hog_features['orient'] = [8, 9]
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

                                                print('Using:', orient, 'orientations', pix_per_cell,
                                                      'pixels per cell and', cell_per_block, 'cells per block')
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

                                                    print(train_time, 'Seconds to train Classifier...')
                                                    # Check the score of the Classifier
                                                    print('Test Accuracy of Classifier = ', accuracy)

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



def set_feat_mgrs_scalers(feature_mgrs, feature_scalers):
    for i in range(len(feature_mgrs)):
        feature_mgrs[i].features_scaler = feature_scalers[i]


def find_cars1(img, clf, feature_mgrs, x_start_stop=[None, None], y_start_stop=[None, None], scale=1.0, cspace='RGB'):
    """
        Define a single function that can extract features using hog sub-sampling and make predictions
        The code below maybe a little confusing, but it can help to only extract hog features once.
        As result, this version of code cost less time finding cars
    """

    assert feature_mgrs
    car_boxes = []

    if x_start_stop == [None, None]:
        x_start_stop = [0, image.shape[1]]
    if y_start_stop == [None, None]:
        y_start_stop = [0, image.shape[0]]

    img_tosearch = img[y_start_stop[0]:y_start_stop[1], x_start_stop[0]:x_start_stop[1], :]

    if cspace != 'RGB':
        if cspace == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
        else:
            raise Exception("color space is %s" % cspace)
    else:
        ctrans_tosearch = np.copy(img_tosearch)


    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))


    # Define window and steps as below
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    # 8 was the step
    step = 16
    nxsteps = (ctrans_tosearch.shape[1] - window) // step + 1
    nysteps = (ctrans_tosearch.shape[0] - window) // step + 1

    features_list = []
    for feat_mgr in feature_mgrs:
        features_list.append(feat_mgr.search_sub_windows_feat(ctrans_tosearch, window, step))

    features_list = np.hstack(features_list)

    for index, features in enumerate(features_list):
         # make a prediction
        test_prediction = clf.predict(features.reshape(1, -1))
        if test_prediction == 1:
            ytop = (index // nxsteps) * step
            xleft = (index % nxsteps) * step
            xbox_left = np.int(xleft * scale)
            ytop_draw = np.int(ytop * scale)
            win_draw = np.int(window * scale)
            car_boxes.append(
                [(xbox_left, ytop_draw + y_start_stop[0]),
                 (xbox_left + win_draw, ytop_draw + win_draw + y_start_stop[0])])

    return car_boxes


def find_cars_image(rgb_image, clf, featuer_mgrs, color_space='RGB'):

    cars_boxes = []

    # define multiple search field
    x_start_stop = [None, None]
    y_start_stops = [[400, 464], [416, 480], [400, 496], [432, 528], [400, 528], [432, 560], [400, 596], [464, 660]]

    image_scales = [1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 3.0, 3.0]

    for i in range(len(y_start_stops)):
        cars_boxes.extend(find_cars1(rgb_image, clf, featuer_mgrs, x_start_stop, y_start_stops[i], image_scales[i], color_space))

    # define a heat map
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    # Add heat to each box in each vehicle box
    vehicle_heatmap = add_heat(heat, cars_boxes)
    # apply threshold to help remove false positives
    vehicle_heatmap = apply_threshold(vehicle_heatmap, 2)
    # visualize the heatmap when displaying
    vehicle_heatmap = np.clip(vehicle_heatmap, 0, 255)
    labels = label(vehicle_heatmap)

    draw_image = draw_labeled_bboxes(np.copy(rgb_image), labels)
    return draw_image



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

if __name__ == "__main__":


    # Read in cars and notcars
    cars = glob.glob('vehicles/**/*.png')
    notcars = glob.glob('non-vehicles/**/*.png')
    print("vehicle samples: %d, non-vehicle samples: %d" % (len(cars), len(notcars)))

    # search for best combination
    search = True
    if search is True:
        classifiers, color_spaces, spatial_features, hist_features, hog_features = set_features_classifiers_sets()
        search_best_combination(cars, notcars, classifiers, color_spaces, spatial_features, hist_features, hog_features)

    # image features parameters setting
    color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 7  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = 0  # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)  # Spatial binning dimensions
    hist_bins = 16  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = False  # Histogram features on or off
    hog_feat = True  # HOG features on or off

    # init needed image feature manager
    feature_mgrs = []
    if spatial_feat is True:
        sp_mgr = SpatialFeature(spatial_size)
        feature_mgrs.append(sp_mgr)
    if hist_feat is True:
        color_hist_mgr = ColorHistFeature(hist_bins)
        feature_mgrs.append(color_hist_mgr)
    if hog_feat is True:
        hog_mgr = HogFeature(orient, pix_per_cell, cell_per_block, hog_channel)
        feature_mgrs.append(hog_mgr)

    """
           Extract the images features from image dataset and construct the feature dataset based on the feature 
           parameters setting. 
           Once we finished extracting the feature dataset, we save it to the 'features_set' folder 
           so that we don't need do it next time
           """
    # define feature_dataset path
    folder_name = 'features_set/'
    suffix = str(color_space) + '_spatial-' + str(spatial_feat) + str(spatial_size) + '_hist-' + str(
        hist_feat) + str(hist_bins) + '_hog-' + str(hog_feat) + str(orient) + str(pix_per_cell) + \
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
        car_features_list = extract_features(cars, feature_mgrs, color_space)
        notcar_features_list = extract_features(notcars, feature_mgrs, color_space)
        t2 = time.time()
        extract_time = round(t2 - t1, 2)

        feature_length = 0
        for i in range(len(feature_mgrs)):
            feature_length += len(car_features_list[i][0])

        # store the dataset into the .p
        save_dict = {'car_features_list': car_features_list, 'notcar_features_list': notcar_features_list,
                     'feature_length': feature_length, 'time': extract_time}
        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print("feature length: %d, feature extract time: %5.2f" % (feature_length, extract_time))

    train_data, test_data, train_labels, test_labels, feature_scalers = construct_train_test_dataset(car_features_list,
                                                                                                     notcar_features_list)
    set_feat_mgrs_scalers(feature_mgrs, feature_scalers)
    """
    Define a model and train it.
    Once we finished training the model, we save it to the 'models' folder 
        so that we don't need do it next time
    """
    # save the modes
    clf_name = 'svm-1-linear'
    folder_name = 'models/'
    model_path = folder_name + clf_name + suffix
    if os.path.isfile(model_path):
        save_dict = pickle.load(open(model_path, "rb"))
        clf = save_dict['classifier']
        train_time = save_dict['train_time']
        accuracy = save_dict['accuracy']
        # if the dataset doesn't exist, create the dataset and store it to a pickle
    else:
        # Use a classifier
        clf = LinearSVC()
        # Check the training time for the SVC
        t = time.time()
        clf.fit(train_data, train_labels)
        t2 = time.time()
        train_time = round(t2 - t, 2)
        accuracy = round(clf.score(test_data, test_labels), 4)

        # save the model info
        save_dict = {'classifier': clf, 'accuracy': accuracy, 'train_time': train_time}
        with open(model_path, 'wb') as f:
            pickle.dump(save_dict, f)

    print(train_time, 'Seconds to train Classifier...')
    # Check the score of the Classifier
    print('Test Accuracy of Classifier = ', accuracy)

    n_predict = 10
    print('My classfier predicts: ', clf.predict(test_data[0:n_predict]))
    print('For these', n_predict, 'labels: ', test_labels[0:n_predict])


    image = cv2.imread('test_images/test1.jpg')
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    draw_image = find_cars_image(rgb_image, clf, feature_mgrs, color_space)

    plt.imshow(draw_image)
    plt.show()
    a = 1




