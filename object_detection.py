import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split


import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog


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


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  block_norm='L2-Hys',
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       block_norm='L2-Hys',
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features



# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):

    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


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

    color_spaces = ['RGB', 'HLS', 'LUV', 'HSV', 'YCrCb']

    spatial_features = dict()
    spatial_features['use'] = [False, True]
    #spatial_features['size'] = [(16, 16), (32, 32), (64, 64)]
    spatial_features['size'] = [(16, 16), (32, 32)]

    hist_features = dict()
    hist_features['use'] = (False, True)
    #hist_features['bin_num'] = (16, 32, 64)
    hist_features['bin_num'] = [16, 32]
    hog_features = dict()
    hog_features['use'] = (False, True)
    hog_features['orient'] = [8, 9]
    hog_features['pix_per_cell'] = [8]
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
    hog_orient_set = hog_features['orient'] = (8, 9)
    hog_pix_per_cell_set = hog_features['pix_per_cell']
    hog_cell_per_block_set = hog_features['cell_per_block']
    hog_channel_set = hog_features['hog_channel']

    # for recording each combination
    color_space_series = []
    spatial_feat_series = []
    spatial_size_series = []
    hist_feat_series = []
    hist_bins_series = []
    hog_feat_series = []
    orient_series = []
    pix_per_cell_series = []
    cell_per_block_series = []
    hog_channel_series = []
    clf_series = []
    train_time_series = []
    accuracy_series = []

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
                                                car_features = extract_features(cars, color_space=color_space,
                                                                                spatial_size=spatial_size,
                                                                                hist_bins=hist_bins,
                                                                                orient=orient,
                                                                                pix_per_cell=pix_per_cell,
                                                                                cell_per_block=cell_per_block,
                                                                                hog_channel=hog_channel,
                                                                                spatial_feat=spatial_feat,
                                                                                hist_feat=hist_feat, hog_feat=hog_feat)
                                                notcar_features = extract_features(notcars, color_space=color_space,
                                                                                   spatial_size=spatial_size,
                                                                                   hist_bins=hist_bins,
                                                                                   orient=orient,
                                                                                   pix_per_cell=pix_per_cell,
                                                                                   cell_per_block=cell_per_block,
                                                                                   hog_channel=hog_channel,
                                                                                   spatial_feat=spatial_feat,
                                                                                   hist_feat=hist_feat,
                                                                                   hog_feat=hog_feat)
                                                # Create an array stack of feature vectors
                                                X = np.vstack((car_features, notcar_features)).astype(np.float64)

                                                # Define the labels vector
                                                y = np.hstack(
                                                    (np.ones(len(car_features)), np.zeros(len(notcar_features))))

                                                # Split up data into randomized training and test sets
                                                rand_state = np.random.randint(0, 100)
                                                X_train, X_test, y_train, y_test = train_test_split(
                                                    X, y, test_size=0.2, random_state=rand_state)

                                                # Fit a per-column scaler
                                                X_scaler = StandardScaler().fit(X_train)
                                                # Apply the scaler to X
                                                X_train = X_scaler.transform(X_train)
                                                X_test = X_scaler.transform(X_test)

                                                # use different classifier to train
                                                for clf_name in classifiers:
                                                    t = time.time()
                                                    clf = classifiers[clf_name]
                                                    clf.fit(X_train, y_train)
                                                    t2 = time.time()
                                                    # training time
                                                    train_time = round(t2 - t, 3)
                                                    # test accuaracy
                                                    accuracy = round(clf.score(X_test, y_test), 4)
                                                    print("accuracy = %3.4f, training time = %3.3f " % (accuracy, train_time))

                                                    # record each combination
                                                    color_space_series.append(color_space)
                                                    spatial_feat_series.append(spatial_feat)
                                                    spatial_size_series.append(spatial_size)
                                                    hist_feat_series.append(hist_feat)
                                                    hist_bins_series.append(hist_bins)
                                                    hog_feat_series.append(hog_feat)
                                                    orient_series.append(orient)
                                                    pix_per_cell_series.append(pix_per_cell)
                                                    cell_per_block_series.append(cell_per_block)
                                                    hog_channel_series.append(hog_channel)
                                                    clf_series.append(clf_name)
                                                    train_time_series.append(train_time)
                                                    accuracy_series.append(accuracy)
    # create a data frame and save it into csv file
    df = pd.DataFrame({'color_space': color_space_series,
                        'spatial_feat': spatial_feat_series,
                        'spatial_size': spatial_size_series,
                        'hist_feat': hist_feat_series,
                        'hist_bins': hist_bins_series,
                        'hog_feat': hog_feat_series,
                        'orient': orient_series,
                        'pix_per_cell': pix_per_cell_series,
                        'cell_per_block': cell_per_block_series,
                        'hog_channel': hog_channel_series,
                        'classifier': clf_series,
                        'train_time': train_time_series,
                        'accuracy': accuracy_series})
    print(df)
    df.to_csv('feature_selection.csv')



if __name__ == "__main__":

    # Read in cars and notcars
    images = glob.glob('mix_data/*.png')
    cars = []
    notcars = []
    for image in images:
        if 'left' in image or 'right' in image or 'mid' in image or 'far' in image:
            cars.append(image)
        elif 'image' in image or 'extra' in image:
            notcars.append(image)
        else:
            cars.append(image)

    #decide the sample size
    # sample_size = 500
    # cars = cars[0:sample_size]
    # notcars = notcars[0:sample_size]

    test = True
    if test is True:
        classifiers, color_spaces, spatial_features, hist_features, hog_features = set_features_classifiers_sets()
        search_best_combination(cars, notcars, classifiers, color_spaces, spatial_features, hist_features, hog_features)


    """
    ### TODO: Tweak these parameters and see how the results change.
    color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = 0  # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)  # Spatial binning dimensions
    hist_bins = 16  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off
    y_start_stop = [None, None]  # Min and max in y to search in slide_window()

    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()

    image = mpimg.imread('bbox-example-image.jpg')
    draw_image = np.copy(image)

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    # image = image.astype(np.float32)/255

    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                           xy_window=(96, 96), xy_overlap=(0.5, 0.5))

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    plt.imshow(window_img)
    plt.show()
    """



