import sys
import re
import argparse
import pickle
import glob

import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

from object_detection import *


def train_model():
    # Read in cars and notcars
    cars = glob.glob('vehicles/**/*.png')
    notcars = glob.glob('non-vehicles/**/*.png')
    print("vehicle samples: %d, non-vehicle samples: %d" % (len(cars), len(notcars)))

    # search for best combination
    search = False
    if search is True:
        classifiers, color_spaces, spatial_features, hist_features, hog_features = set_features_classifiers_sets()
        search_best_combination(cars, notcars, classifiers, color_spaces, spatial_features, hist_features, hog_features)

    # image features parameters setting
    color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 16  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)  # Spatial binning dimensions
    hist_bins = 16  # Number of histogram bins
    spatial_feat = False  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = False  # HOG features on or off

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
    # set the feature manager scaler
    for i in range(len(feature_mgrs)):
        feature_mgrs[i].set_scalers(feature_scalers[i])

    """
    Define a model and train it.
    Once we finished training the model, we save it to the 'models' folder 
        so that we don't need do it next time
    """
    # save the modes
    clf_name = 'svm-10-rbf'
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
        clf = SVC(C=10.0, kernel='rbf')
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
    print('For these', n_predict, 'labels:  ', test_labels[0:n_predict])

    return feature_mgrs, clf


def process_image(image):
    global feature_mgrs
    global clf
    global color_space
    draw_image = find_cars(image, clf, feature_mgrs, color_space='RGB')
    return draw_image





if __name__ == "__main__":

    # set defalut parameter
    parser = argparse.ArgumentParser(prog='object_detection.py', usage='python %(prog)s -i input_file -o [output_file]',
                                     description='detect lane from images or pictures')
    parser.add_argument('-i', '--input_file', type=str, default='test1.jpg',
                        help='input image or video file to process')
    parser.add_argument('-o', '--output_file', type=str, default='project_video_out.mp4', help='processed image or video file')
    args = parser.parse_args()

    """
    .可以匹配任意字符
    .+ 至少一个任意字符
    ^.+ 开始至少一个任意字符
    ^.+\.mp4$  开始至少一个任意字符且.mp4结尾
    """

    # check whether the input file is video or image
    video_pattern = re.compile("^.+\.mp4$")
    image_pattern = re.compile("^.+\.(jpg|jpeg|JPG|png|PNG)$")

    if video_pattern.match(args.input_file):
        if not os.path.exists(args.input_file):
            print("Video input file: %s does not exist.  Please check and try again." % (args.input_file))
            sys.exit(1)
        else:
            file_type = 'video'
    elif image_pattern.match(args.input_file):
        if not os.path.exists(args.input_file):
            print("Image input file: %s does not exist.  Please check and try again." % (args.input_file))
            sys.exit(2)
        else:
            file_type = 'image'
    else:
        print("Invalid video/image filename extension for output.  Must end with '.mp4', '.jpg' '.png'... ")
        sys.exit(3)



    # choose features and train classifier
    feature_mgrs, clf = train_model()

    # process image
    if file_type == 'image':
        print("image processing %s..." % (args.input_file))
        start_time = time.clock()

        image = cv2.imread(args.input_file)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # find car in the image
        draw_image = process_image(rgb_image)

        end_time = time.clock()
        print("running time %s seconds" % (end_time - start_time))

        plt.imshow(draw_image)
        plt.xlim(0, draw_image.shape[1])
        plt.ylim(draw_image.shape[0], 0)
        plt.show()

    # process video
    if file_type == 'video':
        print("video processing %s..." % (args.input_file))
        start_time = time.clock()

        clip1 = VideoFileClip(args.input_file)
        video_clip = clip1.fl_image(process_image)
        video_clip.write_videofile(args.output_file, audio=False)

        end_time = time.clock()
        print("running time %s seconds" % (end_time - start_time))
