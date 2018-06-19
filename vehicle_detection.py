import numpy as np
import cv2

from scipy.ndimage.measurements import label


class Vehicle(object):
    def __init__(self):
        # if detect in video, represent the current frame num of video
        self.frame_num = 0
        # debug mode
        self.debug_mode = False

        # to tell how to extract feature from image
        self.feature_managers = None
        self.feature_classifier = None

        # recent detect vehicle_boxes in recent n frame
        self.recent_nframe_vehicle_boxes = CirculateQueue()

        # the vehicle color
        self.color = None
        # the number of vehicle
        self.num = 0


    def detect(self, image):
        pass

    def set_feature_manager(self, feature_manager):
        self.feature_managers = feature_manager

    def set_feature_classifier(self, feature_classifier):
        self.feature_classifier = feature_classifier

    def get_feature_manager(self):
        return self.feature_managers

    def get_feature_classifier(self):
        return self.feature_classifier

    """ 
        Define some methods that use heat map to get the position of vehicle
    """
    def get_vehicle_heatmap(self, heatmap, vehicle_boxes):
        # Iterate through list of bboxes
        self.recent_nframe_vehicle_boxes.enqueue(vehicle_boxes)
        recent_boxes = self.recent_nframe_vehicle_boxes.get_all_items()
        for boxes in recent_boxes:
            for box in boxes:
                # Add += 1 for all pixels inside each bbox
                # Assuming each "box" takes the form ((x1, y1), (x2, y2))
                heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        # Return updated heatmap
        return heatmap

    def heatmap_threshold(self, heatmap, threshold=5):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    # Define a function to draw bounding boxes
    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        # Return the image
        return img


class Car(Vehicle):
    def __init__(self, feature_mgrs=None, feature_clf=None, debug_mode=False):
        Vehicle.__init__(self)
        self.feature_managers = feature_mgrs
        self.feature_classifier = feature_clf
        self.debug_mode = debug_mode

    def detect(self, image, file_type='image'):
        assert file_type == 'image' or file_type == 'video', "input file should be either image or video !"
        self.frame_num += 1
        feature_mgrs = self.get_feature_manager()
        feature_clf = self.get_feature_classifier()
        assert feature_mgrs or feature_clf, 'have not set feature manager or feature classifier '

        cars_boxes = []

        # get multiple search region and image scale based on the image size
        x_start_stop, y_start_stops, image_scales = self.get_roi(image.shape)

        for i in range(len(y_start_stops)):
            cars_boxes.extend(
                self.__detect_boxes(image, feature_clf, feature_mgrs, x_start_stop, y_start_stops[i], image_scales[i]))

        # define a heat map
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        # Add heat to each box in each vehicle box
        heatmap = self.get_vehicle_heatmap(heat, cars_boxes)

        # apply threshold to help remove false positives
        if file_type == 'image':
            thresholded_heatmap = self.heatmap_threshold(heatmap, 4)
        else:
            thresholded_heatmap = self.heatmap_threshold(heatmap, 13)
        # visualize the heatmap when displaying
        vehicle_heatmap = np.clip(thresholded_heatmap, 0, 255)
        labels = label(vehicle_heatmap)

        draw_image = self.draw_labeled_bboxes(np.copy(image), labels)

        if self.debug_mode == True:
            result = self.__get_debug_info(draw_image, heatmap)
        else:
            result = draw_image
        return result

    def __detect_boxes(self, img, clf, feature_mgrs, x_start_stop=[None, None], y_start_stop=[None, None], scale=1.0):
        """
            Define a single function that can extract features using hog sub-sampling and make predictions
            The code below maybe a little confusing, but it can help to only extract hog features once.
            As result, this version of code cost less time finding cars compared with detect_boxes1
        """

        car_boxes = []

        if x_start_stop == [None, None]:
            x_start_stop = [0, img.shape[1]]
        if y_start_stop == [None, None]:
            y_start_stop = [0, img.shape[0]]

        img_tosearch = img[y_start_stop[0]:y_start_stop[1], x_start_stop[0]:x_start_stop[1], :]

        if scale != 1:
            imshape = img_tosearch.shape
            img_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        # Define window and steps as below
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        # 8 was the step
        step = 16
        nxsteps = (img_tosearch.shape[1] - window) // step + 1
        nysteps = (img_tosearch.shape[0] - window) // step + 1

        features_list = []
        for feat_mgr in feature_mgrs:
            features_list.append(feat_mgr.search_sub_windows_feat(img_tosearch, window, step))

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


    def __detect_boxes1(self, img, clf, feature_mgrs, x_start_stop=[None, None], y_start_stop=[None, None], scale=1.0):
        """
            Get the possible windows from x_start_stop and y_start_stop, then judge each window is car boxes one by one

            """
        if x_start_stop == [None, None]:
            x_start_stop = [0, img.shape[1]]
        if y_start_stop == [None, None]:
            y_start_stop = [0, img.shape[0]]



        # Define window and steps as below
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = np.int(64 * scale)
        # 8 was the step
        step = np.int(16 * scale)
        overlap = (window - step) / window

        # get the slide windows
        windows = self.__slide_window(img, x_start_stop, y_start_stop, xy_window=(window, window),
                               xy_overlap=(overlap, overlap))
        # select the car box from slide windows
        car_boxes = self.__search_windows(img, windows, clf, feature_mgrs)
        return car_boxes

    def get_roi(self, image_shape):
        image_height = image_shape[0]
        image_width = image_shape[1]
        if int(image_height) == 720 and int(image_width) == 1280:
            x_start_stop = [None, None]
            y_start_stops = [[400, 464], [416, 480], [400, 496], [432, 528], [400, 528], [432, 560], [400, 596],
                             [464, 660]]

            image_scales = [1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 3.0, 3.0]
        else:
            raise Exception("bad image shape %d x %d" % (image_width, image_height))
        return x_start_stop, y_start_stops, image_scales


    # Define a function that takes an image,
    # start and stop positions in both x and y,
    # window size (x and y dimensions),
    # and overlap fraction (for both x and y)
    def __slide_window(self, img, x_start_stop=[None, None], y_start_stop=[None, None],
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

    # Define a function you will pass an image
    # and the list of windows to be searched (output of slide_windows())
    def __search_windows(self, img, windows, clf, feature_mgrs):
        # 1) Create an empty list to receive positive detection windows
        on_windows = []
        # 2) Iterate over all windows in the list
        for window in windows:
            # 3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            # 4) Extract scaled features for that window using extract_scaled_feature()
            features_list = []
            for feat_mgr in feature_mgrs:
                features_list.append(feat_mgr.extract_scaled_feature(test_img))

            features_list = np.hstack(features_list).reshape(1, -1)
            # 5) Predict using your classifier
            prediction = clf.predict(features_list)
            # 6) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        # 8) Return windows for positive detections
        return on_windows

    def __get_debug_info(self, draw_image, heatmap):
        """
               assemble many debug screen into one and show debug information

                Return
               ----------
               debug screen:  a assembled debug screen for showing debug information
               """
        # assemble the screen
        debug_screen = np.zeros((720, 1920, 3), dtype=np.uint8)
        # a screen that show car drawed image
        debug_screen[0:720, 0:1280] = draw_image
        # a screen that show car detect heatmap
        heatmap = heatmap * 10
        heatmap = np.dstack((heatmap, heatmap, heatmap))
        debug_screen[0: 360, 1280:1920] = cv2.resize(heatmap, (640, 360), interpolation=cv2.INTER_AREA)

        # a screen that show various debug information
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (128, 128, 0)
        debug_info = np.zeros((360, 640, 3), dtype=np.uint8)

        cv2.putText(debug_info, 'frame num: %d ' % self.frame_num, (30, 60), font, 1, color, 2)
        cv2.putText(debug_info, 'Used features: ', (30, 120), font, 1, color, 2)
        # feature debug info
        screen_height = 150
        for feat_mgr in self.feature_managers:
            feat_info = feat_mgr.get_debug_info()
            cv2.putText(debug_info, feat_info, (30, screen_height), font, 1, color, 2)
            screen_height += 30

        # model debug info
        screen_height += 30
        model_info = 'clf: svm-10-rbf'
        cv2.putText(debug_info, model_info, (30, screen_height), font, 1, color, 2)

        debug_screen[360:720, 1280:1920] = cv2.resize(debug_info, (640, 360), interpolation=cv2.INTER_AREA)
        return debug_screen


class CirculateQueue(object):
    """
    a customize circular queue based on the python list
    this queue is used for storing recent lane fit information
    """
    def __init__(self, maxsize=5):
        # define the max length of queue
        self.maxsize = maxsize
        # the queue
        self.queue = []

    def __str__(self):
        # return the str of the object
        return str(self.queue)

    def size(self):
        # the number of items in the queue
        return len(self.queue)

    def is_empty(self):
        # return True if nothing in self.queue
        return self.queue == []

    def is_full(self):
        # return True if queue if full
        return self.size() == self.maxsize

    def enqueue(self, item):
        # enqueue a item
        if self.is_full():
            self.dequeue()
        self.queue.insert(0, item)

    def dequeue(self):
        # dequeue a item
        if self.is_empty():
            return None
        return self.queue.pop()

    def find(self, value):
        # if find value(content) in the queue, return the index
        # if not, return None
        for i in range(len(self.queue)):
            if self.queue[-1 - i] == value:
                return i
        return None

    def visit(self, index):
        # return the value(content) of the index of queue
        assert 0 <= index < len(self.queue)
        return self.queue[-1 - index]

    def get_tail(self):
        # get the last item which is enqueued
        if self.is_empty():
            return None
        return self.queue[0]

    def get_all_items(self):
        # get all items from the queue
        return self.queue



