import numpy as np
import cv2
from skimage.feature import hog


class ImageFeature(object):
    def __init__(self):
        self.features = None
        # the scaler to standardize features by removing the mean and scaling to unit variance
        self.features_scaler = None

    def extract_feature(self, image):
        pass

    def set_scalers(self, feature_scaler):
        # set the feature scaler
        self.features_scaler = feature_scaler


class SpatialFeature(ImageFeature):
    """
        features: spatial binning of color
    """
    def __init__(self, size=(32, 32)):
        ImageFeature.__init__(self)
        self.size = size

    def extract_scaled_feature(self, image, vis=False, feature_vec=True):
        feature = self.extract_feature(image).reshape(1, -1)
        scaled_feature = self.features_scaler.transform(feature)
        return scaled_feature

    # Define a function to compute binned color features
    def extract_feature(self, image):
        # Use cv2.resize().ravel() to create the feature vector
        self.features = cv2.resize(image, self.size).ravel()
        return self.features

    def search_sub_windows_feat(self, image, window=64, step=16):
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


class ColorHistFeature(ImageFeature):
    """
        features: histogram of color
    """
    def __init__(self, nbins=32, bins_range=(0, 256)):
        ImageFeature.__init__(self)
        self.nbins = nbins
        self.bins_range = bins_range

    def extract_scaled_feature(self, image, vis=False, feature_vec=True):
        feature = self.extract_feature(image).reshape(1, -1)
        scaled_feature = self.features_scaler.transform(feature)
        return scaled_feature

    # Define a function to compute binned color features
    def extract_feature(self, image):
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



class HogFeature(ImageFeature):
    """
        features: histogram of oriented gradient (care about texture, not color)
    """
    def __init__(self, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL'):
        ImageFeature.__init__(self)
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel

    def extract_scaled_feature(self, image, vis=False, feature_vec=True):
        feature = self.extract_feature(image, vis, feature_vec).reshape(1, -1)
        scaled_feature = self.features_scaler.transform(feature)
        return scaled_feature

    def extract_feature(self, image, vis=False, feature_vec=True):
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




