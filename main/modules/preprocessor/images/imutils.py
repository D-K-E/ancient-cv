# author: Kaan Eraslan
# Licence: see, LICENSE
# No warranties, use it on your own risk

import cv2
import numpy as np
import pdb


def getBlockSize(val: int, threshlist: [(int, int)]):
    "Get adjusted block size for adaptive binarization"
    threshlist.sort(key=lambda x: x[1])
    retval = None
    for arange in threshlist:
        if val >= arange[0] and val < arange[1]:
            retval = int(val * arange[0] / arange[1])
    return retval


def assertCond(var, cond: bool, printType=True):
    "Assert condition print message"
    if printType:
        assert cond, 'variable value: {0}\nits type: {1}'.format(var,
                                                                 type(var))
    else:
        assert cond, 'variable value: {0}'.format(var)


class ImageBinarizer:
    "Holds binarization operations on images"

    def __init__(self, image: np.uint8):
        assertCond(image, isinstance(image, np.ndarray))
        assertCond(image, image.dtype == 'uint8', printType=False)
        self.image = np.copy(image)
        if self.image.ndim > 2:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # used by kmeans binarizer
        self.cluster_nb = 2
        self.iteration_nb = 10
        self.stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                              self.iteration_nb,  # number of max iterations
                              # allowed
                              1  # minimum measure of movement signaling the
                              # convergence
                              # if the movement is below this level we consider
                              # the algorithm
                              # as converged
                              )
        self.clip_limit = 2.0
        self.frame_size = (3, 3)

        # Gaussian Blur Options if needed
        self.kernel_size = (5, 5)
        self.stddev = 0

        # adaptive thresholding options
        rownb = self.image.shape[0]
        colnb = self.image.shape[1]
        if rownb < 5 or colnb < 5:
            raise ValueError(
                'Image shape not suitable for binarization methods of this'
                ' class row shape: {}, col shape: {}'.format(rownb,
                                                             colnb))
        threshlist = [(0, 10), (10, 100),
                      (10, 200), (20, 300),
                      (50, 500), (150, 1000), (250, 1000), (50, 1000),
                      (100, 10000), (200, 10000)]

        self.v_block_size = getBlockSize(rownb, threshlist)
        self.h_block_size = getBlockSize(colnb, threshlist)
        if self.v_block_size % 2 == 0:
            self.v_block_size = self.v_block_size + 1
        else:
            self.v_block_size = self.v_block_size
        if self.h_block_size % 2 == 0:
            self.h_block_size = self.h_block_size + 1
        else:
            self.h_block_size = self.h_block_size

    def binarizeAdaptiveMean(self):
        "Apply adaptive thresholding to image"
        imcp = np.copy(self.image)
        # pdb.set_trace()
        binimg = cv2.adaptiveThreshold(imcp,
                                       255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY,
                                       self.v_block_size,
                                       self.h_block_size)
        return binimg

    def binarizeAdaptiveGaussian(self):
        "Apply adaptive thresholding to image"
        imcp = np.copy(self.image)
        binimg = cv2.adaptiveThreshold(imcp,
                                       255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY,
                                       self.v_block_size,
                                       self.h_block_size)
        return binimg

    def binarizeOtsu(self):
        imcp = np.copy(self.image)
        ret2, binimg = cv2.threshold(imcp,
                                     0, 255,
                                     cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return binimg

    def binarizeOtsuGaussian(self):
        "Apply gaussian filtering before applying otsu"
        imcp = np.copy(self.image)
        blured = cv2.GaussianBlur(imcp,
                                  self.kernel_size,
                                  self.stddev)
        ret2, binimg = cv2.threshold(blured,
                                     0, 255,
                                     cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return binimg

    def _k_bin(self, img) -> np.ndarray([], dtype="uint8"):
        """
        Purpose
        -------
        Binarize a grayscale image using k-means from opencv

        Parameters
        -----------
        See attributes

        Returns
        --------
        bin_img: np.ndarray([], dtype="uint8")
            Binarized image
        """
        imcp = np.copy(img)
        refim = np.copy(img)
        imcp = np.float32(imcp)
        imcp = imcp.reshape((-1))
        distances, labels, centers = cv2.kmeans(
            data=imcp,  # original image
            K=self.cluster_nb,  # number of clusters
            bestLabels=None,
            criteria=self.stop_criteria,
            attempts=self.iteration_nb,
            flags=cv2.KMEANS_RANDOM_CENTERS
        )
        # Convert centers to images
        # centers gives the assigned values for centers
        # distances gives the distance between each pixel and its assigned
        # center
        # labels show which pixel is assigned to which center
        #
        labels_flattened = labels.flatten()
        segmented_data = np.take(centers, indices=labels_flattened)
        # since we have two centers we have a binary
        segmented_data = np.uint8(segmented_data)
        min_data = segmented_data.min()
        max_data = segmented_data.max()
        segmented_data[segmented_data == min_data] = 0
        segmented_data[segmented_data == max_data] = 255
        # structure
        bin_img = segmented_data.reshape(refim.shape)
        #
        return bin_img

    def binarize_k_means(self):
        return self._k_bin(self.image)

    def binarize_kbin_clahe(self):
        imcp = np.copy(self.image)
        bin_img = self._kbin_clahe(imcp)
        #
        return bin_img

    def _kbin_clahe(self, img):
        imcp = np.copy(img)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                tileGridSize=self.frame_size)
        image_cv = clahe.apply(imcp)
        bin_img = self._k_bin(image_cv)
        #
        return bin_img

    def binarize_kbin_hist(self):
        "Apply kmeans binarization to image with histogram equilization"
        imcp = np.copy(self.image)
        image_cv = cv2.equalizeHist(imcp)
        bin_img = self._k_bin(image_cv)
        #
        return bin_img

    def binarize_kbin_clahe_batches(self, batch_nb: int):
        "Apply kmeans with clahe with column batches"
        imcp = np.copy(self.image)
        if batch_nb > 10:
            raise ValueError("Please provide batch_nb smaller than 10")

        if imcp.shape[1] < batch_nb:
            raise ValueError("""
            Image has less columns than batch_nb.
            Are you sure the image is not transposed ?
            Here is its shape: {}
            """.format(imcp.shape))

        image_col = imcp.shape[1]
        bin_image = np.copy(imcp)

        batch_range = image_col // batch_nb

        for batch in range(batch_nb):
            b_start = batch * batch_range
            b_end = b_start + batch_range
            img_section = imcp[:, b_start:b_end]
            bin_img = self._kbin_clahe(img=img_section)
            bin_image[:, b_start:b_end] = bin_img

        return bin_image


class EdgeDetector:
    "Regroups edge detection methods"

    def __init__(self, image: np.uint8):
        assertCond(image, isinstance(image, np.ndarray))
        assertCond(image, image.dtype == 'uint8', printType=False)
        self.image = np.copy(image)
        if self.image.ndim > 2:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        self.kernel_size = 5
        rownb = self.image.shape[0]
        colnb = self.image.shape[1]
        if rownb < 5 or colnb < 5:
            raise ValueError(
                'Image shape not suitable for binarization methods of this'
                ' class row shape: {}, col shape: {}'.format(rownb,
                                                             colnb))

    def get_laplacian_edges(self):
        "wrapper on opencv function"
        imcp = np.copy(self.image)
        laplacian = cv2.Laplacian(imcp, cv2.CV_64F)
        return self.getAbsImage(laplacian)

    def getAbsImage(self, img: np.float):
        absim = np.absolute(img)
        return np.uint8(absim)

    def get_hsobel_edges(self):
        "wrapper on opencv functions"
        imcp = np.copy(self.image)
        sobelx = cv2.Sobel(imcp, cv2.CV_64F, 1, 0, self.kernel_size)
        return self.getAbsImage(sobelx)

    def get_vsobel_edges(self):
        imcp = np.copy(self.image)
        sobely = cv2.Sobel(imcp, cv2.CV_64F, 0, 1, self.kernel_size)
        return self.getAbsImage(sobely)

    def get_canny_edges(self, l_hyst=100, h_hyst=210):
        "Wrapper on opencv canny edge detector"
        if l_hyst > h_hyst:
            raise ValueError("""
            Lower value for hysteria thresholding is greater than higher value:
            lower: {},
            higher: {}
            """.format(str(l_hyst), str(h_hyst)))

        imcp = np.copy(self.image)
        canny_image = cv2.Canny(imcp, l_hyst, h_hyst)
        return self.getAbsImage(canny_image)


class ImagePreprocessor:
    "Handler for image preprocessing"

    def __init__(self, image: np.uint8):
        assertCond(image, isinstance(image, np.ndarray))
        assertCond(image, image.dtype == 'uint8', printType=False)
        self.image = np.copy(image)
        self.binarizer = ImageBinarizer(self.image)
        self.edge_detector = EdgeDetector(self.image)

    def transform_pixel_energy(self) -> list:
        "Wrapper for methods of binarizer and edge_detector"
        kbin_image = self.binarizer.binarize_k_means()
        canny_edge_image = self.edge_detector.get_canny_edges()
        gaussianOtsu_image = self.binarizer.binarizeOtsuGaussian()
        adaptive_bin_image = self.binarizer.binarizeAdaptiveMean()
        gaussian_bin_image = self.binarizer.binarizeAdaptiveGaussian()
        kbin_clahe_image = self.binarizer.binarize_kbin_clahe()
        kbin_hist_image = self.binarizer.binarize_kbin_hist()
        laplacian_edges_image = self.edge_detector.get_laplacian_edges()
        vsobel_image = self.edge_detector.get_vsobel_edges()
        hsobel_image = self.edge_detector.get_hsobel_edges()
        return {
            "kbin_image": kbin_image,
            "kbin_clahe_image": kbin_clahe_image,
            "kbin_hist_image": kbin_hist_image,
            "adaptive_bin_image": adaptive_bin_image,
            "gaussian_bin_image": gaussian_bin_image,
            "gaussianOtsu_image": gaussianOtsu_image,
            "laplacian_edges_image": laplacian_edges_image,
            "canny_edge_image": canny_edge_image,
            "vsobel_image": vsobel_image,
            "hsobel_image": hsobel_image
        }
