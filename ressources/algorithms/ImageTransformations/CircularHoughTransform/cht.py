import numpy as np
from scipy import ndimage

img = ndimage.imread("testimage.jpg")


def getGrayScale(img: np.ndarray([], dtype=np.uint8)):
    "Get grayscale image"
    # r: 0.2126
    # g: 0.7152
    # b: 0.0722
    img[:, :, 0] = img[:, :, 0] * 0.2126
    img[:, :, 1] = img[:, :, 1] * 0.7152
    img[:, :, 2] = img[:, :, 2] * 0.0722
    #
    grayim = img.sum(axis=2, dtype=np.uint8)
    #
    return grayim


def getEdges(grayim: np.ndarray([], dtype=np.uint8),
             sigmas=[5,2]):
    "Compute edges of the grayscale image"
    smoothed = ndimage.gaussian_filter(grayim, sigma=sigmas[0])
    glimage = ndimage.gaussian_laplace(smoothed, sigma=sigmas[2])
    #
    return glimage


def getSinCosTheta(img: np.ndarray([], dtype=np.uint8)):
    "Get sincos values of angles in 360 degree"
    rows, cols = img.shape[:2]
    sin = {}
    cos = {}
    thetas = np.linspace(0, 360, num=max(rows, cols), dtype=np.float32)
    #
    for angle in thetas:
        ang = angle * np.pi / 180
        sin[angle] = np.sin(ang)
        cos[angle] = np.cos(ang)
    #
    return sin, cos, thetas


def voteAccumulator(img: np.ndarray([], dtype=np.uint8),
                    edgecoords: np.ndarray([], dtype=np.uint32),
                    radius_minmax: (int, int),
                    sincos: ({},{})):
    #
    sins, coss, thetas = getSinCosTheta(img)
    rmin = radius_minmax[0]
    rmax = radius_minmax[1]
    #
    rownb = img.shape[0]
    colnb = img.shape[1]
    #
    accu = np.zeros((rownb,
                     colnb,
                     rmax), dtype=np.int32)
    #
    for x, y in edgecoords:
        for radius in range(rmin, rmax):
            for angle in range(360):
                a = x - round(radius * sins[angle], 2)
                b = y - round(radius * coss[angle], 2)
                if ((a >= 0 and a < rownb) and
                    (b >= 0 and b < colnb)
                ):
                    accu[a, b, radius] += 1
    #
    return accu

def filterAccumulator(accu: np.ndarray([], dtype=np.int32),
                      thresh: int):
    "Filter accumulator for given threshold of occurance of the center coord"
    accurownb = accu.shape[0]
    accucolnb = accu.shape[1]
    accuradnb = accu.shape[2]
    newaccu = accu[accu[:,]]
