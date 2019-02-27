# tests
import os
import numpy as np
from cv2 import imread, imwrite
import scipy.ndimage as nd
import unittest
import pickle

current_dir = os.getcwd()
moduledir = os.path.join(current_dir, os.pardir)
testdir = os.path.join(moduledir, os.pardir)
imagedir = os.path.join(testdir, 'images')
exportdir = os.path.join(testdir, 'exports')
projectdir = os.path.join(testdir, os.pardir)
maindir = os.path.join(projectdir, 'main')
os.chdir(maindir)

from main.modules.segmenter.pointcarver import SeamMarker


os.chdir(current_dir)

class TestPointCarver(unittest.TestCase):
    def test_pointcarver_calc_energy(self):
        "tests the calc_energy function of pointcarver"

        vietImagePath = os.path.join(imagedir, 'vietHard.jpg')
        compImagePath = os.path.join(imagedir, 'vietEmap.png')
        viet = imread(vietImagePath)
        compImage = imread(compImagePath, 0)

        vietcp = viet.copy()

        carver = SeamMarker(img=vietcp)

        emap = carver.calc_energy(vietcp)
        emap = np.interp(emap, (emap.min(), emap.max()), (0, 256))
        emap = np.uint8(emap)
        comparray = emap == compImage
        result = comparray.all()
        self.asserTrue(result, "Point carver energy calculation function")

    def test_pointcarver_minimum_seam_emap_matrix(self):
        "tests the minimum seam function of pointcarver"
        vietImagePath = os.path.join(imagedir, 'vietHard.jpg')
        matrixPath = os.path.join(exportdir, 'vietSliceMatrix.npy')
        compmatrix = np.load(matrixPath)

        viet = imread(vietImagePath)
        vietcp = viet.copy()
        vietslice = vietcp[:, 550:600]
        carver = SeamMarker(img=vietcp)
        emap = carver.calc_energy(vietslice)
        mat, backtrack = carver.minimum_seam(img=vietslice, emap=emap)
        compmat = mat == compmatrix
        result = compmat.all()
        self.assertTrue(
            result,
            "Point carver minimum seam function emap given, checking matrix"
        )

    def test_pointcarver_minimum_seam_emap_backtrack(self):
        vietImagePath = os.path.join(imagedir, 'vietHard.jpg')

        backtrackPath = os.path.join(exportdir, 'vietSliceBacktrack.npy')
        compBacktrack = np.load(backtrackPath)

        viet = imread(vietImagePath)
        vietcp = viet.copy()
        vietslice = vietcp[:, 550:600]
        carver = SeamMarker(img=vietcp)
        emap = carver.calc_energy(vietslice)
        mat, backtrack = carver.minimum_seam(img=vietslice, emap=emap)
        compback = backtrack == compBacktrack
        result = compback.all()
        self.assertTrue(
            result,
            "Point carver minimum seam function emap given, checking backtrack"
        )

    def test_pointcarver_minimum_seam_backtrack(self):
        vietImagePath = os.path.join(imagedir, 'vietHard.jpg')

        backtrackPath = os.path.join(exportdir, 'vietSliceBacktrack.npy')
        compBacktrack = np.load(backtrackPath)

        viet = imread(vietImagePath)
        vietcp = viet.copy()
        vietslice = vietcp[:, 550:600]
        carver = SeamMarker(img=vietcp)
        mat, backtrack = carver.minimum_seam(img=vietslice)
        compback = backtrack == compBacktrack
        result = compback.all()
        self.assertTrue(
            result,
            "Point carver minimum seam function emap not given, checking backtrack"
        )

    def test_pointcarver_minimum_seam_matrix(self):
        vietImagePath = os.path.join(imagedir, 'vietHard.jpg')

        matrixPath = os.path.join(exportdir, 'vietSliceMatrix.npy')
        compmatrix = np.load(matrixPath)

        viet = imread(vietImagePath)
        vietcp = viet.copy()
        vietslice = vietcp[:, 550:600]
        carver = SeamMarker(img=vietcp)
        mat, backtrack = carver.minimum_seam(img=vietslice)
        compmat = backtrack == compmatrix
        result = compmat.all()
        self.assertTrue(
            result,
            "Point carver minimum seam function emap not given, checking matrix"
        )
