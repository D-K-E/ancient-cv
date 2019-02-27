# author: Kaan Eraslan
# Licence: see, LICENSE
# No warranties, use it on your own risk

import json  # json io
import glob  # finding stuff in paths
import os  # path manipulation
from io import BytesIO
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from utils import assertCond, FilesLoader
import pdb


class TFDetectionDrawer:
    "Draw detections over the image"

    def __init__(self, detection: dict):
        assertCond(detection, isinstance(detection, dict))
        self.detection = detection
        impath = detection['image_path']
        image = Image.open(impath)
        self.imarr = np.array(image, dtype=np.uint8)
        self.image = image.copy()
        self.drawer = ImageDraw.Draw(self.image)
        self.bboxes = detection['detection_boxes']
        if 'detection_masks' in detection:
            self.masks = detection['detection_masks']
        #
        self.scores = detection['detection_scores']
        self.classes = detection['detection_classes']
        self.class_names = detection['class_names']
        self.nb_detection = detection['num_detections']
        self.color = ['white', 'green']

    def drawBbox(self, bbox, color: str):
        "Draw Bbox on the image"
        # left, bottom, right, top = [int(coord) for coord in bbox]
        bottom, left, top, right = [int(coord) for coord in bbox]
        self.drawer.line([(left, top),
                          (left, bottom),
                          (right, bottom),
                          (right, top),
                          (left, top)],
                         width=2, fill=color)

    def addCategoryScore2Bbox(self, category: str,
                              bbox: [float, float, float, float]):
        "Add category name to bbox"
        bottom, left, top, right = [int(coord) for coord in bbox]
        # left, bottom, right, top = [int(coord) for coord in bbox]
        font = ImageFont.load_default()
        cat_width, cat_height = font.getsize(category)
        total_height = (1 + 2 * 0.05) + cat_height
        margin = cat_height * 0.05
        if top > total_height:
            text_bottom = top
        else:
            text_bottom = bottom + total_height
        #
        self.drawer.text(
            (left + margin, text_bottom - cat_height - margin),
            category,
            fill='black',
            font=font)

    def drawMask(self, mask, color: str):
        "Draw mask on the image"
        pass

    def drawBoxCatScoreMask(self, detindex: int, hasMask=False):
        "Draw detection without mask"
        box = self.bboxes[detindex]
        score = self.scores[detindex]
        cat = self.class_names[detindex]
        cat_id = self.classes[detindex]
        cat_score = "score_" + str(score) + "_cat_id_" + str(cat_id)
        cat_score = cat_score + "_cat_" + cat
        color = self.color[detindex % 2]

        self.drawBbox(box, color)
        self.addCategoryScore2Bbox(cat_score, box)
        if hasMask is True:
            mask = self.masks[detindex]
            self.drawMask(mask, color)

    def drawDetection(self):
        "Draw detection on image"
        hasmask = bool('detection_masks' in self.detection)
        for i in range(self.nb_detection):
            self.drawBoxCatScoreMask(detindex=i, hasMask=hasmask)


def getLabelPath(modeldir: str) -> str:
    "Get available label namespaces under the models dir"
    assertCond(modeldir, isinstance(modeldir, str))
    label_glob = os.path.join(modeldir, '*')
    label_ns = glob.glob(label_glob)
    print('Please choose your label namespace among ones that are available')
    print('Here is a list of available label namespaces: ')
    label_ns_dict = {}
    print('')
    for lbl in label_ns:
        bname = os.path.basename(lbl)
        label_ns_dict[bname] = lbl
        print('Label namespace: ', bname)

    strinput = input('Enter your choice: ')
    return label_ns_dict[strinput]


def getConfPath(label_path: str) -> str:
    "get config path from user"
    assertCond(label_path, isinstance(label_path, str))
    config_glob = os.path.join(label_path, '**')
    configs = glob.glob(config_glob, recursive=True)
    confs = {
        os.path.basename(f): f for f in configs
    }
    confs = {
        name: path for name, path in confs.items() if ('.config' in name and
                                                       'conf-' in name)
    }
    print('Please choose the configuration file you have used in inference.')
    print('Here are available configuration files: \n')
    for name, conf in confs.items():
        confdir = os.path.dirname(conf)
        modeldir = os.path.dirname(confdir)
        labeldir = os.path.dirname(modeldir)
        print('label namespace: ', os.path.basename(labeldir))
        print('model name: ', os.path.basename(modeldir))
        print('configuration name: ', name)

    print('')
    confname = input('Enter your choice: ')
    confpath = confs[confname]
    return confpath


def getInputOutputPath(conf_path: str,
                       session_backend='tf'):
    "Get input output path"
    assertCond(conf_path, isinstance(conf_path, str))
    confdir = os.path.dirname(conf_path)
    sessiondir = os.path.join(confdir,
                              session_backend+'session')
    input_dir = os.path.join(sessiondir, 'detection_output')
    output_dir = os.path.join(sessiondir, 'visu')
    return input_dir, output_dir


def getDetectionOutputs(input_dir: str):
    "From input_dir get detection outputs resulting from infer.py"
    assertCond(input_dir, isinstance(input_dir, str))
    loader = FilesLoader(path=input_dir,
                         ext='json')
    outs = loader.loadFiles()
    detections = []
    for out in outs:
        detstr = out['file']
        name = out['dataName']
        detection = json.loads(detstr)
        detections.append((name, detection))
    return detections


def handleDetection(detection: dict, out_path: str):
    "Handle the detection drawing event"
    assertCond(detection, isinstance(detection, dict))
    assertCond(out_path, isinstance(out_path, str))
    # pdb.set_trace()
    drawer = TFDetectionDrawer(detection)
    drawer.drawDetection()
    drawer.image.save(out_path)


def printIntro():
    print('Welcome to visualize.py of ancient-cv.')
    print('This script will help you to visualize your inference results.')
    print('It supposes that you have already run infer.py on certain images.')
    print('Do not worry this script does not modify your original images.')
    print(
        'The program supposes that you are running it from ancient-cv/main/ .'
    )
    strinput = input('Would you like to continue [y/n, y by default]? ')
    if strinput == 'n':
        print('You have chosen to quit.')
        print('Bye bye ..')
        sys.exit(0)

    return


def main():
    "Main program"
    printIntro()
    curdir = os.getcwd()
    # models
    modelsdir = os.path.join(curdir, 'models')
    label_dir = getLabelPath(modelsdir)
    conf_path = getConfPath(label_dir)
    input_path, output_path = getInputOutputPath(conf_path)
    detections = getDetectionOutputs(input_path)
    for name, detection in detections:
        ext = 'png'
        outname = name + '.' + ext
        outpath = os.path.join(output_path, outname)
        handleDetection(detection, outpath)
    #
    print('Written your new images to: {0}'.format(output_path))
    print('Quitting script.')
    print('Bye Bye ..')


if __name__ == '__main__':
    main()
