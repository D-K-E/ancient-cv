# author: Kaan Eraslan
# Licence: see, LICENSE
# No warranties, use it on your own risk
# purpose: load data from data directory

from modules.detector.tf_detector.research.object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem
from modules.detector.tf_detector.research.object_detection.dataset_tools import tf_record_creation_util as trcu

import os  # path manipulation
import sys  # exit system
import json  # json io
import numpy as np  # working with image matrix
from PIL import Image  # imageio and conversion
from PIL import ImageDraw  # image mask io and creation
import io  # generally required for io
import base64  # reading image data of annotations
import glob  # finding stuff in paths
import tensorflow as tf  # for generating tf records
import contextlib2  # several tf records instead of a single huge one
import pdb
import logging

from data.utils import convertStringBytesFeature
from data.utils import convertBytesStrListFeature, convertIntBoolListFeature
from data.utils import convertIntBoolFeature, convertFloatListFeature
from utils import assertCond, augmentImageData_proc

from utils import FilesLoader, convertPilImg2Array


currentdir = os.getcwd()

# data related folders
datadir = os.path.join(currentdir, 'data')


# model related folders
modeldir = os.path.join(currentdir, 'models')

# asset related folders
moduledir = os.path.join(currentdir, 'modules')
preprocessdir = os.path.join(moduledir, 'preprocessor')
preprocessImagedir = os.path.join(preprocessdir, 'images')
assetDir = os.path.join(preprocessImagedir, 'assets')
annotationsDir = os.path.join(assetDir, 'annotations')
labelsDir = os.path.join(assetDir, 'labels')
imagesDir = os.path.join(assetDir, 'images')


class LabelLoader(FilesLoader):
    "Load label files"

    def __init__(self, path: str, ext='txt'):
        super().__init__(path, ext)
        self.labels = self.files

    def getLabels(self):
        "Get labels from path"
        labelFiles = self.loadFiles()
        return labelFiles


class AnnotationsLoader(FilesLoader):
    "Loads the annotations from given path"

    def __init__(self, path: str, ext='json'):
        super().__init__(path, ext)
        self.data = []

    def getAnnotations(self):
        "Get annotations from annotations dir"
        annotationFiles = self.loadFiles()
        annotations = []
        for afile in annotationFiles:
            annotation = json.loads(afile['file'], encoding='utf-8')
            annotation['labelName'] = afile['dirName']
            annotation['ext'] = afile['ext']
            annotation['name'] = afile['name']
            annotation['dataName'] = afile['dataName']
            annotations.append(annotation)
        #
        return annotations

    def getBboxFromMask(self,
                        mask: np.ndarray([], dtype=np.uint8)
                        ) -> [int, int,  # xmin, ymin
                              int, int]:  # xmax, ymax
        """Get bbox values from mask array"""
        assert mask.ndim == 2
        assert mask.dtype == 'uint8'
        nonzeroArray = np.argwhere(mask)
        # [[x1,y1],[x2, y2], ...]
        xmin, ymin = nonzeroArray.min(axis=0)
        xmax, ymax = nonzeroArray.max(axis=0)
        return xmin, ymin, xmax, ymax

    def getMaskFromShape(self, shape: dict,
                         imgHeight: int,
                         imgWidth: int) -> np.ndarray:
        """
        Get mask from shape data

        Description
        -------------
        Takes most of its functionality from the shape_to_mask function of labelme
        project
        """
        mask = np.zeros((imgHeight, imgWidth), dtype=np.uint8)
        mask = Image.fromarray(mask)
        draw = ImageDraw.Draw(mask)
        xy = list(map(tuple, shape['points']))
        shape_type = shape['shape_type']
        if shape_type == 'circle':
            assert len(xy) == 2, 'Shape of shape_type=circle must have 2 points'
            (cx, cy), (px, py) = xy
            d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
        elif shape_type == 'rectangle':
            assert len(
                xy) == 2, 'Shape of shape_type=rectangle must have 2 points'
            draw.rectangle(xy, outline=1, fill=1)
        elif shape_type == 'line':
            raise ValueError('line shaped annotations are not supported.'
                             ' Please revise the annotation')
        elif shape_type == 'linestrip':
            draw.line(xy=xy, fill=1, width=line_width)
        elif shape_type == 'point':
            assert len(xy) == 1, 'Shape of shape_type=point must have 1 points'
            cx, cy = xy[0]
            r = point_size
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
        else:
            assert len(xy) > 2, 'Polygon must have points more than 2'
            draw.polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=np.uint8)
        if mask.ndim > 2:
            mask = np.transpose(mask, (1, 0, 2))
        elif mask.ndim == 2:
            mask = mask.T
        #
        return mask

    def getMaskLabelBboxFromShape(self, shape: dict,
                                  imgHeight: int,
                                  imgWidth: int) -> dict:
        "Get mask label and bbox from shape"
        mask = self.getMaskFromShape(shape, imgHeight, imgWidth)
        label = shape['label']
        bbox = self.getBboxFromMask(mask)
        nbbox = [bbox[0]/imgWidth, bbox[1]/imgHeight,
                 bbox[2]/imgWidth, bbox[3]/imgHeight]
        maskio = io.BytesIO()
        maskim = Image.fromarray(mask)
        maskim.save(maskio, format='PNG')
        return {"mask": maskio.getvalue(),
                "label": label,
                "nbbox": nbbox,
                "bbox": bbox}

    def getShapeInfoFromAnnotation(self, annotation: dict) -> [dict]:
        "Get masks, labels and bboxes from annotation file"
        imageHeight = annotation['imageHeight']
        imageWidth = annotation['imageWidth']
        shapes = []

        for shape in annotation['shapes']:
            info = self.getMaskLabelBboxFromShape(
                shape, imageHeight, imageWidth)
            shapes.append(info)
        #
        return shapes

    def getImageFromAnnotation(self, annotation: dict) -> np.ndarray:
        "Get image from annotation dict"
        fio = io.BytesIO()
        decodedImage = base64.b64decode(annotation['imageData'])
        fio.write(decodedImage)
        image = Image.open(fio)
        imarray = convertPilImg2Array(image)
        return imarray

    def getDataFromAnnotations(self, annotations: [dict]) -> [dict]:
        "Get data from annotations in order to transform it into tfrecord"
        annotated_data = []
        for annotation in annotations:
            impath = annotation['imagePath']
            imname = os.path.basename(impath)
            shapeInfo = self.getShapeInfoFromAnnotation(annotation)
            image = self.getImageFromAnnotation(annotation)
            annotated_data.append({'info': shapeInfo,
                                   'image': image,
                                   'imageName': imname + "_normal",
                                   'labelNamespace': annotation['labelName']
                                   })
            augmentImageData_proc(
                image=np.copy(image),
                holder=annotated_data,
                shapeInfo=shapeInfo,
                imname=imname,
                labelNameSpace=annotation['labelName'])
        self.data = annotated_data
        return annotated_data

    def loadAnnotations(self):
        "Wrapper function for the class"
        annotations = self.getAnnotations()
        return self.getDataFromAnnotations(annotations)


class AnnotationsHandler:
    """
    Handle annotations

    Here is the example folder structure for the object_detection api

    + data
      + labelName
        - labelMapFile
        + train
          - train tfrecords
        + eval
          - eval tfrecords

    + models
      + labelName
        + model
          - piplineConfigFile
          + train
          + eval
    """

    def __init__(self,
                 nb_parts=5,
                 dataPath=datadir,
                 annotationsPath=annotationsDir,
                 modelDir=modeldir,
                 annotationLoader=AnnotationsLoader(annotationsDir),
                 labelLoader=LabelLoader(labelsDir)):
        self.annotationLoader = annotationLoader
        self.labelLoader = labelLoader
        self.nb_parts = nb_parts
        self.dataDir = dataPath
        self.modelDir = modelDir
        self.annotationsPath = annotationsPath
        self.annotations = []
        self.labels = []
        self.label_maps = {}
        self.label_examples = {}

    def getAnnotations(self):
        "get annotations from annotations loader"
        for label in self.labels:
            dname = label['dataName']
            apath = os.path.join(self.annotationLoader.path, dname)
            loader = AnnotationsLoader(apath)
            annotations = loader.loadAnnotations()
            [self.annotations.append(an) for an in annotations]
        return self.annotations

    def getLabels(self):
        "get labels from label loader"
        self.labels = self.labelLoader.getLabels()
        for label in self.labels:
            self.label_examples[label['dataName']] = []
        return self.labels

    def filterLabelsByLabelNamespace(self, name: str):
        "filter out label namespaces except for the given name"
        labels = [l for l in self.labels if l['dataName'] == name]
        self.labels = labels

    def makeLabelNamespace(self):
        "Make a new folder for each label in data folder"
        for label in self.labels:
            dataName = label['dataName']
            labelPath = os.path.join(self.dataDir, dataName)
            if os.path.isdir(labelPath) is False:
                os.mkdir(labelPath)
                trainpath = os.path.join(labelPath, 'train')
                evalpath = os.path.join(labelPath, 'eval')
                os.mkdir(trainpath)
                os.mkdir(evalpath)
            modelLabelPath = os.path.join(self.modelDir, dataName)
            if os.path.isdir(modelLabelPath) is False:
                os.mkdir(modelLabelPath)
        #
        return

    def makeLabelMapFromLabels(self):
        "Create label maps from labels and save it into proper folder"
        for label in self.labels:
            labelPath = os.path.join(self.dataDir, label['dataName'])
            labelFile = label['file']
            labels = labelFile.splitlines()
            labels = [l.strip() for l in labels]
            self.label_maps[label['dataName']] = []
            items = []
            for i, data in enumerate(labels):
                someItem = StringIntLabelMapItem(name=data, id=i+1)
                items.append(someItem)
                self.label_maps[label['dataName']].append({'name': data,
                                                           'id': i+1})
            #
            labelMap = StringIntLabelMap(item=items)
            labelMapPath = os.path.join(labelPath,
                                        label['dataName'] + ".pbtxt")
            with open(labelMapPath, "w",
                      encoding='utf-8', newline='\n') as lbl:
                lbl.write(str(labelMap))
        #
        return

    def findLabelInLabelMap(self, labelName: str, label: str) -> dict:
        "get the label related data in label map using label namespace"
        for item in self.label_maps[labelName]:
            if item['name'] == label:
                return item

    def convertShape2TFExample(self,
                               shapeInfo: dict,
                               labelName: str):
        "convert shape info dict to tf.Example"
        # Find the label item
        # pdb.set_trace()
        item = self.findLabelInLabelMap(labelName,
                                        shapeInfo['label'])
        classes_text = str.encode(item['name'])
        classes = item['id']

        # normalized bbox
        x1, y1, x2, y2 = (shapeInfo['nbbox'][0], shapeInfo['nbbox'][1],
                          shapeInfo['nbbox'][2], shapeInfo['nbbox'][3])

        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
        ymax = max(y1, y2)

        # get mask
        mask = shapeInfo['mask']

        info = {
            'xmin': xmin,
            'xmax': xmax,
            'ymax': ymax,
            'ymin': ymin,
            'classes': classes,
            'classes_text': classes_text,
            'mask': mask
        }
        return info

    def getTFExamplesFromAnnotation(self, annotation):
        "Convert a single annotation to tf.Example"
        imarr = annotation['image']
        imdata = io.BytesIO()
        img = Image.fromarray(imarr)
        imw = img.width
        imh = img.height

        # get imdata and format
        img.save(imdata, format='PNG')
        imdata = imdata.getvalue()

        imformat = b'png'
        examples = []

        imname = str.encode(annotation['imageName'])
        xmins = []
        ymins = []
        ymaxs = []
        xmaxs = []
        masks = []
        classes = []
        classes_texts = []
        #
        for shapeInfo in annotation['info']:
            info = self.convertShape2TFExample(
                shapeInfo=shapeInfo,
                labelName=annotation['labelNamespace']
            )
            xmins.append(info['xmin'])
            ymins.append(info['ymin'])
            xmaxs.append(info['xmax'])
            ymaxs.append(info['ymax'])
            masks.append(info['mask'])
            classes.append(info['classes'])
            classes_texts.append(info['classes_text'])
        #
        featureDict = {
            'image/height': convertIntBoolFeature(imh),
            'image/width': convertIntBoolFeature(imw),
            'image/filename': convertStringBytesFeature(imname),
            'image/source_id': convertStringBytesFeature(imname),
            'image/encoded': convertStringBytesFeature(imdata),
            'image/format': convertStringBytesFeature(imformat),
            'image/object/bbox/xmin': convertFloatListFeature(xmins),
            'image/object/bbox/xmax': convertFloatListFeature(xmaxs),
            'image/object/bbox/ymin': convertFloatListFeature(ymins),
            'image/object/bbox/ymax': convertFloatListFeature(ymaxs),
            'image/object/class/text': convertBytesStrListFeature(classes_texts),
            'image/object/class/label': convertIntBoolListFeature(classes),
            'image/object/mask': convertBytesStrListFeature(masks)

        }
        example = tf.train.Example(features=tf.train.Features(
            feature=featureDict))
        self.label_examples[annotation['labelNamespace']].append(example)
        #
        return example

    def writeTFRecord2DataDir(self, examples: list,
                              labelNamePath: str,
                              isTrain=True):
        "Write a tfrecord file for examples"
        if isTrain is True:
            path = os.path.join(labelNamePath, 'train')
            filename = os.path.join(path, 'train.record')
        else:
            path = os.path.join(labelNamePath, 'eval')
            filename = os.path.join(path, 'eval.record')
        #
        with tf.python_io.TFRecordWriter(filename) as writer:
            #
            for example in examples:
                writer.write(example.SerializeToString())

            #
            writer.close()

    def writeTFRecords2DataDir(self, examples: list,
                               labelNamePath: str,
                               isTrain=True
                               ):
        "Write tf record file in shards"
        if isTrain is True:
            path = os.path.join(labelNamePath, 'train')
            filename = os.path.join(path, 'train.record')
        else:
            path = os.path.join(labelNamePath, 'eval')
            filename = os.path.join(path, 'eval.record')
        #
        with contextlib2.ExitStack() as tf_record_close_stack:
            output_tf_records = trcu.open_sharded_output_tfrecords(
                tf_record_close_stack, base, self.nb_parts
            )
            for index, data in enumerate(examples):
                output_index = index % self.nb_parts
                output_tf_records[output_index].write(data.SerializeToString())

    def generateTFRecordFromAnnotations(self,
                                        labelNameSpace: str,
                                        isSingle=True,
                                        evalRatio=0.3):
        "Wrapper function to generate single tf records from annotations"
        assert isinstance(evalRatio, float)
        assert evalRatio < 1 and evalRatio > 0
        # eval ratio is how much of the data should be used for
        # evaluation
        # pdb.set_trace()
        self.getLabels()
        self.filterLabelsByLabelNamespace(name=labelNameSpace)
        self.getAnnotations()

        self.makeLabelNamespace()
        self.makeLabelMapFromLabels()
        [self.getTFExamplesFromAnnotation(an) for an in self.annotations]
        for labelNameSpace, examples in self.label_examples.items():
            labelNamePath = os.path.join(self.dataDir,
                                         labelNameSpace)
            evalnb = int(len(examples) * evalRatio)
            trainnb = len(examples) - evalnb
            train_examples = examples[:trainnb]
            eval_examples = examples[trainnb:]
            np.random.shuffle(train_examples)
            np.random.shuffle(eval_examples)
            if isSingle:
                self.writeTFRecord2DataDir(examples=train_examples,
                                           labelNamePath=labelNamePath,
                                           isTrain=True)
                self.writeTFRecord2DataDir(examples=eval_examples,
                                           labelNamePath=labelNamePath,
                                           isTrain=False)

            else:
                self.writeTFRecords2DataDir(examples=train_examples,
                                            labelNamePath=labelNamePath,
                                            isTrain=True)
                self.writeTFRecords2DataDir(examples=eval_examples,
                                            labelNamePath=labelNamePath,
                                            isTrain=False)


def printIntro():
    print('Hello fellow ancient object lover !')
    print('Nice to see you here')
    print('Welcome to data loading script of the ancient-cv repository')
    print('Please read the following instructions carefully.')
    print('Please verify the following before proceeding further')
    print('You can leave anytime by pressing ctrl+c in terminal')
    print('This script ensures that everything is in their proper locations')
    print("""

It also creates new folders and files to ensure sanity of the entire pipeline

You can use the default options, if :

    - You are launching this script from "{0}" directory

    - You have used labels that are in "{1}"

    - Your labels are in txt format, each separated with a 
      newline inside the file
      (You can use your own labels if you want, see sampleLabelFile.txt)
      (Just put it under the labels folder in "{2}")

    - Your annotations are in "{3}" under their corresponding label namespace
      (A label namespace is basically a folder name based on the name of
      the label file you have used while annotating your images.
      Your annotations folder should contain a subfolder with this name.
      The subfolder should contain all the annotations.
      You don't need to worry about label namespace if you have used 
      annotate.py
      (If you need an example please look at the sampleAnnotation.json in {4})
      (If you have not yet prepared your annotations, please use the
      annotate.py in {5}. Come back once you have finished.)

    - Your annotations are in json format

    - You want the program to use models in {6}
          """.format(currentdir, labelsDir, labelsDir,
                     annotationsDir,
                     annotationsDir,
                     preprocessImagedir,
                     modeldir)
          )


if __name__ == "__main__":
    printIntro()
    print('\n\n')
    strinput = input('Do all of the above apply to you [y/n, y by default]? ')
    if strinput == 'n':
        print('exiting the script...')
        sys.exit()
    #
    print('Please pick your label namespace you have used while annotating')
    print('Here are available label namespaces: ')
    annoregex = os.path.join(annotationsDir, '*')
    labelnamespaces = glob.glob(annoregex)
    for name in labelnamespaces:
        bname = os.path.basename(name)
        print('\nlabel name space: ', bname, '\n')

    nameSpace = input(
        'Please enter the name of the label name space of your choice: '
    )
    handler = AnnotationsHandler()
    strinput = input(
        """
Would like to generate multiple tfrecords (you should
if you have more than few thousand examples)
[y/n, n by default] ?
            """)
    isSingle = bool(strinput != 'y')
    strinput = input("""how much of the data should be used for evaluation
        [0.3 by default] [enter a value between 0-1]? """)
    if strinput == '':
        evalRatio = 0.3
    else:
        evalRatio = float(strinput)
    handler.generateTFRecordFromAnnotations(isSingle=isSingle,
                                            labelNameSpace=nameSpace,
                                            evalRatio=evalRatio)
    #
    print('loading data is done, please proceed to configuration with config.py')
