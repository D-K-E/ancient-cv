# author: Kaan Eraslan
# license: see, LICENSE
# No Warranties, see LICENSE

from modules.detector.tf_detector.research.object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap

from google.protobuf import json_format
from google.protobuf import text_format

import os
import sys
import numpy as np
import tensorflow as tf
import subprocess
import datetime
from uuid import uuid4
import glob
import json
import pdb
from PIL import Image
from utils import convertPilImg2Array, stripExt, assertCond
from utils import getTFCheckpointPath

curdir = os.getcwd()

# data related

datadir = os.path.join(curdir, 'data')

# modules related
modulesdir = os.path.join(curdir, 'modules')
detectdir = os.path.join(modulesdir, 'detector')
tfmodelsdir = os.path.join(detectdir, 'tf_detector')
researchdir = os.path.join(tfmodelsdir, 'research')
slimdir = os.path.join(researchdir, 'slim')
objdir = os.path.join(researchdir, 'object_detection')
export_script_path = os.path.join(objdir, 'export_inference_graph.py')
protosdir = os.path.join(objdir, 'protos')

modelsdir = os.path.join(curdir, 'models')

sys.path.append(researchdir)
sys.path.append(objdir)
sys.path.append(slimdir)

from modules.detector.tf_detector.research.object_detection.utils import ops


class TFGraphExporter:
    "Freeze tensorflow graphs"

    def __init__(self,
                 config_path: str):
        self.cpath = config_path
        modelConfdir = os.path.dirname(config_path)
        self.sessiondir = os.path.join(modelConfdir, 'tfsession')
        self.ckpt_path = ""

    def getCheckpointPath(self):
        "Get the highest model checkpoint in the model dir"
        self.ckpt_path = getTFCheckpointPath(self.sessiondir)
        return self.ckpt_path

    def export_frozen_tf_graph(self):
        "Export frozen graph using export_inference_graph.py"
        newpath = ":".join(sys.path)[1:]
        odir = os.path.join(self.sessiondir, 'frozen_graph')
        print('export script path: ', export_script_path)
        print('ckpt path: ', self.ckpt_path)
        print('config path: ', self.cpath)
        osenv = os.environ.copy()
        ospath = osenv['PATH']
        ldpath = osenv['LD_LIBRARY_PATH']
        subprocess.run(["python3",
                        export_script_path,
                        "--input_type",
                        "image_tensor",
                        "--pipeline_config_path",
                        self.cpath,
                        "--trained_checkpoint_prefix",
                        self.ckpt_path,
                        "--output_directory",
                        odir],
                       env={"PYTHONPATH": newpath,
                            "PATH": ospath,
                            "LD_LIBRARY_PATH": ldpath}
                       )

    def exportGraph(self):
        "wrapper function for methods"
        self.getCheckpointPath()
        self.export_frozen_tf_graph()
        odir = os.path.join(self.sessiondir, 'frozen_graph')
        print('graph exported to {0}'.format(odir))


class ImageClassifierTF:
    "Classify png images"

    def __init__(self,
                 graph_path: str,
                 min_score_thresh=0.6):
        self.gpath = graph_path
        self.min_score_thresh = min_score_thresh
        frozendir = os.path.dirname(graph_path)
        sessiondir = os.path.dirname(frozendir)
        confdir = os.path.dirname(sessiondir)
        modeldir = os.path.dirname(confdir)  # model name is important
        label_name = os.path.dirname(modeldir)
        label_name = os.path.basename(label_name)
        labelDataDir = os.path.join(datadir, label_name)
        label_proto_path = os.path.join(labelDataDir,
                                             label_name + '.pbtxt')
        with open(label_proto_path, 'r', encoding='utf-8') as f:
            label_map = f.read()
            label_proto = text_format.Parse(label_map,
                                            StringIntLabelMap())
            self.label_map = json_format.MessageToDict(label_proto)
            # pdb.set_trace()
            self.label_map = {
                jf['id']:jf['name'] for jf in self.label_map['item']
            }

        self.modelName = os.path.basename(modeldir)
        self.names = {"num": 'num_detections',
                      "boxes": 'detection_boxes',
                      "scores": 'detection_scores',
                      "classes": 'detection_classes',
                      "masks": 'detection_masks'}
        # for obtaining masks if they are present
        self.output_dir = os.path.join(sessiondir, 'detection_output')
        if os.path.isdir(self.output_dir) is False:
            os.mkdir(self.output_dir)
        self.graph = tf.Graph()
        with self.graph.as_default():
            mygraph_def = tf.GraphDef()
            with tf.gfile.GFile(self.gpath, "rb") as fd:
                serialize_graph = fd.read()
                mygraph_def.ParseFromString(serialize_graph)
                tf.import_graph_def(mygraph_def, name='')

    def get_tensor_map_keys(self, name_fields: [str], tnames):
        "Get tensor map keys from default graph"
        t_map = {}
        for name in name_fields:
            tname = name + ':0'
            if tname in tnames:
                t_map[name] = tf.get_default_graph().get_tensor_by_name(tname)
        #
        return t_map

    def infer(self, session, img,
              input_map, feed_key):
        "Run session with input_map and feed_key"
        imexpand = np.expand_dims(img, 0)
        output = session.run(input_map,
                             feed_dict={feed_key: imexpand})
        output['num_detections'] = int(output['num_detections'][0])
        output['detection_classes'] = output[
            'detection_classes'][0].astype('uint8')
        output['detection_boxes'] = output['detection_boxes'][0]
        output['detection_scores'] = output['detection_scores'][0]
        if 'detection_masks' in output:
            output['detection_masks'] = output['detection_masks'][0]
        #
        return output

    def fixDetectedBboxes(self, detected: dict,
                          img: np.ndarray([], dtype=np.uint8)):
        "change bbox coordinates of the detected bboxes to absolute coords"
        bboxes = detected[self.names['boxes']]
        # bbox == ymin xmin ymax xmax
        bboxes[:, 0] = bboxes[:, 0] * img.shape[0]
        bboxes[:, 2] = bboxes[:, 2] * img.shape[0]
        bboxes[:, 1] = bboxes[:, 1] * img.shape[1]
        bboxes[:, 3] = bboxes[:, 3] * img.shape[1]
        detected[self.names['boxes']] = bboxes
        return detected

    def fillNanSubarray(self,
                       detected: dict, 
                       keyname: str,
                       index: int):
        "Remove subarray using index"
        assertCond(detected, isinstance(detected, dict))
        assertCond(keyname, isinstance(keyname, str))
        assertCond(index, isinstance(index, int))
        if self.names[keyname] in detected:
            detected[self.names[keyname]][index, :].fill(np.nan)
        return detected

    def removeNans(self, detected: dict):
        "Remove nans"
        assertCond(detected, isinstance(detected, dict))
        # pdb.set_trace()

        bboxes = detected[self.names['boxes']]
        classes = detected[self.names['classes']]
        scores = detected[self.names['scores']]

        bboxes = bboxes[~np.isnan(bboxes)]
        bboxes = bboxes.reshape((-1, 4))
        scores = scores[~np.isnan(scores)]
        classes = classes[~np.isnan(classes)]

        detected[self.names['boxes']] = bboxes
        detected[self.names['scores']] = scores
        detected[self.names['classes']] = classes.astype('int32')
        if self.names['masks'] in detected:
            mshape = detected[self.names['masks']].shape
            mshape = mshape[1:]  # leave out the nb of masks
            masks = detected[self.names['masks']]
            masks = masks[~np.isnan(masks)]
            # reshape the mask array
            masks = masks.reshape(mshape)
            detected[self.names['masks']] = masks.astype('uint8')

        return detected

    def filterDetectedByScore(self, detected: dict):
        "Filter detected bboxes and masks by score"
        num_detections = detected[self.names['num']]
        classes = detected[self.names['classes']]
        detected[self.names['classes']] = classes.astype('float32')
        for nb in range(num_detections):
            score = detected[self.names['scores']][nb]
            if score < self.min_score_thresh:
                # pdb.set_trace()
                detected[self.names['boxes']][nb,:].fill(np.nan)
                if self.names['masks'] in detected:
                    detected[self.names['masks']][nb,:,:].fill(np.nan)
                detected[self.names['classes']][nb] = np.nan
                detected[self.names['scores']][nb] = np.nan
                num_detections -= 1
        #
        detected[self.names['num']] = num_detections
        if self.names['masks'] in detected:
            detected[self.names['masks']] = detected[self.names['masks']].astype('uint8')
        detected = self.removeNans(detected)
        return detected

    def getDetectedClassNames(self, detected: dict):
        "Get detected class names from label map"
        classes = detected[self.names['classes']]
        detected['class_names'] = []
        for cs in classes:
            cname = self.label_map[cs]
            detected['class_names'].append(cname)
        #
        return detected

    def classify_image(self,
                       img: np.ndarray([], dtype=np.uint8)):
        "Get bboxes on detected objects in the image given the graph"
        assert img.dtype == 'uint8'
        with self.graph.as_default():
            with tf.Session() as sess:
                tfops = tf.get_default_graph().get_operations()
                tnames = {
                    tensor.name for tfop in tfops for tensor in tfop.outputs
                }
                t_map = self.get_tensor_map_keys(self.names.values(), tnames)
                # pdb.set_trace()
                imtensor = np.expand_dims(img, axis=0)
                # t_map contains tensors
                bboxes = tf.squeeze(t_map[self.names['boxes']], [0])
                scores = tf.squeeze(t_map[self.names['scores']], [0])
                nb_detected_classes = tf.cast(t_map[self.names['num']][0],
                                              tf.float32) ## float type is
                # necessary for filling with nan values later on
                bboxes = tf.slice(bboxes, [0, 0], [nb_detected_classes, -1])
                if self.names['masks'] in t_map:
                    masks = tf.squeeze(t_map[self.names['masks']], [0])
                    masks = tf.slice(masks, [0, 0, 0],
                                     [nb_detected_classes, -1, -1])
                    reframed_masks = ops.reframe_box_masks_to_image_masks(
                        masks, bboxes, img.shape[0], img.shape[1])
                    reframed_masks = tf.cast(
                        tf.greater(reframed_masks, 0.5), tf.float32)
                    t_map[self.names['masks']] = tf.expand_dims(reframed_masks,
                                                                0)
                #
                image_tensor = tf.get_default_graph().get_tensor_by_name(
                    'image_tensor:0')
                detected = self.infer(session=sess,
                                      img=img,
                                      input_map=t_map,
                                      feed_key=image_tensor)
        return detected

    def writeResult(self,
                    detection: dict,
                    name: str):
        "Write results of detections to output folder"
        dtime = str(datetime.datetime.now().replace(microsecond=0).isoformat())
        idstr = str(uuid4())
        savename = name + "--id--" + idstr + "--time--" + dtime + ".json"
        output_path = os.path.join(self.output_dir, savename)
        # pdb.set_trace()
        with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(detection, f, ensure_ascii=False, indent=2)

    def detect_image(self, impath: str):
        "Read image from path and write detection results"
        imname = os.path.basename(impath)
        imbname, imext = stripExt(imname)
        imext = imext.lower()
        assertCond(imext, imext == 'png')
        img = Image.open(impath)
        imarray = convertPilImg2Array(img)
        detection = self.classify_image(imarray)
        detection = self.fixDetectedBboxes(detection, imarray)
        detection = self.filterDetectedByScore(detection)
        detection = self.getDetectedClassNames(detection)
        pdb.set_trace()
        detected = {}
        for key, array in detection.items():
            if isinstance(array, np.ndarray):
                detected[key] = array.tolist()
            else:
                detected[key] = array
        detected['image_path'] = impath
        # pdb.set_trace()
        self.writeResult(detected,
                         imbname)

    def detect_images(self,
                      imgdir: str,
                      imext='png'):
        "Detect images in the image directory"
        imreg = os.path.join(imgdir, '*.'+imext)
        imfiles = glob.glob(imreg)
        for impath in imfiles:
            self.detect_image(impath)


def printIntro():
    print("""

Hello user, welcome to infer.py of ancient-cv!
This script is going to help you infer new bboxes or masks from the
images you might have, using a trained model.
Proceed if following scenarios are applicable to your case:

Scenario 1:

You have trained your model on some data using train.py and you would like to
use your trained model on new similar images that might be interpreted under
the same label namespace.


Scenario 2:

You have some images, and you would like to try out some of the models that 
are already trained using a label namespace of your choice.
You simply want the bounding boxes/masks concerning the images

Else:

This script covers mostly these two use cases. If yours is not covered 
above, it is highly probable that you are trying to do something else than
inferring bounding boxes on images. Please checkout the getting started with 
object detection section of docs to figure out what you might be looking for.

    """)


def exitApplication():
    print('You have chosen to quit the application')
    print('Bye bye ...')
    sys.exit(0)


def getGraphPathFromUser(confglob1):
    print('You have chosen to skip exporting your model')
    frozen_graph_glob = os.path.join(confglob1, '*')
    frozen_graph_glob = os.path.join(frozen_graph_glob, '**')
    graph_paths = glob.glob(frozen_graph_glob, recursive=True)
    graph_paths = [g for g in graph_paths if 'frozen_inference_graph' in g]
    print('Here are available frozen inference graphs:\n')
    for p in graph_paths:
        bname = os.path.basename(p)
        pdirname = os.path.dirname(p)
        sessiondir = os.path.dirname(pdirname)
        confdir = os.path.dirname(sessiondir)
        modeldir = os.path.dirname(confdir)
        labeldir = os.path.dirname(modeldir)
        print('label namespace: ', os.path.basename(labeldir))
        print('model name: ', os.path.basename(modeldir))
        print('configuration name: ', os.path.basename(confdir))
        print('full graph path: ', p)
        print('\n')
    graph_path = input(
        """
        Please enter full path of the frozen_inference_graph you would
        like to use for inference: 
        """
    )
    return graph_path


def getGraphPathFromExport(conffiles: list):
    "Get graph_path after exporting the trained model"
    print('Here are your configuration files: \n')
    for cfile in conffiles:
        confdir = os.path.dirname(cfile)
        modeldir = os.path.dirname(confdir)
        labeldir = os.path.dirname(modeldir)
        print('\n')
        print('label namespace: ', os.path.basename(labeldir))
        print('model name: ', os.path.basename(modeldir))
        print('configuration name: ', os.path.basename(confdir))
        print('full configuration path: ', cfile)
        print('\n')
    #
    confpath = input(
        """
        Please enter the full path of your configuration file: 
        """)
    exporter = TFGraphExporter(config_path=confpath)
    exporter.exportGraph()
    confdir = os.path.dirname(confpath)
    sessiondir = os.path.join(confdir, 'tfsession')
    graphdir = os.path.join(sessiondir, 'frozen_graph')
    graph_path = os.path.join(graphdir, 'frozen_inference_graph.pb')
    return graph_path


def main():
    printIntro()
    strinput = input('Would like to continue ? [y/n, y by default] ')
    if strinput == 'n':
        exitApplication()
    #
    print("""

You need to export your frozen graph from your trained model, if you have not
done so already.

In order to do that, we need the full path of the configuration file you have
used during your training (don't worry we will show all the configuration 
files if you don't remember the exact path).

If you already have a frozen graph you can skip this step.

          """)
    confglob1 = os.path.join(modelsdir, '*')
    confglob = os.path.join(confglob1, '**')
    conffiles = glob.glob(confglob, recursive=True)
    conffiles = [f for f in conffiles if '.config' in f and 'conf' in f]
    # pdb.set_trace()
    strinput = input(
        'Would you like to export your model ? [y/n, y by default] '
    )
    if strinput == 'n':
        graph_path = getGraphPathFromUser(confglob1)
    else:
        graph_path = getGraphPathFromExport(conffiles)
    #
    frozendir = os.path.dirname(graph_path)
    sessiondir = os.path.dirname(frozendir)
    outputdir = os.path.join(sessiondir, 'detection_output')
    strinput = input("""
Please provide the full path to the directory holding the images you want to
infer (only png images are supported, 
do not put slash at the end of path):  
    """
                     )
    minscore_thresh = input("""
Please enter a minimum score threshold to filter weak 
detections [0.5 by default]: 
    """)
    if minscore_thresh == "":
        minscore_thresh = 0.5
    else:
        minscore_thresh = float(minscore_thresh)
    #
    classifier = ImageClassifierTF(graph_path=graph_path,
                                   min_score_thresh=minscore_thresh)
    classifier.detect_images(strinput)
    print(
        'Inference results have been written in json format to {0}'.format(
            outputdir
        )
    )


if __name__ == '__main__':
    main()
