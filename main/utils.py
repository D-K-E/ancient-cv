# author: Kaan Eraslan
# Licence: see, LICENSE
# No warranties, use it on your own risk, see LICENSE

from modules.preprocessor.images.imutils import ImagePreprocessor

import os  # path manipulation
import glob
import io
import tensorflow as tf
import numpy as np  # working with image matrix
from PIL import Image  # imageio and conversion
from google.protobuf import text_format, json_format

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


def augmentImageData_proc(image: np.ndarray([], dtype=np.uint8),
                          holder: list,
                          shapeInfo: list,
                          imname: str,
                          labelNameSpace: str):
    "Augment image data by applying image preprocessor"
    processor = ImagePreprocessor(image)
    imdict = processor.transform_pixel_energy()
    for key, img in imdict.items():
        holder.append({'info': shapeInfo,
                       'image': img,
                       'imageName': imname + "_" + key,
                       'labelNamespace': labelNameSpace
                       })

    return holder


def convertPilImg2Array(image):
    "Convert the pil image to numpy array"
    imarr = np.array(image, dtype=np.uint8)
    if len(imarr.shape) > 2:
        imarr = imarr[:, :, :3]
    #
    return imarr


def convertNpImg2BytesPng(image: np.ndarray):
    "Converts the numpy matrix into png bytes"
    img = Image.fromarray(image)
    imio = io.BytesIO()
    img.save(imio, format='PNG')
    return imio


def stripExt(str1: str, ext_delimiter='.') -> [str, str]:
    "Strip extension"
    strsplit = str1.split(ext_delimiter)
    ext = strsplit.pop()
    newstr = ext_delimiter.join(strsplit)
    return (newstr, ext)


def strip2end(str1: str, fromStr: str):
    "Strip str1 from fromStr until its end"
    assert fromStr in str1
    pos = str1.find(fromStr)
    newstr = str1[:pos]
    return newstr


def strip2ends(str1: str, fromStrs: [str]):
    "Strip a list of string from str1 end"
    newstr = ""
    for fromStr in fromStrs:
        if fromStr in str1:
            newstr = strip2end(str1, fromStr)
        else:
            continue
    return newstr


def stripPrefix(str1: str, prefix: str):
    "Strip prefix from str1"
    assert prefix in str1
    assert str1.startswith(prefix)
    preflen = len(prefix)
    newstr = str1[preflen:]
    return newstr


def stripStr(str1: str,
             prefix: str,
             fromStr: str):
    "Strip string"
    newstr = stripPrefix(str1, prefix)
    return strip2end(newstr, fromStr)


def stripStrs(str1: str,
              prefix: str,
              fromStrs: [str]):
    "Strip string applied to different fromstrings"
    newstr = ""
    for fromStr in fromStrs:
        if fromStr in str1:
            newstr = stripStr(str1, prefix, fromStr)
        else:
            continue
    return newstr


def assertCond(var, cond: bool, printType=True):
    "Assert condition print message"
    if printType:
        assert cond, 'variable value: {0}\nits type: {1}'.format(var,
                                                                 type(var))
    else:
        assert cond, 'variable value: {0}'.format(var)


def setParams2Dict(paramDict: dict,
                   keyNameTree: [str],
                   val):
    "Travers the parameter dict and set a value to its last key name"
    # creates the key if it does not exist
    assertCond(keyNameTree, isinstance(keyNameTree, list))
    var = [(isinstance(k, str), k, type(k)) for k in keyNameTree]
    assertCond(var, all([isinstance(k, str) for k in keyNameTree]),
               printType=False)

    lastindex = len(keyNameTree) - 1
    for i, key in enumerate(keyNameTree):
        if i == lastindex:
            paramDict[key] = val
        else:
            if key not in paramDict:
                paramDict[key] = {}
            paramDict = paramDict[key]


def getRecordNbFromTFRecord(recordFile) -> int:
    "Get number of records from tf record"
    count = 0
    for record in tf.python_io.tf_record_iterator(recordFile):
        count += 1
    return count


def parseConfName(confname: str):
    "Parse a configuration file name"
    timelen = len('time--')
    idlen = len('--id--')
    conftimepos = confname.find('time--')
    confidpos = confname.find('--id--')
    conftime = confname[conftimepos:confidpos]
    conftime = conftime[timelen:]
    confnoext, ext = stripExt(confname)
    confid = confnoext[confidpos+idlen:]
    return {"config_full_name": confname,
            "config_idstr": confid,
            "config_time": conftime,
            "config_extension": ext}


def getTFCheckpointPath(session_path: str):
    "Get highest model checkpoint from session path"
    cglob = os.path.join(session_path, 'model.ckpt-*')
    checkpoints = glob.glob(cglob)
    bnames = [(os.path.basename(p), p) for p in checkpoints]
    fromStrs = ['.data', '.index', '.meta']
    prefix = 'model.ckpt-'
    newnames = []
    for bname, path in bnames:
        newname = stripStrs(bname,
                            prefix,
                            fromStrs)
        newnames.append((newname, path))
    #
    newnames = [(int(name), path) for name, path in newnames]
    newnames.sort(key=lambda x: x[0])
    ckpt_path = newnames[-1][1]  # path
    ckpt_dir = os.path.dirname(ckpt_path)
    bname = os.path.basename(ckpt_path)
    newname = strip2ends(bname, fromStrs)
    ckpath = os.path.join(ckpt_dir, newname)
    return ckpath


class FilesLoader:
    "Load files with given extension from given path"

    def __init__(self, path: str, ext: str):
        self.path = path
        self.ext = ext
        self.files = []

    def getFiles(self):
        "Get files from path with extension"
        filesPath = os.path.join(self.path, '*.' + self.ext)
        filePaths = glob.glob(filesPath)
        return filePaths

    def loadFiles(self, mode='r'):
        "Load files that are fetched from given path"
        filePaths = self.getFiles()
        self.files = []
        if mode == 'rb':
            encoding = None
        elif mode == 'r':
            encoding = 'utf-8'
        else:
            raise ValueError("""Unsupported Mode: {0}, allowed ones r,
                             rb""".format(mode))
        for p in filePaths:
            with open(p, mode, encoding=encoding) as f:
                newFile = f.read()
                name = os.path.basename(p)
                basedir = os.path.dirname(p)
                parentFolderName = os.path.basename(basedir)
                self.files.append(
                    {"file": newFile,
                     "name": name,
                     "ext": self.ext,
                     "dataName": name[:len(name)-len(self.ext)-1],
                     "dirName": parentFolderName
                     })
        return self.files


class FileRW:
    "Read/write a file from given path, name, extension"

    def __init__(self,
                 ipath: str,
                 odir: str,
                 oname: str,
                 oext: str,
                 ):
        self.input_path = ipath
        iname = os.path.basename(ipath)
        self.input_name, self.input_ext = stripExt(iname) 
        self.output_dir = odir
        self.output_name = oname
        self.output_ext = oext

    def readFile(self, mode='r'):
        "read file with given params in constructor"
        with open(self.input_path, mode) as f:
            newFile = f.read()
        return newFile

    def writeFile(self, afile, mode='w'):
        "write file with given params in constructor"
        filepath = os.path.join(self.output_dir,
                                self.output_name + "." + self.output_ext)
        with open(filepath, mode) as f:
            f.write(afile)


class ConfigRW(FileRW):
    "Read config file from given path, name, extension"

    def __init__(self,
                 ipath: str,
                 odir: str,
                 oname: str,
                 oext='config',
                 config={},
                 messageProto=None
                 ):
        super().__init__(ipath=ipath,
                         odir=odir,
                         oname=oname,
                         oext=oext)
        self.config = config
        self.messageProto = messageProto

    def readConfig(self) -> dict:
        "Read config file and parse it as dict"
        configString = self.readFile()
        message = text_format.Parse(configString, self.messageProto)
        self.config = json_format.MessageToDict(message)
        return self.config

    def writeConfig(self):
        "Write config file to specified path with name and ext in constructor"
        message = json_format.ParseDict(self.config, self.messageProto)
        messageString = text_format.MessageToString(message, as_utf8=True)
        self.writeFile(afile=messageString)
