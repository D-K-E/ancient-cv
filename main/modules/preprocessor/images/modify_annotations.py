# author: Kaan Eraslan
# license: see, LICENSE
# No warranties, see LICENSE

import os
import sys
import json
import glob

currentdir = os.getcwd()
assetdir = os.path.join(currentdir, 'assets')
annotationsdir = os.path.join(assetdir, 'annotations')

class PathChanger:
    "Change path of annotation"
    def __init__(self, newparentdir: str,
                 annotation: dict):
        self.anno = annotation
        self.newdir = newparentdir

    def changePath(self):
        'Change path of annotation'
        impath = self.anno['imagePath']
        bname = os.path.basename(impath)
        newpath = os.path.join(self.newdir, bname)
        self.anno['imagePath'] = newpath
        return self.anno


def changeAnnotationPath(annoPath: str, newpath: str):
    "procedure for path changer"
    assert isinstance(annoPath, str)
    assert isinstance(newpath, str)
    with open(annoPath, 'r', encoding='utf-8') as f:
        annotation = json.load(f)
    changer = PathChanger(annotation=annotation, newparentdir=newpath)
    annotation = changer.changePath()
    with open(annoPath, 'w', encoding='utf-8') as f:
        json.dump(annotation, f, ensure_ascii=False, indent=2)


class LabelChanger:
    "Change label of the annotation"

    def __init__(self, shape: dict,
                 oldlabel: str,
                 newlabel: str):
        self.shape = shape
        self.newlabel = newlabel
        self.oldlabel = oldlabel

    def changeLabel(self):
        if self.shape['label'] == self.oldlabel:
            self.shape['label'] = self.newlabel
        return self.shape


def changeLabel4Annotation(oldnew: [(str, str)],
                           annoPath):
    "Procedure for changing labels in annotations"
    assert isinstance(annoPath, str)
    with open(annoPath, 'r', encoding='utf-8') as f:
        annotation = json.load(f)
    shapes = annotation['shapes']
    newshapes = []
    for shape in shapes:
        for on in oldnew:
            old = on[0]
            new = on[1]
            changer = LabelChanger(shape, oldlabel=old, newlabel=new)
            shape = changer.changeLabel()
        #
        newshapes.append(shape)
    annotation['shapes'] = newshapes
    with open(annoPath, 'w', encoding='utf-8') as f:
        json.dump(annotation, f, ensure_ascii=False, indent=2)


def parseOldNewLabels(input_str: str):
    "parse old new labels in input string"
    oldnewlist = input_str.split(';')
    oldnewlist = [el.strip() for el in oldnewlist]
    newlist = []
    for el in oldnewlist:
        newel = el.split(',')
        newel = [n.strip() for n in newel]
        newel = tuple(newel)
        newlist.append(newel)
    return newlist


def getAnnotations(parentdir):
    "Get annotations from parent dir using .json extension"
    annoglob = os.path.join(parentdir, '*.json')
    annotations = glob.glob(annoglob)
    return annotations


def printIntro():
    print("""

Welcome to modify annotations script of ancient-cv.

This script is for the following two use cases.

Use case 1:
    - You have modified a label in a label file and you want now to modify
      annotations that have the old label.

Use case 2:
    - You have moved the images to another location and you want now to change
      the image path registered in annotations

If your use case does not coincide with one of those feel free to create an
issue in the repository

    """)

def printQuit():
    print('You have chosen to quit the script')
    print('Bye Bye ...')
    sys.exit(0)


def main():
    printIntro()
    strinput = input('Would you like to continue [y/n y by default] ')
    if strinput == 'n':
        printQuit()
    #
    annotationGlob = os.path.join(annotationsdir, '*')
    annoLabelNamePaths = glob.glob(annotationGlob)
    annoLabelNamePaths = {os.path.basename(p):p for p in annoLabelNamePaths}
    print('Please choose the label namespace of your annotations.')
    print('Here are available name spaces: ')
    for bname in annoLabelNamePaths.keys():
        print('\n', bname)
    #
    print('\n')
    labelName = input('Please enter your choice: ')
    annoLabelPath = annoLabelNamePaths[labelName]
    annotations = getAnnotations(annoLabelPath)
    usecase = input('Please choose your use case [1 or 2, by default 1]: ')
    if usecase == '2':
        f_path = input(
            'Please enter the full path of the folder containing the images: '
        )
        for annoPath in annotations:
            changeAnnotationPath(annoPath, newpath=f_path)
    #
    print(
        'Please enter your old and new labels separated by , and ;'
    )
    print('Here is an example: \n')
    print("""
first old label, my first new label; second old label, my second new label
    """)
    strinput = input(
        'Please enter your list of modified labels as shown above: '
    )
    oldnewlist = parseOldNewLabels(strinput)
    for annoPath in annotations:
        changeLabel4Annotation(oldnewlist, annoPath)
    #
    print('Modifications are done')
    print('Quitting script.')
    print('Bye bye ...')
    sys.exit(0)

if __name__ == '__main__':
    main()
