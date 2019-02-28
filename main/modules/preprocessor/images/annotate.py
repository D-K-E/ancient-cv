# author: Kaan Eraslan
# license: see, LICENSE
# No warranties, see LICENSE

import os
import glob
import subprocess


def main():
    curdir = os.getcwd()
    assetdir = os.path.join(curdir, 'assets')
    annotationsdir = os.path.join(assetdir, 'annotations')
    labelsdir = os.path.join(assetdir, 'labels')
    labelregex = '*.txt'
    labelregexPath = os.path.join(labelsdir, labelregex)
    labelfiles = glob.glob(labelregexPath)

    print('Here is a list of label namespaces, in this case files containing' 
          ' labels')
    print('Label namespaces are important, because they are also used to '
          'classify the ground truth data of the models later on')
    print('Be careful about not mixing them.')
    print('They are meant to be used as divisions whose semantics can be' 
          ' defined per project basis.\n')
    print('Here are available label namespaces:\n')

    labelDict = {}
    for label in labelfiles:
        namespaceName = os.path.basename(label)
        print('label namespace name: ', namespaceName)
        print('label namespace path: ', label, '\n')
        labelDict[namespaceName] = label

    labelName = input('please write the name of the label namespace of your '
                      'choice without extensions: ')
    labelFileName = os.path.join(annotationsdir, labelName)
    if os.path.isdir(labelFileName) is False:
        os.mkdir(labelFileName)
    labelPath = labelDict[labelName+'.txt']
    subprocess.run(['labelme', 
                    '--labels', 
                    labelPath])


if __name__ == '__main__':
    main()
