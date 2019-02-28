# author: Kaan Eraslan
# license: see, LICENSE
# No warranties, see LICENSE

import os
import sys
import glob
from uuid import uuid4


def stripTxtExt(string: str):
    "Strip txt extension"
    assert isinstance(string, str)
    extlen = len('.txt')
    assert len(string) > extlen
    stripped = string[:len(string)-extlen]
    return stripped


def checkExtension(path: str):
    "Check if the extension of the provided file is .txt"
    extlen = len('.txt')
    ext = path[len(path)-extlen:]
    extlow = ext.lower()
    return extlow == '.txt'


def checkPathInLabelFiles(labelfiles: [str],
                          path: str):
    "Check if path is in labelfiles"
    bnames = [os.path.basename(f).lower() for f in labelfiles]
    strippedBnames = map(stripTxtExt, bnames)
    pathbname = os.path.basename(path)
    pathbname = stripTxtExt(pathbname)
    return pathbname.lower() in strippedBnames


def printIntro():
    print('Hello, and welcome to import_label script of'
          ' ancient-cv.')
    print('This program will help you import files that contain a list of'
          ' newline separated labels.')
    print('The name of these files will be treated as label namespaces.')
    print('Label namespaces help us to distinguish semantics of projects from'
          ' one another, so they are an important part of our pipeline.')
    print('They should be unique.')
    print('This script will simply ensure that there would be no namespace'
          ' collusions between the imported file and the already existing'
          ' files.')
    print('This script is going to be replaced with a graphical user '
          'interface in near future. However until that period users are'
          ' advised to use this script to import their data rather than'
          ' moving their label files to assets/labels/ directory on their'
          ' own.')
    print('\n')
    print('Before proceeding further make sure the following are valid for'
          ' your case: \n')
    print('    - Your file uses newlines to separate labels')
    print('    - Your file has the extension <.txt>')
    print('    - Your file uses utf-8 version of unicode encoding')
    print('\n')


def raiseExtCheck(extCheck, path):
    if extCheck is False:
        raise ValueError("""

Provided path: {0}
The file in the provided path does not have the extension <.txt>.
Please provide a different file or rename the extension of your file.

                         """.format(path)
                         )


def raiseInLabels(inLabel: bool, path: str):
    if inLabel is True:
        bname = os.path.basename(path)
        bname = bname + '--' + str(uuid4())
        raise ValueError("""

provided path: {0}
Label namespace in the provided path already exists.
Either rename your file or choose the one that is available in labels folder 
for your annotation job.
Here is a suggested name for your label file if you would like to rename it: 
{1}
                         """.format(path,bname)
                         )


def seeFileContents(labelfile):
    print('Here is the content of your file\n')

    for line in labelfile.splitlines():
        print('label: ', line)

    print('\n')
    strinput = input('Would you like to quit in order to modify your file'
                     ' ? [y/n, n by default] ')
    if strinput == 'y':
        print('Quitting the script as requested.')
        print('Good luck with the rest of your work')
        sys.exit(0)


def saveFile(oldpath,
             newroot,
             imdir,
             content):
    bname = os.path.basename(oldpath)
    noext = stripTxtExt(bname)
    imfolder = os.path.join(imdir, noext)
    if os.path.isdir(imfolder) is False:
        os.mkdir(imfolder)
    newpath = os.path.join(newroot, bname)
    print('Saving your file to the following path:')
    print(newpath)
    with open(newpath, 'w', encoding='utf-8', newline='\n') as f:
        f.write(content)

    print('Please put your raw png images to the folder: {0}'.format(imfolder))


def main():
    "Instructions for importing label file"
    curdir = os.getcwd()
    assetdir = os.path.join(curdir, 'assets')
    labelsdir = os.path.join(assetdir, 'labels')
    imagesdir = os.path.join(assetdir, 'images')
    labelregex = '*.txt'
    labelregexPath = os.path.join(labelsdir, labelregex)
    labelfiles = glob.glob(labelregexPath)

    printIntro()

    print('If all of the above does not apply to your case you can quit now'
          ' and come back once all of the above is valid for your label file')
    strinput = input('Would you like to quit now ? [y/n, n by default] ')
    if strinput == 'y':
        sys.exit(0)

    print('\n')
    print('Please write the full path to your label file: ')
    path = input('')
    extCheck = checkExtension(path)
    raiseExtCheck(extCheck, path)
    inLabel = checkPathInLabelFiles(labelfiles, path)
    raiseInLabels(inLabel, path)
    makeLabeldirUnderImages(

    with open(path, 'r', encoding='utf-8') as f:
        labelfile = f.read()

    strinput = input('Would like to see the contents of your file ? [y/n, n by'
                     'default] ')
    if strinput == 'y':
        seeFileContents(labelfile)
        #
    saveFile(oldpath=path,
             imdir=imagesdir,
             newroot=labelsdir, 
             content=labelfile)
    print('Label file imported. Quitting the script...')
    sys.exit(0)


if __name__ == '__main__':
    main()
