#####################################################
Getting Started with Object Detection in Ancient-CV
#####################################################

Installation
=============

See install section


General View of the Pipeline
=============================

Object detection involves classifying certain image parts as predefined
objects.

Any pipeline implementing detection of objects in images needs three
components:

- Data that defines which zone in the image are considered as regions of
  interest. These regions of interest are mostly defined as bounding boxes or
  masks.

- A model that is trained to recognize these regions of interests in images.

- Some glue code that manages the handling of the interactions between these
  two.

A very simple use case is the following::

    I have let's say 200 images. They are written in ancient egyptian. Some
    are in demotic, other are in hieroglyphics. I want to detect the name of
    the king.

From this use case we can see the following input fields that are emerging:

- We have a set of labels:

  - hieroglyph king name

  - demotic king name

- We have a lot of images

We can also see the following output field that are emerging:

- The position of the king's name in a document if it exists

Notice that our detector does not give us which king name is in the document
it only gives the position of the name of the king if it exist.

Based on this use case we will see how different parts of object detection api 
of ancient-cv can be used.

Globally speaking we shall see the following parts:

- Creating a label file and annotating images with labels
- Transforming annotations into a dataset
- Choosing a model to train on the dataset with its configurations
- Training the model with the given dataset
- Applying the trained model to previously unseen images
- Verifying and correcting its results and retraining the model with corrected
  annotations.

These parts map fairly straightforward to the scripts provided in the project.
Here is the project outline of the relevant sections:

.. code::

    + ancient-cv
      + main
        + data
        + core
        + models
        + modules
          + preprocessor
            + images
              - annotate.py
              - import_label.py
              - modify_annotations.py
              - imutils.py
              + assets
              + images
              + labels
              + annotations
          + detector
            + tf_models
        - config.py
        - infer.py
        - load.py
        - train.py

You can follow the instructions here, by using images in EgyptianKingName
under the images directory of assets

Creating Your Label File aka Your Label Namespace
==================================================

Somewhere on your computer simply create new file and add the following
labels in it separated by a newline.
Here is an example file::

    Hieroglyph king name
    Demotic king name

And let's say the name of the file is :code:`HieroglyphDemoticKingName.txt`.

In linux terminal:

- :code:`~$ touch HieroglyphDemoticKingName.txt`
- :code:`~$ editorOfYourChoice HieroglyphDemoticKingName.txt`
- Simply copy paste the labels above.
- Save changes and exit.

Label name space is an important part of the api. It help us to distinguish
semantics of different labels per project basis. 
The exact definition of the labels can be determined in collaboration with 
different specialist that will use and/or benefit from the project.
Most of the api is constituted so that different projects can benefit from the
use of object detection without compromising the singularity or uniqueness of
the semantics that they are trying to capture using their data.

That is why to ensure the singularity of each label name space we have
provided the :code:`import_label.py`

Go to the location of the script (see the project outline above) and launch:

:code:`python import_label.py`

Follow the instructions on the terminal.


Annotating Your Data
=====================

:code:`annotate.py` is provided so that you can annotate your data using
`labelme <url to label me>`. It is a great project in general and check them
out when you have the time. 
:code:`annotate.py` not only launches :code:`labelme` with the labels of your
label namespace but also creates some necessary folders for the project later
on.
You can launch the :code:`annotate.py` 

.. code::
    
    PATH_TO_REPO/ancient-cv/main/modules/preprocessor/images$ python annotate.py
    Follow the instructions on terminal

Make sure you save your annotations under the folder: 
:code:`assets/annotations/HieroglyphDemoticKingName`

You can also create a folder with the same label name space under images to 
put your images that are going to be annotated if you want to be
extra organized but it is not necessary.
