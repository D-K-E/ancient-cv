# Author: Kaan Eraslan
# License: see, LICENSE
# No warranties for details, see LICENSE

from google.protobuf import text_format
from google.protobuf import json_format
from utils import getTFCheckpointPath
from utils import ConfigRW
from utils import stripExt
import os
import sys
import subprocess
import glob


curdir = os.getcwd()
modulesdir = os.path.join(curdir, 'modules')
detectdir = os.path.join(modulesdir, 'detector')
tfmodelsdir = os.path.join(detectdir, 'tf_detector')
researchdir = os.path.join(tfmodelsdir, 'research')
objdir = os.path.join(researchdir, 'object_detection')
slimdir = os.path.join(researchdir, 'slim')
modelmainpath = os.path.join(objdir, 'model_main.py')
protosdir = os.path.join(objdir, 'protos')

sys.path.append(researchdir)
sys.path.append(slimdir)
# sys.path.append(objdir)

from modules.detector.tf_detector.research.object_detection.protos.pipeline_pb2 import TrainEvalPipelineConfig as PipelineConfig


def printIntro():
    print("Welcome to training script of ancient-cv")
    print("Before proceeding further please evaluate your case:")
    print("You can quit anytime with ctrl+c")
    print("""
Scenario 1:

1. You have created annotations with annotate.py
2. You have loaded them to their proper place using load.py
3. You have created a valid configuration file using config.py

Scenario 2:

1. You have converted your data to TFRecord required by the object_detection
api
2. You are wondering what to do next ?

Scenario 3:

1. You have your TFRecord ready and it is prepared to fit into the expected
    directory structure of the project
2. You want to change some values on the configuration file.

For Scenario 1:
    Please proceed with the rest of the script.

For Scenario 2:
    You should do the following:
    1. Create folder under the name of the label namespace you have
    used in your annotations.

    2. Put a valid label_map file with the extension .pbtxt in it.
    [for valid label_maps see the examples in the object_detection/data/]

    3. Create a folder under the name 'train' and put your TFRecord for
    training inside it.

    4. Create a folder under the name 'eval' and put your TFRecord for
    evaluation inside it.

    5. Launch config.py to generate a valid configuration file for the
    training job.

    6. Then proceed with the script.

For Scenario 3:

    There are two options:

    1. You can either modify the default values of the desired model inside
    the config file [not recommended]

    2. You can create a new config.py file called 'newConfig.py'.
    Import everything from config.py.
    Subclass the desired model.
    Copy main function from config.py to newConfig.py.
    Add your subclass under the available_models dictionary in main function.
    Call main function at the end of your newConfig.py.
    Launch the newConfig.py with python newConfig.py
    If everything goes okay, you should have your new config with the
    parameters you have set under the folder
    models/LabelNameSpace/YourNewSubClassedModel/conf-uuid.config

    Since in most cases you don't want to change every parameter in a model's
    config, each model comes with different levels of granularity to set
    parameters.
    Best approach is to subclass the desired model, and override
    the proceedure that sets a value to a single field or a group of fields in
    the generated config file.

    """)


def addModelCkpt2Config_proc(confPath: str,
                             session_backend='tf'):
    #
    confname = os.path.basename(confPath)
    confoname, ext = stripExt(confname)
    confPathDir = os.path.dirname(confPath)
    sessionDir = os.path.join(confPathDir,
                              session_backend+'session')
    ckpt_path = getTFCheckpointPath(sessionDir)
    ckpt_name = os.path.basename(ckpt_path)
    print('Here is the last checkpoint: ', ckpt_name)
    print('Please provide training step higher than the number shown above.')
    conf_io = ConfigRW(ipath=confPath,
                       odir=confPathDir,
                       oname=confoname,
                       messageProto=PipelineConfig())
    conf_io.readConfig()
    conf_io.config['trainConfig']['fineTuneCheckpoint'] = ckpt_path
    conf_io.config['trainConfig']['fromDetectionCheckpoint'] = True
    conf_io.config['trainConfig']['loadAllDetectionCheckpointVars'] = True
    conf_io.writeConfig()
    return


def getConfPathFromUser_proc(modelsdir: str):
    "Get configuration file from user using conf files in models dir"
    print('Here is a list of available config files:\n')
    modelsRegexp = os.path.join(modelsdir, "**")
    modelsRegexp = os.path.join(modelsRegexp, "*.config")
    files = glob.glob(modelsRegexp, recursive=True)
    files = {os.path.basename(f): f for f in files}
    files = {
        name: path for name, path in files.items() if (
            '.config' in name and 'conf-time' in name)
    }
    for name, confpath in files.items():
        configPathDir = os.path.dirname(confpath)
        modelPathDir = os.path.dirname(configPathDir)
        labelPathDir = os.path.dirname(modelPathDir)
        labelname = os.path.basename(labelPathDir)
        #
        print("label namespace: ", labelname)
        modelname = os.path.basename(modelPathDir)
        print("model name: ", modelname)
        print("config name: ", name)
        print("\n")
    #
    strinput = input('Please enter the config name of your choice: ')
    return files[strinput]


def main():
    "Main program"
    printIntro()
    curdir = os.getcwd()
    modelsdir = os.path.join(curdir, 'models')
    strinput = input("Do you want to proceed further [y/n, default y]: ")
    #
    if strinput == 'n':
        print('Quitting program... Good luck with the rest of your work.')
        sys.exit(0)
    #
    strinput = input("""
You can either choose among available config files [y] or
Enter directly the full path of the config file yourself [n]

Please enter your answer [y/n, n by default]
    """)
    if strinput == 'y':
        confPath = getConfPathFromUser_proc(modelsdir)
    else:
        confPath = input("Please enter the full path of the config file: ")
    #
    confPathDir = os.path.dirname(confPath)
    strinput = input("Would you like to continue a training job [y/N]? ")
    if strinput == 'y' or strinput == 'Y':
        # Add checkpoint path to config file
        addModelCkpt2Config_proc(confPath)
    sessionDir = os.path.join(confPathDir, 'tfsession')

    nbTrainingSteps = input("Enter number of training steps: ")
    nbSampling = input("Pick 1 of n examples for training. Enter a value for"
                       " n [1 by default]: ")
    if nbSampling == "":
        nbSampling = "1"
    # print("sys path: ", sys.path)
    newpath = ":".join(sys.path)[1:]
    osenv = os.environ.copy()
    ospath = osenv['PATH']
    ldpath = osenv['LD_LIBRARY_PATH']
    # print("os path: ", ospath)
    print('\n')
    print('Training process tend to take long.')
    print('Please verify the following before training starts: ')
    print('\n')
    print('pipeline config path: ', confPath)
    print('session dir to which training events will be saved: ', sessionDir)
    print('number of training steps: ', nbTrainingSteps)
    print('sample 1 of n examples', nbSampling)
    print('\n')
    strinput = input('proceed to training with these values ? [y/n, y by'
                     'default] ')
    if strinput == 'n':
        print('Provided values are not correct')
        print('Quitting the application...')
        sys.exit(0)
    subprocess.run(["python3",
                    modelmainpath,
                    "--pipeline_config_path="+confPath,
                    "--model_dir="+sessionDir,
                    "--num_train_steps="+nbTrainingSteps,
                    "--sample_1_of_n_eval_examples="+nbSampling,
                    "--alsologtostderr"],
                   env={"PYTHONPATH": newpath,
                        "PATH": ospath,
                        "LD_LIBRARY_PATH": ldpath}
                   )


if __name__ == '__main__':
    main()
