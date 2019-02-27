# author: Kaan Eraslan
# Licence: see, LICENSE
# No warranties, use it on your own risk

import json  # json io
import io  # generally required for io
import glob  # finding stuff in paths
import re
import datetime
import uuid
import tensorflow as tf
from google.protobuf import text_format
from google.protobuf import json_format
import pdb
import os  # path manipulation
import sys  # exit system
from utils import assertCond, setParams2Dict, FileRW
from utils import getRecordNbFromTFRecord
from utils import ConfigRW

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
protosdir = os.path.join(objdir, 'protos')

sys.path.append(researchdir)
sys.path.append(objdir)
sys.path.append(slimdir)

from modules.detector.tf_detector.research.object_detection.protos.pipeline_pb2 import TrainEvalPipelineConfig as PipelineConfig
from modules.detector.tf_detector.research.object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap


# Generate Configuration File


class ConfigModel:
    "General config file regroups common methods for all config files"
    AVAILABLE_METRICS_SETS = [
        'oid_challenge_detection_metrics',
        'oid_V2_detection_metrics',
        'coco_mask_metrics',
        'open_images_V2_detection_metrics',
        'weighted_pascal_voc_instance_segmentation_metrics',
        'pascal_voc_instance_segmentation_metrics',
        'weighted_pascal_voc_detection_metrics',
        'pascal_voc_detection_metrics'
    ]
    AVAILABLE_META_MODELS = [
        'faster_rcnn',
        'ssd',
    ]

    def __init__(self, config={}):
        self.config = config

    def setConfigKeys(self):
        "Set config dict keys"
        self.config['model'] = {}
        self.config['train_config'] = {}
        self.config['eval_config'] = {}
        self.config['train_input_reader'] = {}
        self.config['eval_input_reader'] = []

    def setTrainTFRecordPath(self, path: str):
        "Set training tf record path to training_input_reader"
        self.setConfigParamVal(keyNames=['train_input_reader',
                                         'tf_record_input_reader',
                                         'input_path'],
                               val=[path],
                               valType=list,
                               contentType=str)

    def setEvalTFRecordPath2EvalDict(self, evalDict: dict,
                                     paths: [str]):
        "set eval tf record path to eval_input_reader"
        assertCond(paths, isinstance(paths, list))
        assertCond(evalDict, isinstance(evalDict, dict))
        var = [(isinstance(p, str), p, type(p)) for p in paths]
        assertCond(var, all([isinstance(p, str) for p in paths]),
                   printType=False)
        evalDict['tf_record_input_reader'] = {}
        evalDict['tf_record_input_reader']['input_path'] = paths
        return evalDict

    def setEvalShuffle2EvalDict(self,
                                evalDict: dict,
                                isShuffle: bool):
        "Set shuffle field of eval_input_reader"
        assertCond(isShuffle, isinstance(isShuffle, bool))
        assertCond(evalDict, isinstance(evalDict, dict))
        evalDict['shuffle'] = isShuffle
        return evalDict

    def setEvalNumReaders2EvalDict(self,
                                   evalDict: dict,
                                   val: int):
        "set num_readers field of the eval_input_reader"
        assertCond(val, isinstance(val, int))
        assertCond(val, val > 0, printType=False)
        assertCond(evalDict, isinstance(evalDict, dict))
        evalDict['num_readers'] = val
        return evalDict

    def setEvalLabelMapPath2EvalDict(self,
                                     evalDict: dict,
                                     path: str):
        "Set label map path to eval dict"
        assertCond(path, isinstance(path, str))
        assertCond(evalDict, isinstance(evalDict, dict))
        evalDict['label_map_path'] = path
        return evalDict

    def setEvalInputEvalDict(self,
                             labelMapPath: str,
                             inputPathList: [str],
                             isShuffle=False,
                             numReaders=1):
        "Set values to a eval input reader eval dict"
        eDict = {}
        eDict = self.setEvalLabelMapPath2EvalDict(evalDict=eDict,
                                                  path=labelMapPath)
        eDict = self.setEvalNumReaders2EvalDict(evalDict=eDict,
                                                val=numReaders)
        eDict = self.setEvalShuffle2EvalDict(evalDict=eDict,
                                             isShuffle=isShuffle)
        eDict = self.setEvalTFRecordPath2EvalDict(evalDict=eDict,
                                                  paths=inputPathList)
        return eDict

    def setEvalInputReader(self, evalDicts: [dict]):
        "Set eval dicts to eval_input_reader"
        self.setConfigParamVal(keyNames=['eval_input_reader'],
                               val=evalDicts,
                               valType=list,
                               contentType=dict)

    def setTrainLabelMapPath(self, path: str):
        "Set label map path to config both for training and eval"
        self.setConfigParamVal(keyNames=['train_input_reader',
                                         'label_map_path'],
                               val=path,
                               valType=str)

    def setEvalNumExamples(self, val: int):
        "Set num_examples field of eval config"
        assertCond(val, val > 0, printType=False)
        self.setConfigParamVal(keyNames=['eval_config',
                                         'num_examples'],
                               val=val, valType=int)

    def setEvalMetricsSet(self, vals: [str]):
        "Set metrics_set field of eval_config"

        if vals == []:
            self.config['eval_config'].pop('metrics_set', None)
            return
        for val in vals:
            if val not in self.AVAILABLE_METRICS_SETS:
                raise ValueError(
                    "Metrics set not available, see "
                    "available sets: {0}".format(
                        self.AVAILABLE_METRICS_SETS
                    )
                )
        self.setConfigParamVal(keyNames=['eval_config',
                                         'metrics_set'],
                               val=vals,
                               valType=list,
                               contentType=str)

    def setEvalVisualizationsExportPath(self, path: str):
        "visualization_export_dir to eval_config"
        self.setConfigParamVal(keyNames=['eval_config',
                                         'visualization_export_dir'],
                               val=path, valType=str)

    def setMaxEvals(self, val=10):
        "Set max_evals field of eval config"
        assertCond(val, isinstance(val, int))
        if val == 0:
            self.config['eval_config'].pop('max_evals', None)
            return
        self.setConfigParamVal(keyNames=['eval_config',
                                         'max_evals'],
                               val=val,
                               valType=int)

    def setTrainDataAugmentationOpts(self,
                                     val=[{"random_horizontal_flip": {}}]):
        """
        Set data_augmentation_options field of train_config

        Example String Output:
        data_augmentation_options {
            random_horizontal_flip {
            }
            ssd_random_crop {
            }
        }

        Example Json Output:
        'dataAugmentationOptions': [{'randomHorizontalFlip': {}},
        {'ssd_random_crop': {}}],
        """
        assertCond(val, isinstance(val, list))
        if val == []:
            self.config['train_config'].pop('data_augmentation_options',
                                            None)
            return

        self.setConfigParamVal(keyNames=['train_config',
                                         'data_augmentation_options'],
                               val=val,
                               valType=list,
                               contentType=dict)

    def setTrainFineTuneCkpt(self, val: str):
        "Set fine_tune_checkpoint field of train config"
        self.setConfigParamVal(keyNames=['train_config',
                                         'fine_tune_checkpoint'],
                               val=val,
                               valType=str)

    def setTrainDetectionCkpt(self, isCkpt=True):
        "Set from_detection_checkpoint field of train config"
        self.setConfigParamVal(keyNames=['train_config',
                                         'from_detection_checkpoint'],
                               val=isCkpt,
                               valType=bool)

    def setTrainNumSteps(self, val: int):
        "Set num_steps field of train_config"
        assertCond(val, isinstance(val, int))
        if val == 0:
            self.config['train_config'].pop('num_steps', None)
            return
        self.setConfigParamVal(keyNames=['train_config',
                                         'num_steps'],
                               val=val,
                               valType=int)

    def setMetaModel(self, val: str):
        "Set model to config file"
        assertCond(val, isinstance(val, str))
        if val not in self.AVAILABLE_META_MODELS:
            raise ValueError('model: {0} is not supported. '
                             'Available models are {1}'.format(
                                 val,
                                 self.AVAILABLE_MODELS
                             )
                             )
        self.config['model'] = {val: {}}

    def setConfigParamVal(self, keyNames: [str],
                          val,
                          valType=None,
                          contentType=None,
                          noTypeCheck=False):
        "Set value to a parameter on config"
        if noTypeCheck is False:
            assertCond(val, isinstance(val, valType))

        if contentType is not None:
            var = [(isinstance(s, contentType), s, type(s)) for s in val]
            assertCond(var, all([isinstance(s, contentType) for s in val]),
                       printType=False)

        paramDict = self.config
        setParams2Dict(paramDict,
                       keyNameTree=keyNames,
                       val=val)
        self.config = paramDict


class TrainConfigModel(ConfigModel):
    "Handles model specific train config"

    def __init__(self, config={}):
        self.config = config
        self.optType = ""
        confkeys = list(self.config.keys())
        if ('train_config' not in confkeys and
                'trainConfig' not in confkeys):
            self.config['train_config'] = {}

    def setTrainConfigGradientClipping(self, val: float):
        "Set value to gradient_clipping_by_norm field of train config"
        self.setTrainConfigParamVal(keyNameTree=['gradient_clipping_by_norm'],
                                    val=val,
                                    valType=float)

    def setTrainConfigBatchSize(self, val: int):
        "Set value to batch size field of train config"
        self.setTrainConfigParamVal(keyNameTree=['batch_size'],
                                    val=val,
                                    valType=int)

    def setTrainConfigOptimizerKey(self):
        "Set optimizer key to train config"
        self.setTrainConfigParamVal(keyNameTree=['optimizer'],
                                    val={},
                                    valType=dict)

    def setTrainConfigOptUseMovingAverage(self,
                                          val: bool):
        "Set value to train config optimizer field's use_moving_average"
        assertCond(val, isinstance(val, bool))
        self.setTrainConfigParamVal(keyNameTree=['optimizer',
                                                 'use_moving_average'],
                                    val=val,
                                    valType=bool)

    def setTrainConfigOptType(self, optType: str):
        "Set optimizer type"
        assertCond(optType, isinstance(optType, str))
        self.setTrainConfigParamVal(keyNameTree=['optimizer',
                                                 optType],
                                    val={},
                                    valType=dict)
        self.optType = optType

    def setTrainConfigParamVal(self,
                               keyNameTree: [str],
                               val,
                               valType,
                               contentType=None
                               ):
        "Learning rate dict to momentum optimizer"
        keyNameTree.insert(0, 'train_config')
        self.setConfigParamVal(keyNames=keyNameTree,
                               val=val,
                               valType=valType,
                               contentType=contentType)

    def setTrainConfigParamsVal(self, keyNameVals):
        "Set keyname val tuples to train config"
        for keyVal in keyNameVals:
            keyNameTree = keyVal[0]
            val = keyVal[1]
            valType = keyVal[2]
            contentType = keyVal[3]
            self.setTrainConfigParamVal(keyNameTree=keyNameTree,
                                        val=val,
                                        valType=valType,
                                        contentType=contentType)


class ModelConfigModel(ConfigModel):
    "Handles model specific model config"

    def __init__(self,
                 metaModelName: str,
                 modelName: str,
                 config={}
                 ):
        self.config = config
        self.metaModel = metaModelName
        self.modelName = modelName
        super().__init__(config=config)

    def setModelConfigParam(self,
                            keyNameTree: [str],
                            val,
                            valType,
                            contentType=None,
                            noTypeCheck=False):
        "Set va using keyNameTree with func setParams2Dict"
        keyNameTree.insert(0, self.metaModel)
        keyNameTree.insert(0, 'model')
        self.setConfigParamVal(keyNames=keyNameTree,
                               val=val,
                               valType=valType,
                               contentType=contentType,
                               noTypeCheck=noTypeCheck)

    def setNumClasses(self, val: int):
        "Set number of classes parameter for the model"
        assertCond(val, val > 0, printType=False)
        self.setModelConfigParam(keyNameTree=['num_classes'],
                                 val=val,
                                 valType=int)


class ConfigFasterRCNN(ModelConfigModel):
    "General setter for faster_rcnn metamodel"
    AVAILABLE_SCORE_CONVERTERS = ["SOFTMAX", "SIGMOID"]

    def __init__(self,
                 config: dict,
                 modelName: str):
        self.config = config
        self.modelName = modelName
        self.trainConf = TrainConfigModel(config)
        self.metaModel = 'faster_rcnn'
        modelkeys = list(self.config['model'].keys())
        if (self.metaModel not in modelkeys and
                'fasterRcnn' not in modelkeys):
            # the second condition comes from the conversion of keys
            # that comes along with parsing of the document by
            # json_format module
            raise ValueError(
                "provided config's model field keys: {0} do not contain "
                "the name of the meta model of current object instance "
                "which is {1}".format(modelkeys, self.metaModel)
            )
        self.modelName = modelName

        super().__init__(config=config,
                         metaModelName=self.metaModel,
                         modelName=modelName)

    def setImageResizer(self):
        "Set image resizer parameter to model"
        self.setModelConfigParam(keyNameTree=['image_resizer'],
                                 val={},
                                 valType=dict)

    def setKeepAspectRatioResizer(self,
                                  minval: int,
                                  maxval: int):
        "Set keep aspect ratio resizer"
        assertCond(minval, minval > 0, printType=False)
        assertCond([minval, maxval], maxval > minval, printType=False)
        # pdb.set_trace()
        keyNamesMin = ['image_resizer',
                       'keep_aspect_ratio_resizer',
                       'min_dimension'
                       ]
        keyNamesMax = keyNamesMin.copy()
        keyNamesMax[-1] = 'max_dimension'
        self.setModelConfigParam(keyNameTree=keyNamesMin,
                                 val=minval,
                                 valType=int)
        self.setModelConfigParam(keyNameTree=keyNamesMax,
                                 valType=int,
                                 val=maxval)

    def setFirstStageGridAnchorGeneratorScale(self,
                                              scales: [float]):
        "Set value to scales field of first_stage_anchor_generator"

        assertCond(len(scales), len(scales) == 4)
        self.setModelConfigParam(
            keyNameTree=[
                'first_stage_anchor_generator',
                'grid_anchor_generator', 'scales'],
            val=scales,
            valType=list,
            contentType=float
        )

    def setFirstStageGridAnchorGeneratorAspectRatio(self,
                                                    aspect_ratios: [float]
                                                    ):
        "Set aspect ratio for the first_stage_anchor_generator"
        assertCond(len(aspect_ratios), len(aspect_ratios) == 3)

        self.setModelConfigParam(
            keyNameTree=[
                'first_stage_anchor_generator',
                'grid_anchor_generator', 'aspect_ratios'],
            val=aspect_ratios,
            valType=list,
            contentType=float
        )

    def setFirstStageGridAnchorGeneratorHeightStride(self,
                                                     height_stride: int):
        "Set height_stride to first_stage_anchor_generator"
        assertCond(height_stride, height_stride > 0, printType=False)

        self.setModelConfigParam(
            keyNameTree=['first_stage_anchor_generator',
                         'grid_anchor_generator',
                         'height_stride'],
            val=height_stride, valType=int)

    def setFirstStageGridAnchorGeneratorWidthStride(self,
                                                    width_stride: int
                                                    ):
        "Set height_stride to first_stage_anchor_generator"
        assertCond(width_stride, width_stride > 0, printType=False)

        self.setModelConfigParam(
            keyNameTree=['first_stage_anchor_generator',
                         'grid_anchor_generator',
                         'width_stride'],
            val=width_stride,
            valType=int)

    def setFirstStageGridAnchorGenerator(self, scales: [float],
                                         aspect_ratios: [float],
                                         height_stride: int,
                                         width_stride: int):
        "Set grid_anchor_generator to first_stage_anchor_generator"
        self.setFirstStageGridAnchorGeneratorScale(scales=scales)
        self.setFirstStageGridAnchorGeneratorAspectRatio(
            aspect_ratios=aspect_ratios)
        self.setFirstStageGridAnchorGeneratorHeightStride(
            height_stride=height_stride)
        self.setFirstStageGridAnchorGeneratorWidthStride(
            width_stride=width_stride)

    def setFirstStageNmsScoreThreshold(self, val: float):
        "Set first nms score threshold"
        self.setModelConfigParam(keyNameTree=[
            'first_stage_nms_score_threshold'],
            val=val,
            valType=float)

    def setFirstStageNmsIouThreshold(self, val: float):
        "Set first stage nms iou threshold"
        assertCond(val, isinstance(val, float))
        self.setModelConfigParam(keyNameTree=[
            'first_stage_nms_iou_threshold'
        ],
            val=val,
            valType=float)

    def setFirstStageMaxProposals(self, val: int):
        "Set first stage max proposal field"
        assertCond(val, isinstance(val, int))
        self.setModelConfigParam(keyNameTree=[
            'first_stage_max_proposals'
        ],
            val=val,
            valType=int)

    def setFirstStageLocalizationLossWeight(self, val: float):
        "Set first stage localization loss weight"
        assertCond(val, isinstance(val, float))
        self.setModelConfigParam(keyNameTree=[
            'first_stage_localization_loss_weight'
        ],
            val=val,
            valType=float)

    def setFirstStageObjectnessLossWeight(self, val: float):
        "Set first stage objectness loss weight"
        assertCond(val, isinstance(val, float))
        self.setModelConfigParam(keyNameTree=[
            'first_stage_objectness_loss_weight'
        ],
            val=val,
            valType=float)

    def setFirstStageNmsIouScoreMaxProposolLossWeights(self,
                                                       nms_score_thresh,
                                                       nms_iou_thresh: float,
                                                       max_proposal: int,
                                                       local_loss_weight: float,
                                                       obj_loss_weight: float):
        "Set values params"
        self.setFirstStageNmsScoreThreshold(val=nms_score_thresh)
        self.setFirstStageNmsIouThreshold(val=nms_iou_thresh)
        self.setFirstStageMaxProposals(val=max_proposal)
        self.setFirstStageLocalizationLossWeight(
            val=local_loss_weight)
        self.setFirstStageObjectnessLossWeight(
            val=obj_loss_weight)

    def setInitialCropSize(self, val: int):
        "Set initial_crop_size"
        assertCond(val, isinstance(val, int))
        self.setModelConfigParam(keyNameTree=[
            'initial_crop_size'
        ],
            val=val,
            valType=int)

    def setFirstStageMaxpoolKernelSize(self, val: int):
        "Set maxpool kernel size"
        assertCond(val, isinstance(val, int))
        self.setModelConfigParam(keyNameTree=[
            'maxpool_kernel_size'
        ],
            val=val,
            valType=int)

    def setFirstStageMaxpoolStride(self, val: int):
        "Set maxpool stride"
        assertCond(val, isinstance(val, int))
        self.setModelConfigParam(keyNameTree=[
            'maxpool_stride'
        ],
            val=val,
            valType=int)

    def setFirstStageCropSizeMaxpoolKernel(self, crop_size: int,
                                           kernel_size: int,
                                           kernel_stride: int):
        "Set values"
        self.setInitialCropSize(val=crop_size)
        self.setFirstStageMaxpoolKernelSize(val=kernel_size)
        self.setFirstStageMaxpoolStride(val=kernel_stride)

    def setFirstStageCNNHyperparams(self, optype: str):
        "Set first stage box predictor conv hyperparams"
        assertCond(optype, isinstance(optype, str))
        params = {}
        params['op'] = optype
        params['regularizer'] = {}
        params['initializer'] = {}
        self.setModelConfigParam(keyNameTree=[
            'first_stage_box_predictor_conv_hyperparams'
        ],
            val=params,
            valType=dict)

    def setFirstStageCNNRegularizer(self,
                                    regularizerType: str,
                                    keyvals: list
                                    ):
        "Set regularizer 2 first stage box predictor conv hyperparams"
        assertCond(regularizerType, isinstance(regularizerType, str))
        assertCond(keyvals, isinstance(keyvals, list))
        for keyval in keyvals:
            param = keyval[0]
            assertCond(param, isinstance(param, str))
            val = keyval[1]
            self.setModelConfigParam(keyNameTree=[
                'first_stage_box_predictor_conv_hyperparams',
                'regularizer',
                regularizerType,
                param],
                val=val,
                valType=None,
                noTypeCheck=True)

    def setFirstStageCNNInitializer(self,
                                    initializerType: str,
                                    keyvals: list):
        "Set first stage cnn initializer"
        assertCond(initializerType, isinstance(initializerType, str))
        assertCond(keyvals, isinstance(keyvals, list))

        for keyval in keyvals:
            param = keyval[0]
            val = keyval[1]
            self.setModelConfigParam(keyNameTree=[
                'first_stage_box_predictor_conv_hyperparams',
                'initializer',
                initializerType,
                param],
                val=val,
                valType=None,
                noTypeCheck=True)

    def setFirstStageBoxPredictor(self, optype: str,
                                  regularizerType: str,
                                  regularizerKeyvals: list,
                                  initializerType: str,
                                  initializerKeyvals: list):
        "set first stage box predictor"
        self.setFirstStageCNNHyperparams(optype=optype)
        self.setFirstStageCNNRegularizer(
            regularizerType=regularizerType,
            keyvals=regularizerKeyvals
        )
        self.setFirstStageCNNInitializer(
            initializerType=initializerType,
            keyvals=initializerKeyvals
        )

    def setFeatureExtractor(self):
        "Set feature extractor to config"
        extractorType = {"type": self.modelName}
        self.setModelConfigParam(keyNameTree=['feature_extractor'],
                                 val=extractorType,
                                 valType=dict)

    def setFirstStageFeatureStride(self, val: int):
        "Set feature extractor first stage feature stride"
        assertCond(val, val > 0, printType=False)
        self.setModelConfigParam(keyNameTree=['feature_extractor',
                                              'first_stage_features_stride'],
                                 val=val,
                                 valType=int)

    def setSecondStageMaskRCNNUseDropOut(self, val: bool):
        "Set second stage mask_rcnn_box_predictor use dropout"
        self.setModelConfigParam(
            keyNameTree=[
                'second_stage_box_predictor',
                'mask_rcnn_box_predictor',
                'use_dropout'],
            val=val, valType=bool)

    def setSecondStageMaskRCNNDropOutProbability(self, val: float):
        "Set second stage dropout probability"
        self.setModelConfigParam(keyNameTree=['second_stage_box_predictor',
                                              'mask_rcnn_box_predictor',
                                              'dropout_keep_probability'],
                                 val=val,
                                 valType=float)

    def setSecondStageFCHyperparams(self, optype: str):
        "Set second stage fc hyperparams"
        keyNames = [
            'second_stage_box_predictor',
            'mask_rcnn_box_predictor',
            'fc_hyperparams',
            'op']
        regKeys = keyNames.copy()
        initKeys = keyNames.copy()
        regKeys[-1] = 'regularizer'
        initKeys[-1] = 'initializer'

        self.setModelConfigParam(keyNameTree=keyNames,
                                 val=optype, valType=str)
        self.setModelConfigParam(keyNameTree=regKeys,
                                 val={},
                                 valType=dict)
        self.setModelConfigParam(keyNameTree=initKeys,
                                 val={}, valType=dict)

    def setSecondStageRegularizer(self,
                                  regularizerType: str,
                                  keyvals: list
                                  ):
        "Set second stage regularizer"
        assertCond(regularizerType, isinstance(regularizerType, str))
        assertCond(keyvals, isinstance(keyvals, list))

        for keyval in keyvals:
            param = keyval[0]
            val = keyval[1]
            assertCond(param, isinstance(param, str))
            self.setModelConfigParam(
                keyNameTree=[
                    'second_stage_box_predictor',
                    'mask_rcnn_box_predictor',
                    'fc_hyperparams',
                    'regularizer',
                    regularizerType,
                    param],
                val=val,
                valType=None, noTypeCheck=True)

    def setSecondStageInitializer(self,
                                  initializerType: str,
                                  keyvals=list):
        "Set second stage initializer"
        assertCond(initializerType, isinstance(initializerType, str))
        assertCond(keyvals, isinstance(keyvals, list))
        for keyval in keyvals:
            param = keyval[0]
            assertCond(param, isinstance(param, str))
            val = keyval[1]
            self.setModelConfigParam(
                keyNameTree=[
                    'second_stage_box_predictor',
                    'mask_rcnn_box_predictor',
                    'fc_hyperparams',
                    'initializer',
                    initializerType,
                    param],
                val=val, valType=None, noTypeCheck=True)

    def setSecondStageBoxPredictor(self,
                                   optype: str,
                                   regularizerType: str,
                                   regularizerKeyvals: list,
                                   initializerType: str,
                                   initializerKeyvals: list
                                   ):
        "Set second stage box predictor vals"
        self.setSecondStageFCHyperparams(optype=optype)
        self.setSecondStageRegularizer(regularizerType=regularizerType,
                                       keyvals=regularizerKeyvals)
        self.setSecondStageInitializer(initializerType=initializerType,
                                       keyvals=initializerKeyvals)

    def setSecondStagePostProcessing(self):
        "Set second stage post processing param dict"
        self.setModelConfigParam(
            keyNameTree=[
                'second_stage_post_processing'
            ], val={},
            valType=dict)

    def setSecondStageBatchNonMaxSuppression(self):
        "Set second stage post processing batch_non_max_suppression"
        self.setModelConfigParam(keyNameTree=[
            'second_stage_post_processing',
            'batch_non_max_suppression'
        ], val={}, valType=dict)

    def setSecondStageBNMSScoreThreshold(self, val: float):
        "Set score threshold to batch_non_max_suppression in second stage"
        self.setModelConfigParam(keyNameTree=['second_stage_post_processing',
                                              'batch_non_max_suppression',
                                              'score_threshold'],
                                 val=val,
                                 valType=float)

    def setSecondStageBNMSIouThreshold(self, val: float):
        "Set iou threshold to batch_non_max_suppression in second stage"
        assertCond(val, isinstance(val, float))
        self.setModelConfigParam(keyNameTree=['second_stage_post_processing',
                                              'batch_non_max_suppression',
                                              'iou_threshold'
                                              ],
                                 val=val, valType=float)

    def setSecondStageBNMSMaxDetectionsPerClass(self, val: int):
        "Set max_detections_per_class to batch_non_max_suppression"
        self.setModelConfigParam(keyNameTree=[
            'second_stage_post_processing',
            'batch_non_max_suppression',
            'max_detections_per_class'
        ],
            val=val, valType=int)

    def setSecondStageBNMSMaxTotalDetections(self, val: int):
        "Set max_total_detections to batch_non_max_suppression"
        self.setModelConfigParam(keyNameTree=[
            'second_stage_post_processing',
            'batch_non_max_suppression',
            'max_total_detections'], val=val, valType=int)

    def setSecondStageBNMS(self, score_threshold: float,
                           iou_threshold: float,
                           max_detections_per_class: int,
                           total_detections: int):
        "set batch_non_max_suppression values"
        self.setSecondStageBNMSScoreThreshold(val=score_threshold)
        self.setSecondStageBNMSIouThreshold(val=iou_threshold)
        self.setSecondStageBNMSMaxDetectionsPerClass(
            val=max_detections_per_class)
        self.setSecondStageBNMSMaxTotalDetections(val=total_detections)

    def setSecondStageScoreConverter(self, val: str):
        "Set score converter to second stage"
        if val not in ConfigFasterRCNN.AVAILABLE_SCORE_CONVERTERS:
            raise ValueError('converter: {0} not available.'
                             'Following converters are available {1}'.format(
                                 val, self.AVAILABLE_SCORE_CONVERTERS
                             ))
        self.setModelConfigParam(keyNameTree=[
            'second_stage_post_processing',
            'score_converter',
        ],
            val=val, valType=str)

    def setSecondStageLocalizationLossWeight(self, val: float):
        "Set second stage localization loss weight"
        self.setModelConfigParam(keyNameTree=[
            'second_stage_localization_loss_weight'],
            val=val, valType=float)

    def setSecondStageClassificationLossWeight(self, val: float):
        "Set second stage classification loss weight"
        self.setModelConfigParam(keyNameTree=[
            'second_stage_classification_loss_weight'],
            val=val,
            valType=float)


class ConfigFasterRCNNInceptionV2Model(ConfigFasterRCNN):
    "Sample Config for FasterRCNNInceptionV2 model"

    def __init__(self,
                 config={}):
        self.config = config
        self.modelName = 'faster_rcnn_inception_v2'
        super().__init__(config=config,
                         modelName=self.modelName)

        # default values
        self.aspectMinVal = 600
        self.aspectMaxVal = 1024
        #  feature extractor params
        self.feature_stride = 16
        #  anchor generator params
        self.scales = [0.25, 0.5, 1.0, 2.0]
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.height_stride = 16
        self.width_stride = 16
        #  first stage box predictor params
        self.first_optype = 'CONV'
        self.first_regOpt = {
            'regularizerType': 'l2_regularizer',
            'keyvals': [('weight', 0.0)]
        }
        self.first_initOpt = {
            'initializerType': 'truncated_normal_initializer',
            'keyvals': [('stddev', 0.01)]
        }
        self.nms_score_thresh = 0.0
        self.nms_iou_thresh = 0.7
        self.max_proposal = 300
        self.first_localization_loss_weight = 2.0
        self.objectness_loss_weight = 1.0
        #  initial crop and the rest
        self.initial_crop_size = 14
        self.maxpool_kernel_size = 2
        self.maxpool_stride = 2
        #  second stage box predictor params
        self.use_dropout = False
        self.dropout_keep_probability = 1.0
        self.second_optype = 'FC'
        self.second_regOpt = {
            'regularizerType': 'l2_regularizer',
            'keyvals': [('weight', 0.0)]
        }
        self.second_initOpt = {
            'initializerType': 'variance_scaling_initializer',
            'keyvals': [('factor', 1.0),
                        ('uniform', True),
                        ('mode', 'FAN_AVG')]
        }
        #  batch_non_max_suppression params
        self.bnms_score_threshold = 0.0
        self.bnms_iou_threshold = 0.6
        self.bnms_max_detections_per_class = 100
        self.bnms_max_total_detections = 300
        self.score_converter = 'SOFTMAX'
        #  second localization classification loss params
        self.second_localization_loss_weight = 2.0
        self.second_classification_loss_weight = 1.0
        # default train config vals
        self.batch_size = 1
        self.gradient_clipping_by_norm = 10.0
        self.use_moving_average = False
        self.opt_type = 'momentum_optimizer'
        self.initial_learning_rate = 0.003
        self.momentum_optimizer_value = 0.9
        self.schedules = [{"step": 900000,
                           "learning_rate": 0.00003},
                          {"step": 1200000,
                           "learning_rate": 0.000003}]

    def setDefaultValConfig2Model(self):
        "Set config params"
        # image resizer
        self.setImageResizer()
        self.setKeepAspectRatioResizer(minval=self.aspectMinVal,
                                       maxval=self.aspectMaxVal)
        # feature extractor
        self.setFeatureExtractor()
        self.setFirstStageFeatureStride(val=self.feature_stride)

        # anchor generator
        self.setFirstStageGridAnchorGenerator(scales=self.scales,
                                              aspect_ratios=self.aspect_ratios,
                                              height_stride=self.height_stride,
                                              width_stride=self.width_stride)

        # first stage box predictor
        self.setFirstStageBoxPredictor(
            optype=self.first_optype,
            regularizerType=self.first_regOpt['regularizerType'],
            regularizerKeyvals=self.first_regOpt['keyvals'],
            initializerType=self.first_initOpt['initializerType'],
            initializerKeyvals=self.first_initOpt['keyvals'])
        self.setFirstStageNmsIouScoreMaxProposolLossWeights(
            nms_score_thresh=self.nms_score_thresh,
            nms_iou_thresh=self.nms_iou_thresh,
            max_proposal=self.max_proposal,
            local_loss_weight=self.first_localization_loss_weight,
            obj_loss_weight=self.objectness_loss_weight)
        self.setFirstStageCropSizeMaxpoolKernel(crop_size=self.initial_crop_size,
                                                kernel_size=self.maxpool_kernel_size,
                                                kernel_stride=self.maxpool_stride)

        # second stage box predictor
        self.setSecondStageBoxPredictor(
            optype=self.second_optype,
            regularizerType=self.second_regOpt['regularizerType'],
            regularizerKeyvals=self.second_regOpt['keyvals'],
            initializerType=self.second_initOpt['initializerType'],
            initializerKeyvals=self.second_initOpt['keyvals'])

        self.setSecondStageMaskRCNNUseDropOut(val=self.use_dropout)
        self.setSecondStageMaskRCNNDropOutProbability(
            val=self.dropout_keep_probability)

        # second stage post processing
        self.setSecondStagePostProcessing()
        self.setSecondStageBatchNonMaxSuppression()

        # batch_non_max_suppression
        self.setSecondStageBNMS(
            score_threshold=self.bnms_score_threshold,
            iou_threshold=self.bnms_iou_threshold,
            max_detections_per_class=self.bnms_max_detections_per_class,
            total_detections=self.bnms_max_detections_per_class)

        # score converter
        self.setSecondStageScoreConverter(val=self.score_converter)

        # second localization classification
        self.setSecondStageLocalizationLossWeight(
            val=self.second_localization_loss_weight)
        self.setSecondStageClassificationLossWeight(
            val=self.second_classification_loss_weight)

    def setConfig2Model(self):
        "Set additional values and default values"
        self.setDefaultValConfig2Model()

    def setTrainConfigMomentumOptimizer(self, val: float):
        "Set momentum_optimizer_value to momentum_optimizer field"
        keyNames = ['optimizer', self.trainConf.optType,
                    'momentum_optimizer_value']
        self.trainConf.setTrainConfigParamVal(keyNameTree=keyNames,
                                              val=val,
                                              valType=float)

    def setTrainConfigMOMSLInitialLearningRate(self, val: float):
        "Set initial_learning_rate for manual_step_learning_rate"
        keyNames = ['optimizer',
                    self.trainConf.optType,
                    'learning_rate',
                    'manual_step_learning_rate',
                    'initial_learning_rate']
        self.trainConf.setTrainConfigParamVal(keyNameTree=keyNames,
                                              val=val,
                                              valType=float)

    def setTrainConfigMOMSLSchedules(self,
                                     vals: list
                                     ):
        "Set schedules for the manual_step_learning_rate"
        keyNames = ['optimizer',
                    self.trainConf.optType,
                    'learning_rate',
                    'manual_step_learning_rate',
                    'schedule']
        self.trainConf.setTrainConfigParamVal(keyNameTree=keyNames,
                                              val=vals,
                                              valType=list,
                                              contentType=dict)

    def setConfig4TrainConfig(self):
        "set train_config for the config"
        self.trainConf.setTrainConfigBatchSize(val=self.batch_size)
        self.trainConf.setTrainConfigOptUseMovingAverage(
            val=self.use_moving_average)
        self.trainConf.setTrainConfigGradientClipping(
            val=self.gradient_clipping_by_norm)
        self.trainConf.setTrainConfigOptType(optType=self.opt_type)
        self.setTrainConfigMomentumOptimizer(
            val=self.momentum_optimizer_value)
        self.setTrainConfigMOMSLInitialLearningRate(
            val=self.initial_learning_rate)
        self.setTrainConfigMOMSLSchedules(vals=self.schedules)

    def setConfigVals(self, num_classes: int):
        "Set config with default values"
        # num classes
        self.setNumClasses(val=num_classes)
        self.setConfig2Model()
        self.setConfig4TrainConfig()


class ConfigFasterRCNNInceptionResnetV2AtrousModel(ConfigFasterRCNNInceptionV2Model):
    "Generate Config file for faster rcnn"

    def __init__(self,
                 config={}):
        self.modelName = 'faster_rcnn_inception_resnet_v2'
        super().__init__(config=config,
                         modelName=self.modelName)

        # default values
        self.feature_stride = 8
        self.height_stride = 8
        self.width_stride = 8
        self.atrous_rate = 2
        self.initial_crop_size = 17
        self.maxpool_kernel_size = 1
        self.maxpool_stride = 1

    def setFirstStageAtrousRate(self, val: int):
        "First stage atrous rate"
        assertCond(val, val > 0, printType=False)

        self.setModelConfigParam(keyNameTree=['first_stage_atrous_rate'],
                                 val=val,
                                 valType=int)

    def setConfig2Model(self):
        "Set config model params at once"
        self.setDefaultValConfig2Model()
        # atrous rate
        self.setFirstStageAtrousRate(val=self.atrous_rate)

    def setConfigVals(self, num_classes: int):
        "Set config with default values"
        self.setNumClasses(val=num_classes)
        self.setConfig2Model()
        self.setConfig4TrainConfig()


class ConfigFasterRCNNInceptionResnetV2AtrousCosineLRModel(ConfigFasterRCNNInceptionResnetV2AtrousModel):
    def __init__(self,
                 config={}):
        self.modelName = 'faster_rcnn_inception_resnet_v2'
        super().__init__(config=config,
                         modelName=self.modelName)

        # train config default values
        self.train_total_steps = 1200000
        self.warmup_learning_rate = 0.00006,
        self.warmup_steps = 20000
        self.learning_rate_base = 0.006

    def setTrainConfigLRBase(self, val: float):
        "Set learning_rate_base for cosine decay learning rate"
        keyNameTree = ['optimizer',
                       self.trainConf.optType,
                       'cosine_decay_learning_rate',
                       'learning_rate_base']
        self.trainConf.setTrainConfigParamVal(
            keyNameTree=keyNameTree,
            val=val,
            valType=float)

    def setTrainConfigTotalSteps(self, val: int):
        "Set total_steps for train config cosine learning rate"
        keyNameTree = ['optimizer',
                       self.trainConf.optType,
                       'cosine_decay_learning_rate',
                       'total_steps']
        self.trainConf.setTrainConfigParamVal(keyNameTree=keyNameTree,
                                              val=val,
                                              valType=int)

    def setTrainConfigLRWarmup(self, val: float):
        "Set learning_rate_base for cosine decay learning rate"
        keyNameTree = ['optimizer',
                       self.trainConf.optType,
                       'cosine_decay_learning_rate',
                       'warmup_learning_rate']
        self.trainConf.setTrainConfigParamVal(keyNameTree=keyNameTree,
                                              val=val,
                                              valType=float)

    def setTrainConfigWarmupSteps(self, val: int):
        "Set total_steps for train config cosine learning rate"
        keyNameTree = ['optimizer',
                       self.trainConf.optType,
                       'cosine_decay_learning_rate',
                       'warmup_steps']
        self.trainConf.setTrainConfigParamVal(keyNameTree=keyNameTree,
                                              val=val,
                                              valType=int)

    def setConfig4TrainConfig(self):
        "set train_config for the config"
        self.trainConf.setTrainConfigBatchSize(
            val=self.batch_size)
        self.trainConf.setTrainConfigOptUseMovingAverage(
            val=self.use_moving_average)
        self.trainConf.setTrainConfigGradientClipping(
            val=self.gradient_clipping_by_norm)
        self.trainConf.setTrainConfigOptType(optType=self.opt_type)
        self.setTrainConfigMomentumOptimizer(
            val=self.momentum_optimizer_value)
        #
        self.setTrainConfigLRBase(val=self.learning_rate_base)
        self.setTrainConfigTotalSteps(val=self.total_steps)
        self.setTrainConfigLRWarmup(val=self.warmup_learning_rate)
        self.setTrainConfigWarmupSteps(val=self.warmup_steps)


class ConfigMaskRCNNInceptionV2(ConfigFasterRCNNInceptionV2Model):
    "Mask rcnn config file based on faster rcnn inception v2"

    def __init__(self,
                 config={}):
        super().__init__(config=config)
        self.modelName = 'faster_rcnn_inception_v2'
        # default values
        self.aspectMinVal = 800
        self.aspectMaxVal = 1365
        self.number_of_stages = 3
        self.predict_instance_masks = True
        self.mask_height = 15
        self.mask_width = 15
        self.mask_prediction_conv_depth = 0
        self.mask_prediction_num_conv_layers = 2
        self.second_conv_optType = 'CONV'
        self.second_conv_regOpt = {
            'regularizerType': 'l2_regularizer',
            'keyvals': [('weight', 0.0)]}
        self.second_conv_initOpt = {
            'initializerType': 'truncated_normal_initializer',
            'keyvals': [('stddev', 0.01)]}
        self.second_mask_prediction_loss = 4.0
        self.mask_type = 'PNG_MASKS'
        self.load_instance_mask_train = True

    def setNumberOfStages(self, val: int):
        "set number_of_stages"
        self.setModelConfigParam(keyNameTree=['number_of_stages'],
                                 val=val,
                                 valType=int)

    def setPredictMaskInstance(self, val: bool):
        "set predict_instance_masks"
        self.setModelConfigParam(keyNameTree=[
            'second_stage_box_predictor',
            'mask_rcnn_box_predictor',
            'predict_instance_masks'],
            val=val,
            valType=bool)

    def setMaskHeight(self, val: int):
        "set mask height"
        assertCond(val, val > 0, printType=False)
        self.setModelConfigParam(keyNameTree=[
            'second_stage_box_predictor',
            'mask_rcnn_box_predictor',
            'mask_height'
        ],
            val=val,
            valType=int)

    def setMaskWidth(self, val: int):
        "set mask width"
        assertCond(val, val > 0, printType=False)
        self.setModelConfigParam(keyNameTree=[
            'second_stage_box_predictor',
            'mask_rcnn_box_predictor',
            'mask_width'
        ],
            val=val,
            valType=int)

    def setMaskPredictionConvDepth(self, val: int):
        "set mask prediction conv depth"
        self.setModelConfigParam(keyNameTree=[
            'second_stage_box_predictor',
            'mask_rcnn_box_predictor',
            'mask_prediction_conv_depth'
        ],
            val=val,
            valType=int)

    def setMaskPredictionNumConvLayers(self, val: int):
        "set mask prediction conv depth"
        self.setModelConfigParam(keyNameTree=[
            'second_stage_box_predictor',
            'mask_rcnn_box_predictor',
            'mask_prediction_num_conv_layers'
        ],
            val=val,
            valType=int)

    def setSecondStageMaskPredictionLossWeight(self, val: float):
        "set second_stage_mask_prediction_loss_weight"
        self.setModelConfigParam(
            keyNameTree=['second_stage_mask_prediction_loss_weight'],
            val=val,
            valType=float)

    def setSecondConvHyperparamsOptType(self, val: str):
        self.setModelConfigParam(
            keyNameTree=[
                'second_stage_box_predictor',
                'mask_rcnn_box_predictor',
                'conv_hyperparams',
                'op'
            ],
            val=val,
            valType=str)

    def setSecondStageCNNRegularizer(self,
                                     regularizerType: str,
                                     keyvals: list
                                     ):
        "Set regularizer 2 first stage box predictor conv hyperparams"
        assertCond(regularizerType, isinstance(regularizerType, str))
        assertCond(keyvals, isinstance(keyvals, list))
        for keyval in keyvals:
            param = keyval[0]
            assertCond(param, isinstance(param, str))
            val = keyval[1]
            self.setModelConfigParam(keyNameTree=[
                'second_stage_box_predictor',
                'mask_rcnn_box_predictor',
                'conv_hyperparams',
                'regularizer',
                regularizerType,
                param],
                val=val, valType=None, noTypeCheck=True)

    def setSecondStageCNNInitializer(self,
                                     initializerType: str,
                                     keyvals: list):
        "Set first stage cnn initializer"
        assertCond(initializerType, isinstance(initializerType, str))
        assertCond(keyvals, isinstance(keyvals, list))

        for keyval in keyvals:
            param = keyval[0]
            val = keyval[1]
            self.setModelConfigParam(
                keyNameTree=[
                    'second_stage_box_predictor',
                    'mask_rcnn_box_predictor',
                    'conv_hyperparams',
                    'initializer',
                    initializerType,
                    param],
                val=val,
                valType=None,
                noTypeCheck=True)

    def setSecondStageCNNParams(self, optype: str,
                                regularizerType: str,
                                regularizerKeyvals: list,
                                initializerType: str,
                                initializerKeyvals: list):
        "Second stage cnn parameters"
        self.setSecondConvHyperparamsOptType(val=optype)
        self.setSecondStageCNNRegularizer(regularizerType=regularizerType,
                                          keyvals=regularizerKeyvals)
        self.setSecondStageCNNInitializer(initializerType=initializerType,
                                          keyvals=initializerKeyvals)

    def setMaskType2EvalDict(self, evalDict: dict,
                             mask_type: str):
        "Set mask type to eval dict"
        assertCond(mask_type, isinstance(mask_type, str))
        assertCond(evalDict, isinstance(evalDict, dict))
        evalDict['mask_type'] = mask_type
        return evalDict

    def setMaskType2TrainInput(self, mask_type: str):
        "set masktype 2 train_input_reader"
        self.setConfigParamVal(keyNames=['train_input_reader',
                                         'mask_type'],
                               val=mask_type,
                               valType=str)

    def setLoadInstanceMask2TrainInput(self, isLoad: bool):
        "set load_instance_masks to train_input_reader"
        self.setConfigParamVal(keyNames=['train_input_reader',
                                         'load_instance_masks'],
                               val=isLoad,
                               valType=bool)

    def setLoadInstanceMask2EvalDict(self,
                                     evalDict: dict,
                                     isLoad: bool):
        "set load_instance_masks to eval dict"
        assertCond(isLoad, isinstance(isLoad, bool))
        assertCond(evalDict, isinstance(evalDict, dict))
        evalDict['load_instance_masks'] = isLoad
        return evalDict

    def setEvalInputEvalDict(self,
                             labelMapPath: str,
                             inputPathList: [str],
                             isShuffle=False,
                             numReaders=1,
                             isLoad=True,
                             mask_type='PNG_MASKS'):
        "Overrides configmodel method"
        eDict = {}
        eDict = self.setEvalLabelMapPath2EvalDict(evalDict=eDict,
                                                  path=labelMapPath)
        eDict = self.setEvalNumReaders2EvalDict(evalDict=eDict,
                                                val=numReaders)
        eDict = self.setEvalShuffle2EvalDict(evalDict=eDict,
                                             isShuffle=isShuffle)
        eDict = self.setEvalTFRecordPath2EvalDict(evalDict=eDict,
                                                  paths=inputPathList)
        eDict = self.setLoadInstanceMask2EvalDict(evalDict=eDict,
                                                  isLoad=isLoad)
        eDict = self.setMaskType2EvalDict(evalDict=eDict,
                                          mask_type=mask_type)
        return eDict

    def setDefaultValConfig2Model(self):
        "Set config params"

        self.setMaskType2TrainInput(mask_type=self.mask_type)
        self.setLoadInstanceMask2TrainInput(
            isLoad=self.load_instance_mask_train)

        # image resizer
        self.setImageResizer()
        self.setKeepAspectRatioResizer(
            minval=self.aspectMinVal,
            maxval=self.aspectMaxVal)

        # number of stages
        self.setNumberOfStages(
            val=self.number_of_stages)
        # feature extractor
        self.setFeatureExtractor()
        self.setFirstStageFeatureStride(
            val=self.feature_stride)

        # anchor generator
        self.setFirstStageGridAnchorGenerator(
            scales=self.scales,
            aspect_ratios=self.aspect_ratios,
            height_stride=self.height_stride,
            width_stride=self.width_stride)

        # first stage box predictor
        self.setFirstStageBoxPredictor(
            optype=self.first_optype,
            regularizerType=self.first_regOpt['regularizerType'],
            regularizerKeyvals=self.first_regOpt['keyvals'],
            initializerType=self.first_initOpt['initializerType'],
            initializerKeyvals=self.first_initOpt['keyvals'])

        self.setFirstStageNmsIouScoreMaxProposolLossWeights(
            nms_score_thresh=self.nms_score_thresh,
            nms_iou_thresh=self.nms_iou_thresh,
            max_proposal=self.max_proposal,
            local_loss_weight=self.first_localization_loss_weight,
            obj_loss_weight=self.objectness_loss_weight)

        self.setFirstStageCropSizeMaxpoolKernel(crop_size=self.initial_crop_size,
                                                kernel_size=self.maxpool_kernel_size,
                                                kernel_stride=self.maxpool_stride)

        # second stage box predictor
        # mask box
        self.setPredictMaskInstance(val=self.predict_instance_masks)
        self.setMaskHeight(val=self.mask_height)
        self.setMaskWidth(val=self.mask_width)
        self.setMaskPredictionConvDepth(val=self.mask_prediction_conv_depth)
        self.setMaskPredictionConvDepth(
            val=self.mask_prediction_num_conv_layers)

        # second stage box predictor
        self.setSecondStageBoxPredictor(
            optype=self.second_optype,
            regularizerType=self.second_regOpt['regularizerType'],
            regularizerKeyvals=self.second_regOpt['keyvals'],
            initializerType=self.second_initOpt['initializerType'],
            initializerKeyvals=self.second_initOpt['keyvals'])

        self.setSecondStageMaskRCNNUseDropOut(val=self.use_dropout)
        self.setSecondStageMaskRCNNDropOutProbability(
            val=self.dropout_keep_probability)

        # second stage conv hyperparams
        self.setSecondStageCNNParams(
            optype=self.second_conv_optType,
            regularizerType=self.second_conv_regOpt['regularizerType'],
            regularizerKeyvals=self.second_conv_regOpt['keyvals'],
            initializerType=self.second_conv_initOpt['initializerType'],
            initializerKeyvals=self.second_conv_initOpt['keyvals'])

        # second stage post processing
        self.setSecondStagePostProcessing()
        self.setSecondStageBatchNonMaxSuppression()

        # batch_non_max_suppression
        self.setSecondStageBNMS(
            score_threshold=self.bnms_score_threshold,
            iou_threshold=self.bnms_iou_threshold,
            max_detections_per_class=self.bnms_max_detections_per_class,
            total_detections=self.bnms_max_detections_per_class)

        # score converter
        self.setSecondStageScoreConverter(val=self.score_converter)

        # second localization classification
        self.setSecondStageLocalizationLossWeight(
            val=self.second_localization_loss_weight)
        self.setSecondStageClassificationLossWeight(
            val=self.second_classification_loss_weight)
        self.setSecondStageMaskPredictionLossWeight(
            val=self.second_mask_prediction_loss)

    def setConfigVals(self, num_classes: int):
        "Set config with default values"
        self.setNumClasses(num_classes)
        self.setConfig2Model()
        self.setConfig4TrainConfig()


# Utility functions

def getLabelDname(bnames: [str]):
    print("Welcome to config script of ancient-cv")
    print("Please provide the name of the label space you would configure")
    print('Available ones are: ')
    for bname in bnames:
        print('\n', bname)
    labelDname = input('Enter your choice: ')
    if labelDname not in bnames:
        raise ValueError("Your input: {0} is not among available "
                         "label spaces: {1}".format(labelDname, bnames)
                         )
        sys.exit(1)  # failure
    return labelDname


def parseParamsFromInput(str1: str):
    "Parse parameters to set to model config"
    parsedKeyvals = str1.split(';')
    parsedKeyvals = [p.strip() for p in parsedKeyvals]
    keyvals = []
    for p in parsedKeyvals:
        el = p.split(',')
        el = [e.strip() for e in el]
        el = tuple(el)
        keyvals.append(el)
    #
    return keyvals


def convertStr2Type(str1: str):
    "Convert string to appropriate type"
    assert isinstance(str1, str)
    if str1.isdigit():
        return int(str1)
    elif '.' in str1:
        try:
            return float(str1)
        except ValueError:
            pass
    else:
        return str1


def setParams2Config(confObj, params: list):
    "Set params to config file"
    if params == []:
        return confObj
    #
    attrs = confObj.__dict__
    for param in params:
        paramKey = param[0]
        paramVal = convertStr2Type(param[1])
        if paramKey in attrs:
            confObj.__setattr__(paramKey, paramVal)
        else:
            print('\n{0} is not a valid attribute for this config file'.format(
                paramKey))
            print('Skipping it')
    #
    return confObj


def makeLabelModelPaths(dirname: str, output_name: str):
    "make label model path"
    if os.path.isdir(dirname) is False:
        os.mkdir(dirname)
        labelModelConfDir = os.path.join(dirname, output_name)
        labelModelSessionDir = os.path.join(labelModelConfDir, 'tfsession')
        modelVisualization = os.path.join(labelModelSessionDir, 'visu')
        os.mkdir(labelModelConfDir)
        os.mkdir(labelModelSessionDir)
        os.mkdir(modelVisualization)
    else:
        labelModelConfDir = os.path.join(dirname, output_name)
        labelModelSessionDir = os.path.join(labelModelConfDir, 'tfsession')
        modelVisualization = os.path.join(labelModelSessionDir, 'visu')
        os.mkdir(labelModelConfDir)
        os.mkdir(labelModelSessionDir)
        os.mkdir(modelVisualization)
    #
    return labelModelConfDir, labelModelSessionDir, modelVisualization


def makeOutputName():
    "create an output name for config file"
    dtime = str(datetime.datetime.now().replace(microsecond=0).isoformat())
    output_name = 'conf-time--' + dtime
    output_name = output_name + '--id--' + str(uuid.uuid4())
    return output_name


def getNbClassesFromLabelMap(labelMapPath: str) -> int:
    "get number of classes from label map"
    with open(labelMapPath, 'r') as f:
        labelFile = f.read()
    tfile = text_format.Parse(labelFile, StringIntLabelMap())
    jdict = json_format.MessageToDict(tfile)
    nb_classes = len(jdict['item'])
    return nb_classes


def getUserParams(confObj):
    "Get user params"
    print('Here is the default settings of your model: ')
    attrs = confObj.__dict__
    for key, val in attrs.items():
        print('\n', 'attribute name: ', key)
        print('attribute value: ', str(val))
        print('attribute type: ', type(val))

    #
    print('You can provide new values to attributes with following types: ')
    print('float, str, int, bool') 
    strinput = input(
        'Would you like to provide new values [y/n, n by default]?: '
    )
    if strinput == 'y':
        print(
            'Please enter your new values as key value pairs as in following: '
        )
        print('\nParameter Name 1, new value; Parameter Name 2, new value')
        params = input('Please enter your pairs as shown above: ')
        params = parseParamsFromInput(params)
        confObj = setParams2Config(confObj, params)

    print('Here is the current settings of your model: ')
    attrs = confObj.__dict__
    for key, val in attrs.items():
        print('\n', 'attribute name: ', key)
        print('attribute value: ', str(val))
        print('attribute type: ', type(val))
    #
    strinput = input(
        'Would you like to continue with these parameters [y/n, y by default]: '
    )
    if strinput == 'n':
        print('You have chosen to stop the execution of current script')
        print('Bye bye ...')
        sys.exit(0)
    return confObj


def useCustomConfig(labelDname: str,
                    modelName: str,
                    currentdir: str):
    "Use custom config given by user"
    strinput = input(
        'Would you like to use your own config file [y/n n by default]? '
    )
    if strinput == 'y':
        output_name = makeOutputName()
        modelsDir = os.path.join(currentdir, 'models')
        modelLabelDir = os.path.join(modelsDir, labelDname)
        modelLabelModelDir = os.path.join(modelLabelDir, modelName)
        (labelModelConfDir, labelModelSessionDir,
         modelVisualization) = makeLabelModelPaths(modelLabelModelDir,
                                                   output_name)


def main():
    "main access point of the function"
    available_models = {
        'faster_rcnn_inception_resnet_v2_atrous_cosine_lr': ConfigFasterRCNNInceptionResnetV2AtrousCosineLRModel,
        'faster_rcnn_inception_resnet_v2_atrous': ConfigFasterRCNNInceptionResnetV2AtrousModel,
        'faster_rcnn_inception_v2': ConfigFasterRCNNInceptionV2Model,
        'mask_rcnn_inception_v2': ConfigMaskRCNNInceptionV2
    }
    currentdir = os.getcwd()
    datadir = os.path.join(currentdir, 'data')
    dirregex = '[a-zA-Z]*'
    datadirContents = os.path.join(datadir, dirregex)
    datadirContents = os.path.join(datadirContents, '')
    folderNames = glob.glob(datadirContents)
    bnames = [os.path.basename(os.path.dirname(f)) for f in folderNames]
    labelDname = getLabelDname(bnames)
    dataLabelDir = os.path.join(datadir, labelDname)
    modelsDir = os.path.join(currentdir, 'models')
    modelLabelDir = os.path.join(modelsDir, labelDname)
    labelMapPath = os.path.join(dataLabelDir, labelDname+'.pbtxt')
    evalDir = os.path.join(dataLabelDir, 'eval')

    trainDir = os.path.join(dataLabelDir, 'train')
    strinput = input("Do your tfrecords are seperated into shards?"
                     "Currently only single tfrecords are supported [y/n]: ")
    if strinput == 'n':
        print("Unfortunately currently only single record for train and eval"
              " are supported. But that will change in near future."
              " Feel free to reload your annotations as a single record for"
              " train and eval using load.py")
    trainRecordPath = os.path.join(trainDir, 'train.record')
    evalRecordPath = os.path.join(evalDir, 'eval.record')
    recordCount = getRecordNbFromTFRecord(trainRecordPath)
    evRecordCount = getRecordNbFromTFRecord(evalRecordPath)
    recordTotal = recordCount + evRecordCount
    pipeConf = ConfigModel()
    pipeConf.setConfigKeys()
    pipeConf.setTrainTFRecordPath(path=trainRecordPath)
    pipeConf.setTrainLabelMapPath(path=labelMapPath)
    pipeConf.setTrainDataAugmentationOpts()
    pipeConf.setEvalNumExamples(val=evRecordCount)
    pipeConf.setEvalMetricsSet(vals=['pascal_voc_detection_metrics'])
    pipeConf.setMaxEvals()
    strinput = input("Enter number of training steps [default 10000]: ")
    if strinput == "":
        stepnb = 10000
    else:
        stepnb = int(strinput)
    pipeConf.setTrainNumSteps(val=stepnb)
    pipeConf.setTrainDetectionCkpt(False)
    strinput = input("choose a meta model. Available ones are: {0}. ".format(
        pipeConf.AVAILABLE_META_MODELS))
    pipeConf.setMetaModel(val=strinput)
    print('Please choose a model.')
    print('Available ones are: ')
    for amod in available_models.keys():
        print('\n', amod)
    strinput = input("Enter the model of your choice: ")
    if strinput not in available_models:
        print("your input: {0} is not among available_models: {1}".format(
            strinput, available_models))
        sys.exit(1)
    #
    modelLabelDir = os.path.join(modelsDir, labelDname)
    modelName = strinput

    config = pipeConf.config
    modelConf = available_models[modelName](config)
    # input reader of eval is set from modelConf in order to
    # adapt to changes due to mask use.
    edict = modelConf.setEvalInputEvalDict(labelMapPath=labelMapPath,
                                           inputPathList=[evalRecordPath])
    modelConf.setEvalInputReader(evalDicts=[edict])
    #
    output_name = makeOutputName()
    modelLabelModelDir = os.path.join(modelLabelDir, modelName)
    #
    (labelModelConfDir, labelModelSessionDir,
     modelVisualization) = makeLabelModelPaths(modelLabelModelDir, output_name)
    modelConf.setEvalVisualizationsExportPath(path=modelVisualization)
    nb_classes = getNbClassesFromLabelMap(labelMapPath)
    modelConf = getUserParams(modelConf)
    modelConf.setConfigVals(num_classes=nb_classes)

    confWriter = ConfigRW(ipath='',
                          odir=labelModelConfDir,
                          config=modelConf.config,
                          oname=output_name,
                          messageProto=PipelineConfig()
                          )
    confWriter.writeConfig()
    print('config file is written please proceed to training...')


if __name__ == '__main__':
    main()
