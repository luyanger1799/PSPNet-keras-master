'''
PSPNet　的相关配置属性
'''

BATCH_SIZE = 3
NUM_EPOCHS = 50
VERBOSE = 1

DATASET_DIR = "CamVid"
MODEL_PATH = "model/model.h5"
WEIGHTS_PATH = "model/weights.h5"
CHECKPOINT_DIR = "checkpoint/"
PREDICTION_DIR = "predition/"
TEST_IMAGE_PATH = "prediction/Seq05VD_f03270.png"

OPTIMIZER = 'nadam'
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']

INPUT_SHAPE = (480, 480, 3)
OUTPUT_SHAPE = (240, 240, 3)
IMAGE_SIZE = (480, 480)