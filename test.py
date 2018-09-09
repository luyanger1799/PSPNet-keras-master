import util
import config as cfg
import tensorflow as tf
from keras import backend as K
import numpy as np
import helpers
import os
from PSPNet import PSPNet


from keras.models import Model, load_model

K.tensorflow_backend._get_available_gpus()

# this needs to get generalized
class_names_list, label_values = helpers.get_label_info(os.path.join("CamVid", "class_dict.csv"))

num_classes = len(label_values)

#get data name list
train_input_names, train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = util.prepare_data(cfg.DATASET_DIR)

test_data = []
test_labels = []

print("Loading test data ...")
for img_name in test_input_names:
    input_image = util.load_image(img_name)
    with tf.device('/cpu:0'):
        input_image = np.float32(input_image) / 255.0

        test_data.append(input_image)
        print(img_name)


for labels_name in test_output_names:
    output_image = util.load_image(labels_name, is_data=False)
    with tf.device('/cpu:0'):
        output_image = np.float32(helpers.one_hot_it(label=output_image, label_values=label_values))

        test_labels.append(output_image)
        print(labels_name)

test_data = np.array(test_data)
test_labels = np.array(test_labels)

print("Finish loading the data. ")


print("Start testing...")

pspnet = PSPNet()
pspnet.test(x_test=test_data, y_test=test_labels)

#test_result = pspnet.model.evaluate(x=test_data, y=test_labels, batch_size=cfg.BATCH_SIZE, verbose=cfg.VERBOSE)
#print("loss = {:.2f}  accuracy = {:.2f}".format(test_result[0], test_result[1]))

#model = UNet.load_model(cfg.MODEL_PATH)






