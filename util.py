import os
import cv2
import config as cfg


# Get a list of the training, validation, and testing file paths
def prepare_data(dataset_dir):
    train_input_names=[]
    train_output_names=[]
    test_input_names=[]
    test_output_names=[]
    val_input_names=[]
    val_output_names=[]
    #遍历文件并添加数据
    for file in os.listdir(dataset_dir + "/train"):
        cwd = os.getcwd()#返回当前的工作目录 current working directory
        train_input_names.append(cwd + "/" + dataset_dir + "/train/" + file)
    for file in os.listdir(dataset_dir + "/train_labels"):
        cwd = os.getcwd()
        train_output_names.append(cwd + "/" + dataset_dir + "/train_labels/" + file)
    for file in os.listdir(dataset_dir + "/val"):
        cwd = os.getcwd()#返回当前的工作目录 current working directory
        val_input_names.append(cwd + "/" + dataset_dir + "/val/" + file)
    for file in os.listdir(dataset_dir + "/val_labels"):
        cwd = os.getcwd()
        val_output_names.append(cwd + "/" + dataset_dir + "/val_labels/" + file)
    for file in os.listdir(dataset_dir + "/test"):
        cwd = os.getcwd()
        test_input_names.append(cwd + "/" + dataset_dir + "/test/" + file)
    for file in os.listdir(dataset_dir + "/test_labels"):
        cwd = os.getcwd()
        test_output_names.append(cwd + "/" + dataset_dir + "/test_labels/" + file)
    train_input_names.sort(), train_output_names.sort(), val_input_names.sort(), val_output_names.sort(), test_input_names.sort(), test_output_names.sort()#字母表排序
    return train_input_names, train_output_names, val_input_names, val_output_names, test_input_names, test_output_names

#图像加载和颜色空间的转换
def load_image(path, is_data=True):
    image = cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_BGR2RGB)
    #print(image.shape)
    if is_data is True:
        image = cv2.resize(image, cfg.IMAGE_SIZE)
    else:
        image = cv2.resize(image, (240, 240))
    #print(image.shape)
    return image


