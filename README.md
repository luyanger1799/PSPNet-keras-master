# PSPNet-keras-master
PSPNet的tensorflow + keras实现<br>
论文来源于CVPR2017  - 金字塔场景解析网<br>
采用ResNet50 + 金字塔池化<br>
数据集为CamVid街景数据集<br>
训练直接python train.py<br>
相关配置参数在config.py文件<br>
数据集构成：<br>
dataset:<br>
    --train<br>
    --train_labels<br>
    --val<br>
    --val_labels<br>
    --test<br>
    --test_labels<br>
