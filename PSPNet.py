from keras.layers import Conv2D, Concatenate, Input, Dropout, ZeroPadding2D,\
    AveragePooling2D, BatchNormalization, Activation, Add, UpSampling2D, MaxPooling2D
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.initializers import glorot_uniform
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import config as cfg

class PSPNet(object):

    #初始化
    def __init__(self):
        self.BATCH_SIZE = cfg.BATCH_SIZE
        self.VERBOSE = cfg.VERBOSE
        self.NUM_EPOCHS = cfg.NUM_EPOCHS
        self.LOSS = cfg.LOSS
        self.OPTIMIZER = cfg.OPTIMIZER
        self.METRICS = cfg.METRICS
        self.INPUT_SHAPE = cfg.INPUT_SHAPE
        self.model = None
        self.history = None

    def identity_block(self, x, f_kernel_size, filters, dilation, pad):
        filters_1, filters_2, filters_3 = filters
        x_shortcut = x

        # stage 1
        x = Conv2D(filters=filters_1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                   kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization(momentum=0.95, axis=-1)(x)
        x = Activation(activation='relu')(x)

        # stage 2
        x = ZeroPadding2D(padding=pad)(x)
        x = Conv2D(filters=filters_2, kernel_size=f_kernel_size, strides=(1, 1),
                   dilation_rate=dilation, kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization(momentum=0.95, axis=-1)(x)
        x = Activation(activation='relu')(x)

        # stage 3
        x = Conv2D(filters=filters_3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                   kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization(momentum=0.95, axis=-1)(x)

        # stage 4
        x = Add()([x, x_shortcut])
        x = Activation(activation='relu')(x)
        return x

    def convolutional_block(self, x, f_kernel_size, filters, strides, dilation, pad):
        filters_1, filters_2, filters_3 = filters
        x_shortcut = x

        # stage 1
        x = Conv2D(filters=filters_1, kernel_size=(1, 1), strides=strides, padding='valid',
                   kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization(momentum=0.95, axis=-1)(x)
        x = Activation(activation='relu')(x)

        # stage 2
        x = ZeroPadding2D(padding=pad)(x)
        x = Conv2D(filters=filters_2, kernel_size=f_kernel_size, strides=(1, 1), dilation_rate=dilation,
                   kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization(momentum=0.95, axis=-1)(x)
        x = Activation(activation='relu')(x)

        # stage 3
        x = Conv2D(filters=filters_3, kernel_size=(1, 1), strides=(1, 1),
                   kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization(momentum=0.95, axis=-1)(x)
        x = Activation(activation='relu')(x)

        # stage 4
        x_shortcut = Conv2D(filters=filters_3, kernel_size=(1, 1), strides=strides, padding='valid',
                            kernel_initializer=glorot_uniform(seed=0))(x_shortcut)
        x_shortcut = BatchNormalization(momentum=0.95, axis=-1)(x_shortcut)

        # stage 5
        x = Add()([x, x_shortcut])
        x = Activation(activation='relu')(x)
        return x

    def ResNet50(self, inputs):
        # inputs = Input(shape=(224, 224, 3))

        # stage 1
        #conv1_1_
        x = ZeroPadding2D(padding=(1, 1))(inputs)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization(momentum=0.95, axis=-1)(x)
        x = Activation(activation='relu')(x)

        #conv1_2
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization(momentum=0.95, axis=-1)(x)
        x = Activation(activation='relu')(x)

        # conv1_3
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization(momentum=0.95, axis=-1)(x)
        x = Activation(activation='relu')(x)

        #pool1
        x = ZeroPadding2D(padding=(1, 1))(x)
        x_stage_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        # x_stage_1 = Dropout(0.25)(x_stage_1)

        # stage 2
        x = self.convolutional_block(x_stage_1, f_kernel_size=(3, 3), filters=[64, 64, 256], strides=1, pad=(1, 1), dilation=1)
        x = self.identity_block(x, f_kernel_size=(3, 3), filters=[64, 64, 256], pad=(1, 1), dilation=1)
        x_stage_2 = self.identity_block(x, f_kernel_size=(3, 3), filters=[64, 64, 256], pad=(1, 1), dilation=1)
        # x_stage_2 = Dropout(0.25)(x_stage_2)

        # stage 3
        x = self.convolutional_block(x_stage_2, f_kernel_size=(3, 3), filters=[128, 128, 512], strides=2, pad=(1, 1), dilation=1)
        x = self.identity_block(x, f_kernel_size=(3, 3), filters=[128, 128, 512], pad=(1, 1), dilation=1)
        x = self.identity_block(x, f_kernel_size=(3, 3), filters=[128, 128, 512], pad=(1, 1), dilation=1)
        x_stage_3 = self.identity_block(x, f_kernel_size=(3, 3), filters=[128, 128, 512], pad=(1, 1), dilation=1)
        # x_stage_3 = Dropout(0.25)(x_stage_3)

        # stage 4
        x = self.convolutional_block(x_stage_3, f_kernel_size=(3, 3), filters=[256, 256, 1024], strides=1, pad=(2, 2), dilation=2)
        x = self.identity_block(x, f_kernel_size=(3, 3), filters=[256, 256, 1024], pad=(2, 2), dilation=2)
        x = self.identity_block(x, f_kernel_size=(3, 3), filters=[256, 256, 1024], pad=(2, 2), dilation=2)
        x = self.identity_block(x, f_kernel_size=(3, 3), filters=[256, 256, 1024], pad=(2, 2), dilation=2)
        x = self.identity_block(x, f_kernel_size=(3, 3), filters=[256, 256, 1024], pad=(2, 2), dilation=2)
        x_stage_4 = self.identity_block(x, f_kernel_size=(3, 3), filters=[256, 256, 1024], pad=(2, 2), dilation=2)
        # x_stage_4 = Dropout(0.25)(x_stage_4)

        # stage 5
        x = self.convolutional_block(x_stage_4, f_kernel_size=(3, 3), filters=[512, 512, 2048], strides=1, pad=(4, 4), dilation=4)
        x = self.identity_block(x, f_kernel_size=(3, 3), filters=[256, 256, 2048], pad=(4, 4), dilation=4)
        x_stage_5 = self.identity_block(x, f_kernel_size=(3, 3), filters=[256, 256, 2048], pad=(4, 4), dilation=4)
        # x_stage_5 = Dropout(0.25)(x_stage_5)

        return x_stage_5


    #构建网络结构
    def build_pspnet(self, num_classes):

        #ResNet50 提取特征
        inputs = Input(shape=self.INPUT_SHAPE)

        res_features = self.ResNet50(inputs)

        #金字塔池化
        x_c1 = AveragePooling2D(pool_size=60, strides=60, name='ave_c1')(res_features)
        x_c1 = Conv2D(filters=512, kernel_size=1, strides=1, padding='same', name='conv_c1')(x_c1)
        x_c1 = BatchNormalization(momentum=0.95, axis=-1)(x_c1)
        x_c1 = Activation(activation='relu')(x_c1)
        #x_c1 = Dropout(0.2)(x_c1)
        x_c1 = UpSampling2D(size=(60, 60), name='up_c1')(x_c1)

        x_c2 = AveragePooling2D(pool_size=30, strides=30, name='ave_c2')(res_features)
        x_c2 = Conv2D(filters=512, kernel_size=1, strides=1, padding='same', name='conv_c2')(x_c2)
        x_c2 = BatchNormalization(momentum=0.95, axis=-1)(x_c2)
        x_c2 = Activation(activation='relu')(x_c2)
        #x_c2 = Dropout(0.2)(x_c2)
        x_c2 = UpSampling2D(size=(30, 30), name='up_c2')(x_c2)

        x_c3 = AveragePooling2D(pool_size=20, strides=20, name='ave_c3')(res_features)
        x_c3 = Conv2D(filters=512, kernel_size=1, strides=1, padding='same', name='conv_c3')(x_c3)
        x_c3 = BatchNormalization(momentum=0.95, axis=-1)(x_c3)
        x_c3 = Activation(activation='relu')(x_c3)
        #x_c3 = Dropout(0.2)(x_c3)
        x_c3 = UpSampling2D(size=(20, 20), name='up_c3')(x_c3)

        x_c4 = AveragePooling2D(pool_size=10, strides=10, name='ave_c4')(res_features)
        x_c4 = Conv2D(filters=512, kernel_size=1, strides=1, padding='same', name='conv_c4')(x_c4)
        x_c4 = BatchNormalization(momentum=0.95, axis=-1)(x_c4)
        x_c4 = Activation(activation='relu')(x_c4)
        #x_c4 = Dropout(0.2)(x_c4)
        x_c4 = UpSampling2D(size=(10, 10), name='up_c4')(x_c4)

        x_c5 = Conv2D(filters=512, kernel_size=1, strides=1, name='conv_c5', padding='same')(res_features)
        x_c5 = BatchNormalization(momentum=0.95, axis=-1)(x_c5)
        x_c5 = Activation(activation='relu')(x_c5)
        #x_c5 = Dropout(0.2)(x_c5)

        x = Concatenate(axis=-1, name='concat')([x_c1, x_c2, x_c3, x_c4, x_c5])
        x = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', name='sum_conv_1_11')(x)
        x = BatchNormalization(momentum=0.95, axis=-1)(x)
        x = Activation(activation='relu')(x)

        x = UpSampling2D(size=(4, 4))(x)
        # x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', name='sum_conv_1_21')(x)
        # x = BatchNormalization(momentum=0.95, axis=-1)(x)
        # x = Activation(activation='relu')(x)

        outputs = Conv2D(filters=num_classes, kernel_size=1, strides=1, padding='same', name='sum_conv_2', activation='softmax')(x)


        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=self.OPTIMIZER, loss=self.LOSS, metrics=self.METRICS)



    #训练
    def train(self, x_train, y_train, x_val, y_val):
        train_generator = ImageDataGenerator()
        val_generator = ImageDataGenerator()
        checkpoint = ModelCheckpoint(filepath=cfg.CHECKPOINT_DIR + "model-{epoch:02d}.h5", monitor='val_acc', save_best_only=True)
        self.model.fit_generator(train_generator.flow(x=x_train, y=y_train, batch_size=self.BATCH_SIZE),
                                 steps_per_epoch=len(x_train)//self.BATCH_SIZE, epochs=self.NUM_EPOCHS,
                                 validation_data=val_generator.flow(x=x_val, y=y_val, batch_size=self.BATCH_SIZE), callbacks=[checkpoint])
    #保存模型和权重
    def save_model(self):
        # json_string = self.model.to_json()
        # open(cfg.MODEL_PATH, 'w').write(json_string)
        # self.model.save_weights(cfg.WEIGHTS_PATH, overwrite=True)
        self.model.save(cfg.MODEL_PATH)

    #模型加载
    def load_model(self):
        # self.model = model_from_json(open(cfg.MODEL_PATH).read())
        # self.model.load_weights(cfg.WEIGHTS_PATH)
        self.model = load_model(cfg.MODEL_PATH)
        self.model.compile(optimizer=self.OPTIMIZER, loss=self.LOSS, metrics=self.METRICS)

    #模型训练
    def continue_train(self, x_train, y_train, x_val, y_val):

        self.load_model()
        train_generator = ImageDataGenerator()
        val_generator = ImageDataGenerator()
        checkpoint = ModelCheckpoint(filepath=cfg.CHECKPOINT_DIR + "model-{epoch:02d}.h5", monitor='val_acc', save_best_only=True)
        self.model.fit_generator(train_generator.flow(x=x_train, y=y_train, batch_size=self.BATCH_SIZE),
                                 steps_per_epoch=len(x_train) // self.BATCH_SIZE, epochs=self.NUM_EPOCHS,
                                 validation_data=val_generator.flow(x=x_val, y=y_val, batch_size=self.BATCH_SIZE), callbacks=[checkpoint])
    #测试
    def test(self, x_test, y_test):
        self.load_model()
        test_generator = ImageDataGenerator()

        test_result = self.model.evaluate_generator(generator=test_generator.flow(x=x_test, y=y_test, batch_size=self.BATCH_SIZE),
                                                    steps=len(x_test)//self.BATCH_SIZE)
        print("loss = {:.2f}  accuracy = {:.2f}".format(test_result[0], test_result[1]))

    #预测
    def predict(self, x_predict):
        self.load_model()
        # predict_generator = ImageDataGenerator()
        # predict_batch = self.model.predict_generator(generator=predict_generator.flow(x=x_predict, batch_size=1), steps=len(x_predict))
        # return predict_batch
        #return self.model.predict_on_batch(x_predict)
        return self.model.predict(x_predict)