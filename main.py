import os
import cv2
import numpy as np
import keras.backend as K

from keras.models import Model
from keras.optimizers import Nadam, SGD
from keras.utils import multi_gpu_model, plot_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import Conv2D, PReLU, Conv2DTranspose, add, concatenate, Input, Dropout, BatchNormalization, Activation

########################### data preprocessing ###########################

def data_prepare():
    """将图像分割用的图片从JPEGImages文件夹提取到SegTrainImg和SegValImg里"""
    dataset_path = r'D:My_project\deeplearning\datasets\warwick_qu_dataset'
    imgpath = r'D:\My_project\deeplearning\datasets\warwick_qu_dataset\Warwick_QU_Dataset'

    trainimg_savepath = os.path.join(dataset_path,'SegTrainImg')
    trainmask_savepath = os.path.join(dataset_path, 'SegTrainMaskImg')
    valimg_savepath = os.path.join(dataset_path,'SegValImg')
    valmask_savepath = os.path.join(dataset_path, 'SegValMaskImg')
    testimg_savepath = os.path.join(dataset_path,'SegtestImg')
    testmask_savepath = os.path.join(dataset_path, 'SegTestMask')

    filelist = os.listdir(imgpath)
    for file in filelist:
        if 'testA' in file:
            img = cv2.imread(os.path.join(imgpath,file))
            savepath = os.path.join(valimg_savepath,file) if 'anno' not in file else os.path.join(valmask_savepath,file)
            cv2.imwrite(savepath,img)
        elif 'testB' in file:
            img = cv2.imread(os.path.join(imgpath, file))
            savepath = os.path.join(testimg_savepath, file) if 'anno' not in file else os.path.join(testmask_savepath,file)
            cv2.imwrite(savepath, img)
        elif 'train' in file:
            img = cv2.imread(os.path.join(imgpath, file))
            savepath = os.path.join(trainimg_savepath, file) if 'anno' not in file else os.path.join(trainmask_savepath,file)
            cv2.imwrite(savepath, img)
    print("Done")

def mask2label(mask):
    """
    将原始的mask图像转换为label数据格式，即按类别顺序分为0,1,2,...
    :param mask: 原始mask图像
    :return: 转换后的label
    """
    mask[mask>0] = 1
    mask = mask[:,:,:,0]
    mask = mask[:,:,:,np.newaxis]
    return mask

########################### data generator ###########################

def train_generator(img_size, batch_size, gen_arg_dict={}, seed=1):
    """
    训练数据生成器
    :param img_size: 生成的目标图片尺寸
    :param batch_size: 批量大小
    :param gen_arg_dict: 数据增广参数
    :param seed: 随机抽样的随机种子
    :return: 标准化后的图像以及转换为类别序号的mask
    """
    train_datagen = ImageDataGenerator(**gen_arg_dict)  # 设置数据增广实例
    gen_arg_dict['rescale'] = 1  # mask不需要标准化，方便后续处理
    mask_datagen = ImageDataGenerator(**gen_arg_dict)  # mask的数据增广应当和原始图片一模一样
    # 加载数据发生器，注意，classes是存放img/mask数据的文件夹名称，且img/mask两者的随机种子seed和batchsize要保持一致
    img_generator = train_datagen.flow_from_directory(DATA_PATH,classes=['SegTrainImg'],target_size=img_size,seed=seed,batch_size=batch_size)
    mask_generator = mask_datagen.flow_from_directory(DATA_PATH,classes=['SegTrainMaskImg'],target_size=img_size,seed=seed,batch_size=batch_size)
    train_generator = zip(img_generator,mask_generator)  # 将图片数据和mask数据整合
    for(img, mask) in train_generator:
        img = img[0]
        mask = mask2label(mask[0])
        yield (img,mask)

def val_generator(img_size, batch_size=1):
    val_datagen = ImageDataGenerator(rescale=1./255)
    img_generator = val_datagen.flow_from_directory(DATA_PATH,classes=['SegValImg'],target_size=img_size,batch_size=batch_size,seed=1)
    mask_datagen = ImageDataGenerator()
    mask_generator = mask_datagen.flow_from_directory(DATA_PATH, classes=['SegValMaskImg'], target_size=img_size,batch_size=batch_size, seed=1)
    val_generator = zip(img_generator, mask_generator)  # 将图片数据和mask数据整合
    for (img,mask) in val_generator:
        img = img[0]
        mask = mask2label(mask[0])
        yield (img,mask)

def test_generator(img_size, batch_size=1):
    test_datagen = ImageDataGenerator(rescale=1./255)
    img_generator = test_datagen.flow_from_directory(DATA_PATH,classes=['SegtestImg'],target_size=img_size,batch_size=batch_size)

    for img in img_generator:
        img = img[0]
        yield img

########################### loss functions ###########################
def dice_coef(y_true, y_pred):
    """dice指标"""
    smooth = 1e-6  # 防止分母为0的极小值
    y_true_f =y_true# K.flatten(y_true)
    y_pred_f =y_pred# K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f,axis=(0,1,2))
    denom =K.sum(y_true_f,axis=(0,1,2)) + K.sum(y_pred_f,axis=(0,1,2))
    return K.mean((2. * intersection + smooth) /(denom + smooth))

def dice_loss(smooth, thresh):
    """dice_loss,注意"""
    def dice(y_true, y_pred):
        return 1-dice_coef(y_true, y_pred)
    return dice

########################### model building blocks ###########################

def downstage_resBlock(x, stage_id, keep_prob, stage_num=5):
    """
    Vnet左侧的压缩路径的一个stage层
    :param x: 该stage的输入
    :param stage_id: int,表示第几个stage，原论文中从上到下依次是1-5
    :param keep_prob: dropout保留元素的概率，如果不需要则设置为1
    :param stage_num: stage_num是Vnet设置的stage总数
    :return: stage下采样后的输出和stage下采样前的输出，下采样前的输出需要与Vnet右侧的扩展路径连接，所以需要输出保存。
    """
    x0 = x  # x0是stage的原始输入
    # Vnet每个stage的输入会进行特定次数的卷积操作，1~3个stage分别执行1~3次卷积，3以后的stage均执行3次卷积
    # 每个stage的通道数(卷积核个数或叫做feature map数量)依次增加两倍，从16，32，64，128到256
    for _ in range(3 if stage_id > 3 else stage_id):
        x=PReLU()(BatchNormalization()(Conv2D(16 * (2 ** (stage_id - 1)), 5, activation=None, padding ='same', kernel_initializer ='he_normal')(x)))
        print('conv_down_stage_%d:' %stage_id,x.get_shape().as_list())#输出收缩路径中每个stage内的卷积
    x_add=PReLU()(add([x0, x]))
    x_add=Dropout(keep_prob)(x_add)

    if stage_id<stage_num:
        x_downsample=PReLU()(BatchNormalization()(Conv2D(16 * (2 ** stage_id), 2, strides=(2, 2), activation=None, padding ='same', kernel_initializer ='he_normal')(x_add)))
        return x_downsample,x_add  # 返回每个stage下采样后的结果,以及在相加之后的结果
    else:
        return x_add,x_add  # 返回相加之后的结果，为了和上面输出保持一致，所以重复输出

def upstage_resBlock(forward_x, x, stage_id):
    """
    Vnet右侧的扩展路径的一个stage层
    :param forward_x: 对应压缩路径stage层下采样前的特征，与当前stage的输入进行叠加(不是相加)，补充压缩损失的特征信息
    :param x: 当前stage的输入
    :param stage_id: 当前stage的序号，右侧stage的序号和左侧是一样的，从下至上是5到1
    :return:当前stage上采样后的输出
    """
    input = concatenate([forward_x, x], axis=-1)
    for _ in range(3 if stage_id > 3 else stage_id):
        input=PReLU()(BatchNormalization()(Conv2D(16 * (2 ** (stage_id - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(input)))
        print('conv_down_stage_%d:' % stage_id, x.get_shape().as_list())  # 输出收缩路径中每个stage内的卷积
    conv_add=PReLU()(add([x, input]))
    if stage_id>1:
        # 上采样的卷积也称为反卷积，或者叫转置卷积
        conv_upsample=PReLU()(BatchNormalization()(Conv2DTranspose(16 * (2 ** (stage_id - 2)), 2, strides=(2, 2), padding='valid', activation=None, kernel_initializer='he_normal')(conv_add)))
        return conv_upsample
    else:
        return conv_add

########################### Vnet ###########################

def Vnet(pretrained_weights=None, input_size = (256, 256, 1), num_class=1, is_training=True, stage_num=5):
    """
    Vnet网络构建
    :param pretrained_weights:是否加载预训练参数
    :param input_size: 输入图像尺寸(w,h,c),c是通道数
    :param num_class:  数据集的类别总数
    :param is_training:  是否是训练模式
    :param stage_num:  Vnet的网络深度，即stage的总数，论文中为5
    :return: Vnet网络模型
    """
    keep_prob = 0.5 if is_training else 1.0  # dropout概率
    left_featuremaps=[]
    input_data = Input(input_size)
    x = PReLU()(BatchNormalization()(Conv2D(16, 5, activation = None, padding = 'same', kernel_initializer='he_normal')(input_data)))

    # 数据经过Vnet左侧压缩路径处理
    for s in range(1,stage_num+1):
        x, featuremap=downstage_resBlock(x, s, keep_prob, stage_num)
        left_featuremaps.append(featuremap)  # 记录左侧每个stage下采样前的特征

    # Vnet左侧路径跑完后，需要进行一次上采样(反卷积)
    x_up = PReLU()(BatchNormalization()(Conv2DTranspose(16 * (2 ** (s - 2)),2,strides=(2, 2),padding='valid',activation=None, kernel_initializer='he_normal')(x)))

    # 数据经过Vnet右侧扩展路径处理
    for d in range(stage_num-1,0,-1):
        x_up = upstage_resBlock(left_featuremaps[d - 1], x_up, d)
    if num_class>1:
        conv_out=Conv2D(num_class, 1, activation='softmax', padding = 'same', kernel_initializer = 'he_normal')(x_up)
    else:
        conv_out=Conv2D(num_class, 1, activation='sigmoid', padding = 'same', kernel_initializer = 'he_normal')(x_up)

    model=Model(inputs=input_data,outputs=conv_out)
    print(model.output_shape)

    model_dice=dice_loss(smooth=1e-5,thresh=0.5)  # dice损失函数,二分类时可以使用，多分类需要修改
    if num_class > 1:
        model.compile(optimizer=SGD(lr=0.001,momentum=0.99,decay=1e-6), loss='sparse_categorical_crossentropy', metrics = ['ce'])  # metrics看看需不需要修改
    else:
        model.compile(optimizer=SGD(lr=0.001, momentum=0.99, decay=1e-6), loss='binary_crossentropy',
                      metrics=['binary_accuracy'])
        # model.compile(optimizer=SGD(lr=0.001, momentum=0.99, decay=1e-6), loss=dice_loss,metrics=[model_dice])  # 如果需要使用dice和dice_loss函数，则改为注释中的样子
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    plot_model(model, to_file='model.png')  # 绘制网络结构
    return model

model = Vnet(input_size = (512,1024,1), num_class=1, is_training=True, stage_num=5)

