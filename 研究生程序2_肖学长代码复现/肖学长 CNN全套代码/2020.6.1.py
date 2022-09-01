# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 20:16:34 2020

@author: ZhaoDf
"""
#########################导入函数包###################################################
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io as scio
#import os  
#########################读取数据#####################################################
def read_data_file(data_file):
    with open(data_file, 'r',encoding='utf8') as f:
        data=[]
        for line in f:
            line  = line.split()
            #number=[float(i) for i in line.split()]
            line = np.array(line, dtype=np.float64)
            #data.append(number)
            data.append(line)
        data = np.array(data)
    return data

dl = read_data_file('99.txt')
qk = read_data_file('111.txt')
sth = read_data_file('124.txt')
zc = read_data_file('137.txt')
we = read_data_file('176.txt')
ar = read_data_file('191.txt')
sa = read_data_file('203.txt')

print(dl.shape)
print(qk.shape)
print(sth.shape)
print(zc.shape)
print(we.shape)
print(ar.shape)
print(sa.shape)

dl = dl[:,0]
qk = qk[:,0]
sth = sth[:,0]
zc = zc[:,0]
we = we[:,0]
ar = ar[:,0]
sa = sa[:,0]
##########################数据截取##########################################################
def splitdata(data,n,c):
    processed_data=[]
    for i in range((data.shape[0]-n)//c+1):
        aa = data[i * c : i * c + n]
        aa = aa.reshape(n,1)
        processed_data.append(aa)
    processed_data = np.array(processed_data)
#     print(processed_data.shape)
    return processed_data

dl = splitdata(dl,1024,500)
qk = splitdata(qk,1024,500)
sth = splitdata(sth,1024,500)
zc = splitdata(zc,1024,500)
we = splitdata(we,1024,500)
ar = splitdata(ar,1024,500)
sa = splitdata(sa,1024,500)

print(dl.shape)
print(qk.shape)
print(sth.shape)
print(zc.shape)
print(we.shape)
print(ar.shape)
print(sa.shape)
##数据合并###############################################################
X = np.concatenate((dl,qk),axis=0)
X = np.concatenate((X,sth),axis=0)
X = np.concatenate((X,zc),axis=0)
X = np.concatenate((X,we),axis=0)
X = np.concatenate((X,ar),axis=0)
X = np.concatenate((X,sa),axis=0)

#XX=X.reshape(6802,1024)
#scio.savemat('XX.mat', {'data': XX }) 
##数据标签###############################################################
Y = np.zeros(X.shape[0])
#print(Y.shape)
Y[969:1939]=1
Y[1939:2911]=2
Y[2911:3883]=3
Y[3883:4857]=4
Y[4857:5830]=5
Y[5830:6802]=6
Y_TSNE = Y.reshape(6802,1)  ### 用于tsne降维可视化 ###
#print(Y_TSNE.shape)

Y = np.eye(7)[Y.astype(int).reshape(-1)]
print(Y)
##抽取其中N个样本作为训练集，其余作为测试集##################################
permutation = list(np.random.permutation(X.shape[0]))
shuffled_X = X[permutation,:]
shuffled_Y = Y[permutation,:]
shuffled_Y_TSNE = Y_TSNE[permutation,:]  ####  TSNE

def train_test_set(shuffled_X,shuffled_Y,percent):
    X_train = shuffled_X[0:math.floor(X.shape[0]*percent),]
    Y_train = shuffled_Y[0:math.floor(Y.shape[0]*percent),]
    Y_TSNE_train = shuffled_Y_TSNE[0:math.floor(Y.shape[0]*percent),]  ##### TSNE
    
    X_test = shuffled_X[math.floor(X.shape[0]*percent):,]
    Y_test = shuffled_Y[math.floor(Y.shape[0]*percent):,]
    Y_TSNE_test = shuffled_Y_TSNE[math.floor(Y.shape[0]*percent):,]  ##### TSNE
    return X_train,Y_train,X_test,Y_test,Y_TSNE_train,Y_TSNE_test

X_train,Y_train,X_test,Y_test,Y_TSNE_train,Y_TSNE_test = train_test_set(shuffled_X,shuffled_Y,0.7)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

np.save("X_train.npy",X_train)
np.save("Y_train.npy",Y_train)
np.save("X_test.npy",X_test)
np.save("Y_test.npy",Y_test)
np.save("Y_TSNE_train.npy",Y_TSNE_train)
np.save("Y_TSNE_test.npy",Y_TSNE_test)
##搭建网络###################################################################
import pydot
import graphviz
import keras.backend as K
import matplotlib.pyplot as plt
from IPython.display import SVG
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.utils import layer_utils
from keras.preprocessing import image
from keras import layers, regularizers
from keras.models import Model, load_model
from keras.utils.data_utils import get_file
from keras.utils.vis_utils import model_to_dot
from keras.layers import Flatten, Conv2D, MaxPooling1D
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import AveragePooling1D, MaxPooling1D, Dropout
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D,Conv1D
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization

K.set_image_data_format('channels_last')

def d1_model_ya(input_shape):
    
    X_input = Input(input_shape)
    
    X = Conv1D(32, kernel_size=4, padding='same', strides=2,name = 'layer1')(X_input)  
    X = Activation('relu')(X)
    X = MaxPooling1D(pool_size=2, name='max_pool1')(X)
     
    X = Conv1D(32, kernel_size=4, padding='same', strides=2,name = 'layer2')(X)

    X = Activation('relu')(X)
    X = MaxPooling1D(pool_size=2, name='max_pool2')(X)

    X = Conv1D(32, kernel_size=4, padding='same', strides=2,name = 'layer3')(X) 
    X = Activation('relu')(X)
    X = MaxPooling1D(pool_size=2, name='max_pool3')(X)
    

    X = Dropout(0.5)(X)    
    
    X = Flatten()(X)
    
    X = Dense(64, name='fc1')(X)
    X = Activation('relu')(X)
    X = Dropout(0.5)(X) 
    X = Dense(7, activation='softmax')(X)

    d1_model_ya = Model(inputs=X_input,output=X)
    
    return d1_model_ya
##############################网络训练并保存###################################################################
d1_model_ya = d1_model_ya(input_shape=(1024,1))
d1_model_ya.summary()
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
d1_model_ya.compile(optimizer = opt, loss = "categorical_crossentropy", metrics = ["accuracy"])

from time import time
start = time()
history=d1_model_ya.fit(X_train, Y_train, epochs=10,batch_size=32, validation_split=0.2)
#d1_model_ya.fit(X_train, Y_train, epochs = 10, batch_size=32)
end = time()
print("CPU_time =" + str(end-start))

d1_model_ya.save('d1_model_ya.h5')


#import scipy.io as scio
y1=history.history['acc']
y2=history.history['val_acc']
y3=history.history['loss']
y4=history.history['val_loss']
scio.savemat('acc.mat', {'data': y1 }) 
scio.savemat('val_acc.mat', {'data': y2 })
scio.savemat('loss.mat', {'data': y3 })
scio.savemat('val_loss.mat', {'data': y4 }) ##示例：存为mat文件
##############################绘制网络训练曲线#################################################################
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}

font3 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}

font4 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}


#fig = plt.figure()
figsize = 7,5
figure, ax = plt.subplots(figsize=figsize)
plt.tick_params(labelsize=12)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.plot(history.history['acc'],'-r',label='training acc',linewidth=1.5)
plt.plot(history.history['val_acc'],'-b',label='val acc',linewidth=1.5)
plt.title('model accuracy',font2)
plt.ylabel('accuracy',font2)
plt.xlabel('epoch',font2)
plt.legend(loc='lower right',prop=font2)

#fig = plt.figure()
figsize = 7,5
figure, ax = plt.subplots(figsize=figsize)
plt.tick_params(labelsize=12)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.plot(history.history['loss'],'-r',label='training loss',linewidth=1.5)
plt.plot(history.history['val_loss'],'-b', label='val loss',linewidth=1.5)
plt.title('model loss',font2)
plt.ylabel('loss',font2)
plt.xlabel('epoch',font2)
plt.legend(loc='upper right',prop=font3)

figsize = 7,5
figure, ax = plt.subplots(figsize=figsize)
plt.tick_params(labelsize=12)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.plot(history.history['acc'],'-g',label='training acc',linewidth=1.5)
plt.plot(history.history['val_acc'],'-r',label='val acc',linewidth=1.5)
plt.plot(history.history['loss'],'-y',label='training loss',linewidth=1.5)
plt.plot(history.history['val_loss'],'-b', label='val loss',linewidth=1.5)
plt.title('model loss and accuracy',font2)
plt.ylabel('value',font2)
plt.xlabel('epoch',font2)
plt.legend(loc='best',prop=font2)

##################模型评估###########################################################################
#################模型评估############################################################################
################模型评估############################################################################
d1_model = load_model('d1_model_ya.h5')
start2 = time()
preds = d1_model.evaluate(x = X_test, y = Y_test)
end2 = time()
print("CPU_time2 =" + str(end2-start2))
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
###################################绘制混淆矩阵#####################################################
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools

model=load_model("d1_model_ya.h5")
y_pred=model.predict(X_test).argmax(axis=1)
y_true=Y_test.argmax(axis=1)
print(y_pred.shape,y_true.shape)
C= confusion_matrix(y_true, y_pred)
 
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):### jet Blues Reds Greens gray cyan Paired
   
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
         
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    
    plt.title(title)
    
    cb = plt.colorbar()
    for l in cb.ax.yaxis.get_ticklabels():
        l.set_family('Times New Roman')
        
    tick_marks = np.arange(len(classes))
    
    plt.xticks(tick_marks, classes, rotation=45,fontsize=10)
    plt.yticks(tick_marks,classes,fontsize=10) 
    fmt = '.2f' if normalize else 'd'
#    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        
        plt.text(j, i, format(cm[i, j], fmt),size = 12, family = "Times New Roman",color = "black", horizontalalignment="center")

    plt.tight_layout()
    plt.ylabel('True label',font2)
    plt.xlabel('Predicted label',font2)
    
attack_types = ['NOR', 'IRF1', 'BF1', 'ORF1','IRF2', 'BF2', 'ORF2',]

figsize = 7,5
figure, ax = plt.subplots(figsize=figsize)
plt.tick_params(labelsize=12)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.title('Confusion matrix',font2)
#plot_confusion_matrix(C, classes=attack_types, normalize=True)
plot_confusion_matrix(C, classes=attack_types, normalize=False)



##########################Tsne可视化##################################################################
from sklearn.manifold import TSNE

dense_layer = Model(inputs=d1_model_ya.input,  
                    outputs=d1_model_ya.get_layer('fc1').output)

dense1_output = dense_layer.predict(X_test)
dense1_output.shape

tsne = TSNE(n_components=2, random_state=0)
X_test_tsne = tsne.fit_transform(dense1_output)

Y_TSNE_test.shape
Y_TSNE_test=Y_TSNE_test.reshape(2041)


scio.savemat('X_test_tsne.mat', {'data': X_test_tsne })
scio.savemat('Y_TSNE_test.mat', {'data': Y_TSNE_test })

figsize = 7,5
figure, ax = plt.subplots(figsize=figsize)
plt.tick_params(labelsize=12)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

plt.plot(X_test_tsne[:, 0][Y_TSNE_test==0], X_test_tsne[:, 1][Y_TSNE_test==0], "g^", label="NOR",markersize=5)
plt.plot(X_test_tsne[:, 0][Y_TSNE_test==1], X_test_tsne[:, 1][Y_TSNE_test==1], "bs", label="IRF1",markersize=5)
plt.plot(X_test_tsne[:, 0][Y_TSNE_test==2], X_test_tsne[:, 1][Y_TSNE_test==2], "ro", label="BF1",markersize=5)
plt.plot(X_test_tsne[:, 0][Y_TSNE_test==3], X_test_tsne[:, 1][Y_TSNE_test==3], "y+", label="ORF1",markersize=5)
plt.plot(X_test_tsne[:, 0][Y_TSNE_test==4], X_test_tsne[:, 1][Y_TSNE_test==4], "md", label="IRF2",markersize=5)
plt.plot(X_test_tsne[:, 0][Y_TSNE_test==5], X_test_tsne[:, 1][Y_TSNE_test==5], "k<", label="BF2",markersize=5)
plt.plot(X_test_tsne[:, 0][Y_TSNE_test==6], X_test_tsne[:, 1][Y_TSNE_test==6], "c>", label="ORF2",markersize=5)

plt.legend(loc="upper left",prop=font4)
plt.ylabel('Component1',font2)
plt.xlabel('Component2',font2)
#plt.title("tsen",font2)
#plt.savefig("D:/")



