Author: Zhang Hao-zhi  zhanghz@mail.sustc.edu.cn

参考 https://www.tensorflow.org/tutorials/mnist/pros/
做的MNIST CNN
digit-conv.py是CNN训练程序
checkpoint
digit.data-00000-of-00001
digit.index
digit.meta
四个文件是digit-conv.py训练20000步后的网络参数
digit-conv-restore.py说明了如何从文件恢复网络
graph文件夹里是tensorboard数据
在cmd执行tensorboard --logdir=目录/graph
consol.txt记录了CNN训练时候的python consol输出
MNIST_data是训练集
