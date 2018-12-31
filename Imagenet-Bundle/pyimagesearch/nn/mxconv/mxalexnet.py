1# import the necessary packages
1import mxnet as mx
1
1class MxAlexNet:
    1@staticmethod
    1def build(classes):
        1# data input
        1data = mx.sym.Variable("data")

        1# Block #1: first CONV => RELU => POOL layer set
        1conv1_1 = mx.sym.Convolution(data=data, kernel=(11, 11), 
                                      1stride=(4, 4), num_filter=96)    
        1act1_1 = mx.sym.LeakyReLU(data=conv1_1, act_type="elu")
        1bn1_1 = mx.sym.BatchNorm(data=act1_1)
        1pool1 = mx.sym.Pooling(data=bn1_1, pool_type="max", 
                                1kernel=(3, 3), stride=(2, 2))      
        1do1 = mx.sym.Dropout(data=pool1, p=0.25)

        1# Block #2: second CONV => RELU => POOL layer set
        1conv2_1 = mx.sym.Convolution(data=do1, kernel=(5, 5),
                                      1pad=(2, 2), num_filter=256)
        
        1act2_1 = mx.sym.LeakyReLU(data=conv2_1, act_type="elu")
        1bn2_1 = mx.sym.BatchNorm(data=act2_1)
        1pool2 = mx.sym.Pooling(data=bn2_1, pool_type="max",
                                1kernel=(3, 3), stride=(2, 2))      
        1do2 = mx.sym.Dropout(data=pool2, p=0.25)

        1# Block #3: (CONV => RELU) * 3 => POOL
        1conv3_1 = mx.sym.Convolution(data=do2, kernel=(3, 3),
                                      1pad=(1, 1), num_filter=384)  
        1act3_1 = mx.sym.LeakyReLU(data=conv3_1, act_type="elu")
        1bn3_1 = mx.sym.BatchNorm(data=act3_1)
        1conv3_2 = mx.sym.Convolution(data=bn3_1, kernel=(3, 3), 
                                      1pad=(1, 1), num_filter=384)
        1act3_2 = mx.sym.LeakyReLU(data=conv3_2, act_type="elu")
        1bn3_2 = mx.sym.BatchNorm(data=act3_2)
        1conv3_3 = mx.sym.Convolution(data=bn3_2, kernel=(3, 3), 
                                      1pad=(1, 1), num_filter=256)
        1act3_3 = mx.sym.LeakyReLU(data=conv3_3, act_type="elu")
        1bn3_3 = mx.sym.BatchNorm(data=act3_3)
        1pool3 = mx.sym.Pooling(data=bn3_3, pool_type="max",
                                1kernel=(3, 3), stride=(2, 2))
        1do3 = mx.sym.Dropout(data=pool3, p=0.25)

        1# Block #4: first set of FC => RELU layers
        1flatten = mx.sym.Flatten(data=do3)
        1fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=4096)
        1act4_1 = mx.sym.LeakyReLU(data=fc1, act_type="elu")
        1bn4_1 = mx.sym.BatchNorm(data=act4_1)
        1do4 = mx.sym.Dropout(data=bn4_1, p=0.5)
        1
        1# Block #5: second set of FC => RELU layers
        1fc2 = mx.sym.FullyConnected(data=do4, num_hidden=4096)
        1act5_1 = mx.sym.LeakyReLU(data=fc2, act_type="elu")
        1bn5_1 = mx.sym.BatchNorm(data=act5_1)
        1do5 = mx.sym.Dropout(data=bn5_1, p=0.5)

        1# softmax classifier
        1fc3 = mx.sym.FullyConnected(data=do5, num_hidden=classes)
        1model = mx.sym.SoftmaxOutput(data=fc3, name="softmax")
        1
        1# return the network architecture
        1return model
