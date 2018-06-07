import find_mxnet  
import mxnet as mx  
  
def get_symbol(num_classes = 10):  
    data = mx.symbol.Variable('data')  
    # first conv  
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), stride=(1,1), pad=(2,2), num_filter=32)  
    bn1 = mx.symbol.BatchNorm(data=conv1)  
    relu1 = mx.symbol.Activation(data=bn1, act_type="relu")  
    pool1 = mx.symbol.Pooling(data=relu1, pool_type="max",  
              kernel=(3,3), stride=(2,2))  
    # second conv  
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), stride=(1,1), pad=(2,2), num_filter=32)  
    bn2 = mx.symbol.BatchNorm(data=conv2)  
    relu2 = mx.symbol.Activation(data=bn2, act_type="relu")  
    pool2 = mx.symbol.Pooling(data=relu2, pool_type="avg",  
              kernel=(3,3), stride=(2,2))  
    # third conv  
    conv3 = mx.symbol.Convolution(data=pool2, kernel=(5,5), stride=(1,1), pad=(2,2), num_filter=64)  
    bn3 = mx.symbol.BatchNorm(data=conv3)  
    relu3 = mx.symbol.Activation(data=bn3, act_type="relu")  
    pool3 = mx.symbol.Pooling(data=relu3, pool_type="avg",  
              kernel=(3,3), stride=(2,2), name="final_pool")  
    # first fullc  
    flatten = mx.symbol.Flatten(data=pool3)  
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=64)  
    relu4 = mx.symbol.Activation(data=fc1, act_type="relu")  
    # second fullc  
    fc2 = mx.symbol.FullyConnected(data=relu4, num_hidden=num_classes)  
    # loss  
    softmax = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')  
    return softmax  
