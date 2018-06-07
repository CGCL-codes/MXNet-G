import find_mxnet  
import mxnet as mx  

prefix = '/home/yegeyan/MXNet-G/example/image-classification/model/mnist'  
iteration = 3 
model_load = mx.model.FeedForward.load(prefix, iteration)  
data_shape = (784, )

test = mx.io.MNISTIter(
            image       = "/home/yegeyan/mxnet_data/mnist/t10k-images-idx3-ubyte",
            label       = "/home/yegeyan/mxnet_data/mnist/t10k-labels-idx1-ubyte",
            input_shape = data_shape,
            batch_size  = 1) 

[prob, data, label] = model_load.predict(test, return_data=True)

accuracy_num = 0

for i in range(len(prob)):
    if prob[i].tolist().index(max(prob[i].tolist())) == label[i]:
        accuracy_num += 1
        
print "test accuracy:", float(accuracy_num) / len(prob)

