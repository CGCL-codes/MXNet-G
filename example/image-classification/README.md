Installing MXNet
------------

The following installation instructions have been tested on Ubuntu 14.04

**1. Prerequisites**

Install the following NVIDIA libraries to setup MXNet with GPU support:

**(1)** Install CUDA 7.5 following the NVIDIA's guide ( http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Installation_Guide_Linux.pdf ).

**(2)** Install cuDNN 5 for CUDA 7.5 following the NVIDIA's guide ( https://developer.nvidia.com/rdp/cudnn-archive#a-collapse51a  ). You may need to register with NVIDIA for downloading the cuDNN library.
Note: Make sure to add CUDA install path to LD_LIBRARY_PATH. Example: export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH

**2. Install and Build**

Building MXNet from source is a 2 step process. Firstly, build the MXNet core shared library, libmxnet.so, from the C++ sources. Secondly, build the language specific bindings. Example - Python bindings

**(1)** Build the MXNet core shared library, libmxnet.so, from the C++ sources
- Install build tools and git.
```javascript
$ sudo apt-get update
$ sudo apt-get install -y build-essential git
```
- Install OpenBLAS, OpenCV.
```javascript
sudo apt-get install -y libopenblas-dev liblapack-dev libopencv-dev
```
- Download MXNet sources and build MXNet core shared library.
```javascript
$ git clone --recursive https://github.com/apache/incubator-mxnet
$ cd mxnet
$ make -j $(nproc)
```
You can explore and use more compilation options in make/config.mk.

**(2)** Build the MXNet Python binding
- Install prerequisites
```javascript
sudo apt-get install ¨Cy python-setuptools python-pip python-numpy python-scipy python-matplotlib
```
- Install the MXNet Python binding.
```javascript
cd python
sudo python setup.py install develop --user
```

Running MXNet
------------

**1. Set up password-less authentication between machines**

If you don't already have an SSH key, generate one via
```javascript
ssh-keygen
```
You'll then need to add your public key to each machine, by appending your public key file~/.ssh/id_rsa.pub to ~/.ssh/authorized_keys on each machine. If your home directory is on a shared filesystem visible to all machines, then simply run
```javascript
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
```
**2. Copy mxnet to other machines**

**(1)** Change path
```javascript
cd example/image-classification
```
**(2)** Write IPs to hosts file. This hosts file contains IPs of the machines in the cluster. These machines should be able to communicate with each other without using passwords.
An example of the contents of the hosts file would be:
```
10.10.10.21
10.10.10.22 
10.10.10.23
```
**(3)** Send mxnet to other machines, simply run
```javascript
python scp_mxnet.py
```
**3. Set the training data ratio for each worker node**

Write training data ratio for each worker node to data_sharding file. Each line in data_sharding file has two digits, which indicate the start and end of the training data, respectively. An example of the contents of the data_sharding file would be (two workers):
```
0 0.3
0.3 1
```

**4. Partition workers into multiple groups**

Write the grouping results to groups file. Each line in groups file indicate one group, which may has one or more IPs of workers. An example of the contents of the groups file would be (three groups):
```
10.10.10.21
10.10.10.22 10.10.10.23
10.10.10.24 10.10.10.25 10.10.10.26
```
**5. Start distributed training**

You can start distributed training just run the *.sh in mxnet/example/image-classification. For example, if you want to run inception-bn small on Cifar10 using GSP, simply run:
```javascript
time ./ gsp_cifar10_inception_small.sh
```
Other models, datasets, and parallelization schemes can be used in similar ways. In the running script, some important parameters are explained as follows:
```
-n: denotes the number of worker nodes to be launched.
-s: denotes the number of parameter server nodes to be launched.
-i: denotes the network interface to be used
--launcher: denotes the mode of communication, here you can use "ssh" mode
-H: denotes the hosts file to be used
--data-dir: denotes the location of the datasets
--batch-size: denotes the batch size of training data to be used
--lr: denotes the learning rate
--lr-factor: denotes the decrease factor of the learning rate
--num-epoch: denotes the epochs num
--num-examples: denotes the num of training examples in one epoch
--gpus: denotes the gpus to be used
--kv-store: denotes the parallelization scheme to be used. (BSP:dist_sync, ASP:dist_async, GSP:dist_gsync, SSP:dist_ssync)
--data-allocator: indicates whether to use data sharding strategy. (0:not use, 1:use)
--staleness: denotes the staleness of workers can't exceed this value
```
More information about MXNet can refer to: https://mxnet.incubator.apache.org/index.html

Support or Contact
------------
If you have any questions, please contact Geyan Ye (<gyye@hust.edu.cn>).
