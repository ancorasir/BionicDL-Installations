# Installation

## Install Archiconda (skip this):
Skip this step as many software compiled for jetson are using python3.6. So we will use the python 3.6 with the system.

Download Installation script from [github](https://github.com/Archiconda/build-tools/releases) under ~/Downloads
```bash
cd ~/Downloads
bash Archiconda3-0.2.3-Linux-aarch64.sh
```
Run a new terminal to activate anaconda environment

## Install DeepClaw
```bash
# Install dependency
sudo pip3 install numpy=1.16.1 pyyaml

git clone https://github.com/bionicdl-sustech/DeepClawBenchmark.git
git clone https://github.com/ancorasir/DeepClaw.git
cd DeepClawBenchmark
pip install .
```

```bash
# run exmple to move ur10
from deepclaw.driver.arms.UR10eController import UR10eController
from deepclaw.utils.IO import read_yaml

robot = UR10eController(read_yaml('/config/ur10.yaml'))

state = robot.get_state()
target = state['tool_vector_actual']
# target[2] = target[2] + 0.01
# target = [0.449768254860253, -0.4716830680747666, 0.19583974070730575, 3.1415, 0, -0.00]
print(target)
robot.move_p(target)
robot.get_state()['tool_vector_actual']
```

## Install librealsense
The following codes will build librealsense from source and install it on the system. It will also install python wrapper for librealsense. The bash script inlcudes some buiding options that can help build the library successfully. If you follow the instructions from the [intel website](https://github.com/IntelRealSense/librealsense/tree/v2.32.1/wrappers/python), you might have problem buiding the lib.
```bash
git clone https://github.com/JetsonHacksNano/installLibrealsense
cd installLibrealsense
# Add -DPYTHON_EXECUTABLE=/usr/bin/python3 to sh script line 93
./buildLibrealsense.sh
```
**Note:** If you met problem after successful buiding the source codes, you can mannually execute the commands after line 100 in buildLibrealsense.sh.
```bash
sudo make install
echo 'export PYTHONPATH=$PYTHONPATH:/usr/local/lib' >> ~/.bashrc
cd $LIBREALSENSE_DIRECTORY
# Copy over the udev rules so that camera can be run from user space
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && udevadm trigger
```
**Issue**: With SDK 2.35.2, there is a bug with cuda on. Solution: navigate to src/proc/pointcloud.h in the source tree and add `#include "synthetic-stream.h"` next to `#include "../include/librealsense2/hpp/rs_frame.hpp"`

Test if you have successfully installed the realsense library
```bash
realsense-viewer
```
Test if you have successfully installed the python wrapper in python
```python
import pyrealsense2 as rs
```

## Install tensorflow 
Install tensorflow and its dependencies from pip source in China
```bash
sudo apt-get install python3-pip
sudo apt-get install libhdf5-dev
sudo pip3 install -U -i https://pypi.tuna.tsinghua.edu.cn/simple numpy==1.16.1 future==0.17.1 mock==3.0.5 h5py==2.9.0 keras_preprocessing==1.0.5 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
sudo pip3 install --pre -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v43 tensorflow==1.15.2+nv20.03
```

## Install Pytorch 1.4

```bash
# Install [torch](https://forums.developer.nvidia.com/t/pytorch-for-jetson-nano-version-1-4-0-now-available/72048)
wget https://nvidia.box.com/shared/static/ncgzus5o23uck9i5oth2n8n06k340l6k.whl -O torch-1.4.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base
sudo pip3 install Cython
sudo pip3 install numpy torch-1.4.0-cp36-cp36m-linux_aarch64.whl

# Install torchvision
sudo apt-get install libjpeg-dev zlib1g-dev
git clone --branch v0.5.0 https://github.com/pytorch/vision torchvision 
cd torchvision
sudo python3 setup.py install
cd ..
```

```python
import torch
print(torch.__version__)
print('CUDA available: ' + str(torch.cuda.is_available()))
print('cuDNN version: ' + str(torch.backends.cudnn.version()))
a = torch.cuda.FloatTensor(2).zero_()
print('Tensor a = ' + str(a))
b = torch.randn(2).cuda()
print('Tensor b = ' + str(b))
c = a + b
print('Tensor c = ' + str(c))
import torchvision
print(torchvision.__version__)
```

## Install PyCUDA
```bash
sudo apt-get install libboost-all-dev
sudo apt-get install python-setuptools
```

You need to download PyCUDA from https://pypi.org/project/pycuda/#files. In the same directory of your PyCUDA download, run this terminal
```
$ tar xzvf pycuda-VERSION.tar.gz
$ cd pycuda-VERSION
```
Go to configure.py and change the /usr/bin/env python into /usr/bin/env python3
```
./configure.py
make -j4
sudo python3 setup.py install
sudo pip3 install .
```
## Install matplotlib
```bash
sudo apt install libfreetype6-dev
pip3 install matplotlib
```

## Install Keras
```bash
pip3 install keras
```

## Install scipy
```bash
pip3 install scipy
```
