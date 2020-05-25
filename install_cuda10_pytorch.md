## Install python3.6
**note:** It is suggested to use anaconda to manage different python versions. Skip following python installation steps.
```bash
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt-get install python3.6
sudo unlink /usr/bin/python
# can not link python3 to python3.6, other wise the terminal won't open
sudo ln -s /usr/bin/python3.6 /usr/bin/python
sudo apt-get install python3.6-dev
```

## Install nvidia driver
```bash
sudo apt-get remove --purge '^nvidia-.*'
# The following packages will be REMOVED:
#   libcuda1-390* libcuinj64-7.5* nvidia-390* nvidia-cuda-dev* nvidia-cuda-doc*
#   nvidia-cuda-gdb* nvidia-cuda-toolkit* nvidia-modprobe* nvidia-opencl-dev*
#   nvidia-opencl-icd-390* nvidia-prime* nvidia-profiler* nvidia-settings*
#   nvidia-visual-profiler*
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-440
# remember to disable security boot option after reboot, try 430 if 440 is not found
```

## Install CUDA10 and cudnn
Download [cuda 10.0 runfile](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=runfilelocal)
```bash
sudo sh cuda_10.0.130_410.48_linux.run --override --silent --toolkit
sudo apt install nvidia-cuda-toolkit # this might install cuda7.5 for you
nvcc --version # might show cuda 7.5. Don't worry, you have both 7.5 and 10.0 installed
nvidia-smi # will show cuda 10.0
# Download cudnn https://developer.nvidia.com/rdp/cudnn-download, extract the tgz file to ~/Download/cuda
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda # 即将上述代码放入~/.bashrc文件保存后source ~/.bashrc
```

## Install Pytorch
```bash
# under base environment of anaconda
pip install torch==1.4+cu100 torchvision==0.5+cu100 -f https://download.pytorch.org/whl/torch_stable.html
pip install cython pyyaml==5.1
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
Test if installation is successful:
```python
from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)
torch.cuda.is_available()
```

```bash
git clone https://github.com/facebookresearch/detectron2
pip install -e detectron2
pip install jupyterlab
```

## Issues
ImportError: cannot import name '_imaging'
remove pillow and PIL install under /usr/lib/python3/dist-packages
