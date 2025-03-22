# This is a host machine preparation script.

#Be in home dir
cd

# Create a virtualenv
sudo apt install python3.10-venv
cd .virtualenvs/
python3.10 -m venv "cvMedMamba"
source cvMedMamba/bin/activate

# Clone this repo
mkdir CVproj & cd CVproj
git clone https://github.com/vodkolav/MedMamba.git
cd MedMamba/ || exit
mkdir data
mkdir checkpoints

# install specific cuda
sudo apt purge cuda-* # first uninstall all other versions of cuda that we don't need
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
# we need specifically cuda-toolkit, as it does not update nvidia drivers, which 
sudo apt install cuda-toolkit-11-8


# add these lines to the end of ~/.bashrc of current user
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
# reopen terminal  (or source ~/.bashrc)
echo CUDA_HOME is now $CUDA_HOME
# add newly installed cuda to path
#export PATH=$PATH:/usr/local/cuda-11.8/bin

# install specific pytorch 2.0.1+cu118
pip uninstall torch torchaudio torchvision # first uninstall all other versions of torch that we don't need
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu118  --no-cache-dir

# download and install mamba_ssm
wget https://github.com/state-spaces/mamba/releases/download/v2.0.4/mamba_ssm-2.0.4+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm-2.0.4+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# compile and install causal-conv1d
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d/
export MAX_JOBS=8 CAUSAL_CONV1D_FORCE_BUILD="TRUE" CAUSAL_CONV1D_SKIP_CUDA_BUILD="FALSE"
python setup.py bdist_wheel --dist-dir=dist
# if it complains about wrong version of gcc, use instructions in install_gcc_11.sh 
python setup.py install

cd ../MedMamba
# install the rest of medmamba dependencies
pip install -r requirements.txt

# install medmnist datasets library and gdown tool for downloading from google drive
pip install medmnist gdown

# download our dataset (dermamnist)
python -m medmnist download --size=224 --root=data

# download our checkpoints
# link: https://drive.google.com/file/d/1hihAMKhV4gxne8y1p-xjVgjC4vB7Pcm7/view?usp=sharing
export FILE_ID=1hihAMKhV4gxne8y1p-xjVgjC4vB7Pcm7
export FILENAME="checkpoints/DermaMNIST.pth"
gdown "https://drive.google.com/uc?id=$FILE_ID" -O "$FILENAME"
