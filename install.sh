# A script that installs the starting kit, the environment to execute it,
# and downloads the data

wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda

wget -q http://repo.continuum.io/miniconda/Miniconda-3.6.0-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b -p /home/ubuntu/miniconda
export PATH=/home/ubuntu/miniconda/bin:$PATH
sudo apt-get update
sudo apt-get install --yes build-essential
sudo apt-get install --yes unzip
sudo apt-get install --yes screen
conda update --yes --quiet conda

conda install --yes numpy
conda install --yes pandas
conda install --yes scikit-learn
conda install --yes scikit-image
conda install --yes joblib
conda install --yes cloudpickle
conda install --yes gitpython



# Clone starting kit and download data
git clone https://github.com/ramp-kits/pollenating_insects_3
cd pollenating_insects_3
python download_data.py

# Install keras, tensorflow, mxnet, and pytorch
cd ~
echo "export PATH=\$HOME/miniconda/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc
pip install h5py==2.7.1
pip install tensorflow-gpu==1.3.0
pip install mxnet-cu80==0.11.0
pip install keras==2.0.8
pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl 
pip install torchvision==0.1.9
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn6_6.0.21-1+cuda8.0_amd64.deb
sudo dpkg -i libcudnn6_6.0.21-1+cuda8.0_amd64.deb
