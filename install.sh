# A script that installs the starting kit, the environment to execute it,
# and downloads the data

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