SETUP

Create a new python environment 3.6.13

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

git clone --recursive https://github.com/NVIDIAGameWorks/kaolin

cd .\kaolin\

python .\setup.py develop

cd ..

cd .\emd\

python .\setup.py install

cd ..

cd .\torch_packages\

pip install .\torch_cluster-1.5.9-cp36-cp36m-win_amd64.whl

pip install .\torch_scatter-2.0.9-cp36-cp36m-win_amd64.whl

pip install .\torch_sparse-0.6.12-cp36-cp36m-win_amd64.whl

pip install torch_geometric

pip install h5py

for visualization:

pip install matplotlib

conda install jupyter

pip install transforms3d==0.3.1

FINISH

EMD: https://github.com/Colin97/MSN-Point-Cloud-Completion/tree/master/emd

Kaolin: https://github.com/NVIDIAGameWorks/kaolin/