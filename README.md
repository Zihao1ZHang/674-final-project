SETUP

WARNING the following package version may vary based on the CPU and GPU model 

We test it under the following environment:

CPU: Ryzen 9 3900x / i711700k

GPU: RTX 3070 / RTX 3080ti

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

How to train:

To run original model: change the line 9 in main.py to: from model import SA_net

To run model with 4 level: change the line 9 in main.py to: from model_level4 import SA_net

To run model with learnable skip_attention: change the line 9 in main.py to: from model_learnable import SA_net

After change modification, run main.py

How to test:

To run original model: test_model.py

To run model with 4 level: test_model_level4.py

To run model with learnable skip_attention: test_model_learnable.py