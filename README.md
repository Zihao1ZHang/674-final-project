SETUP

WARNING the following package version may vary based on the CPU and GPU model 

We test it under the following environment:

CPU: Ryzen 9 3900x / i711700k

GPU: RTX 3070 / RTX 3080ti

Create a new python environment 3.6.13(Anaconda3)

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

How to train:

To run original model: change the line 9 in main.py to: from model import SA_net

To run model with 4 level: change the line 9 in main.py to: from model_level4 import SA_net

To run model with learnable skip_attention: change the line 9 in main.py to: from model_learnable import SA_net

After change modification, run main.py

How to test:

To run original model: test_model.py

To run model with 4 level: test_model_level4.py

To run model with learnable skip_attention: test_model_learnable.py

The test results are saved in the result folder




Source code each of the team members wrote:

The implementations of loss functions are from:
EMD: https://github.com/Colin97/MSN-Point-Cloud-Completion/tree/master/emd
Chamfer distance from Kaolin: https://github.com/NVIDIAGameWorks/kaolin/

The reference of the model: https://github.com/RaminHasibi/SA_Net 
(We refer to the baseline of this Github project to build the model in model.py)

Shenghao Guan: test_model.py(11-19, 22-64(Use the plot_pcds from https://github.com/lynetcha/completion3d to plot point cloud)),</br>
               model.py() (11-46, 66-87), fix bugs in others' code</br>
Zehui Lin: model.py(90-125, 166-187, 201-205), main.py(79-106), fix bugs in others' code</br>
Zihao Zhang: main.py(13-58(Use the ShapenetDataProcess from https://github.com/lynetcha/completion3d), 61-76),</br>
             model.py(128-164, 190-196, 207-232), fix bugs in others' code</br>
Together: The extension of the models(additional folding block and learnable)
