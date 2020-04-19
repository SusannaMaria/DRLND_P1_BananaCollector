# Install Environment
Installation for Windows10/64 PC -  Dell Inspiron P65F with NVIDIA GPU
| CPU             | GPU |
:-------------------------:|:-------------------------:
![](static/cpu_info.jpg)  |  ![](static/gpu_info.jpg)

1. Anaconda

https://repo.anaconda.com/archive/Anaconda3-2020.02-Windows-x86_64.exe

2. Environment
Python 3.6 Environment and not newer is necessary because of Unity 0.4 dependencies like Tensorflow
```
    conda create -n unity_mlagent python=3.6
    conda activate unity_mlagent
```
3. Tensorflow 

    Download of wheel because pip install was not working directly
https://files.pythonhosted.org/packages/fd/70/1a74e80292e1189274586ac1d20445a55cb32f39f2ab8f8d3799310fcae3/tensorflow-1.7.1-cp36-cp36m-win_amd64.whl
```
    pip install tensorflow-1.7.1-cp36-cp36m-win_amd64.whl
```
4. Unity ML Agents

    Download Unity ml agents https://codeload.github.com/Unity-Technologies/ml-agents/zip/0.4.0b and unzip
```
    cd ml-agents-0.4.0b\python
    pip install .
```

5. Pytorch

    Pytorch will be used for the DQN Agent
```
    conda install -c pytorch pytorch
```

6. Banana Environment

    Download https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip
