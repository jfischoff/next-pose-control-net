## Installation

### Setup

Install shared libraries for opencv

```
sudo apt-get update
sudo apt-get install libgl1-mesa-glx
sudo apt-get install ffmpeg
```


Create a virtual environment to install the dependencies

```
python3 -m venv $(pwd)/.venv
```

Activate the virtual environment

```
source .venv/bin/activate
```

### Install Dependencies

```
pip install -r requirements.txt
```

then 

```
pip install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-2.0-cp38-cp38-linux_x86_64.whl
```

if you are on a tpu machine