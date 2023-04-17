## Installation

### Setup

Install shared libraries for opencv

```
sudo apt-get update
sudo apt-get install libgl1-mesa-glx
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