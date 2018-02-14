# embed2learn
Embedding to Learn

## Installation (tested on Ubuntu 16.04 LTS)

### Step 1
* Install MuJoCo 1.5.0
* Install patchelf 0.9+ (Ubuntu <=16.04, due to bug in mujoco-py)
 ```shell
 sudo add-apt-repository ppa:jamesh/snap-support
 sudo apt-get update
 sudo apt install patchelf
 ```

### Step 2
```shell
sudo apt install python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig libglew-dev libglfw3 libglew1.13 patchelf libosmesa6-dev swig
```

### Step 3
```shell
pip install --upgrade pipenv
```

### Step 4
```shell
cd embed2learn
pipenv install
```

### Step 5
This is a workaround because rllab's setup of MuJoCo is kind of awkward. I may submit a pull request at some point to fix this. Note: you have to perform this every time you regenerate your pipenv.
```shell
export HERE=`pwd`
export RLLAB=`pipenv --venv`/src/rllab
cd ${RLLAB}
bash scripts/setup_mujoco.sh
```

### Step 6
We also need a workaround which redirect's rllab's config file from the package to your project. Again, fixing this is a TODO.
```shell
cd ${HERE}
cp ${RLLAB}/rllab/config_personal_template.py ${HERE}/config_personal.py
ln -s ${HERE}/config_personal.py ${RLLAB}/rllab/config_personal.py
```

You should edit `config_personal.py` to set your preferences. In particular, I recommend you add the following to redirect data logs to your project directory.
```python
import os.path as osp
LOG_DIR = osp.abspath(osp.join('.', 'data'))
```

## Quickstart
In one shell:
```shell
pipenv run python launchers/trpo_pr2_arm_theano.py
```
In another shell:
```shell
pipenv run python -m rllab.viskit.frontend ./data
```
Then navigate to [http://localhost:5000](http://localhost:5000)

## More info
### System-specific bugs and workarounds
On GPU-based systems, mujoco-py has a bug in finding the NVIDIA OpenGL drivers for non-Docker hosts. See https://github.com/openai/mujoco-py/issues/44. Workaround below.
```shell
# Replace 'nvidia-387' with the NVIDIA driver present on your system
cd embed2learn
echo "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-390/libGL.so" > .env
```

### Dependency Info
From OpenAI Gym
```shell
sudo apt install python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
```
From OpenAI mujoco-py
```shell
sudo apt install libglew-dev patchelf libosmesa6-dev
```
From dm_control
```shell
sudo apt install libglfw3-dev libglew-dev libglew1.13
```

From rllab
```shell
sudo apt install swig
```
