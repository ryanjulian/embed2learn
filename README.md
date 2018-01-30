# embed2learn
Embedding to Learn

## Prerequisites (tested on Ubuntu 16.04 LTS)

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
sudo apt install python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig libglew-dev libglfw3 libglew1.13 patchelf 
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

### Info
From OpenAI Gym
```shell
sudo apt install python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
```
From OpenAI mujoco-py
```shell
sudo apt install libglew-dev patchelf
```
From dm_control
```shell
sudo apt install libglfw3-dev libglew-dev libglew1.13
```


