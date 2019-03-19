# embed2learn
Embedding to Learn

## Installation

### Step 1
Follow the standard garage [setup instructions](http://rlgarage.readthedocs.io/en/latest/user/installation.html).

Also install:
* pipenv
* pyenv (optional but highly recommended)

### Step 2
Check out this repository and setup submodules

```sh
git clone git@github.com:ryanjulian/embed2learn-private.git
cd embed2learn-private
git checkout new-garage
git submodule init
git submodule update
```

### Step 3
Setup the pipenv

```
cd embed2learn-private
pipenv install --dev
```

### Step 4
Fixup some existing install issues

```sh
pipenv run pip uninstall -y mujoco_py
pipenv run pip install mujoco_py
pipenv run python -c 'import mujoco_py'
pipenv run python launchers/ppo_point_embed.py  # prints an error
export SCRIPT_DIR=<paste directory from error>
mkdir -p $SCRIPT_DIR
cp external/garage/scripts/run_experiment.py $SCRIPT_DIR
```

### Step 5
Run
```sh
pipenv run python launchers/ppo_point_embed.py
```

#### Known Issues and Workarounds
* PointEnv plotting broken on MacOS (pygame issue)
* gym-sawyer task space control broken
* baselines install seems broken with pipenv
* mujoco_py install is broken -- numpy versions mismatch

    workaround:
    ```sh
    pipenv run pip uninstall -y mujoco_py
    pipenv run pip install mujoco_py
    pipenv run python -c 'import mujoco_py'
    ```

 * garage data directory is in site-packages
 * garage scripts/run_experiment.py not copied during setup

     workaround:
	 ```sh
    pipenv run python launchers/ppo_point_embed.py  # prints an error
    export SCRIPT_DIR=<paste directory from error>
    mkdir -p $SCRIPT_DIR
    cp external/garage/scripts/run_experiment.py $SCRIPT_DIR
    ```

* multiworld assets not copied during setup.py (make PR)

      workaround: use as a submodule for now

* gym-sawyer vendor files files not copied by setup.py

      workaround: use as a submodule for now

