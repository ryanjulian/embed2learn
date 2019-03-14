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

## Refactoring issues
### TODO
* ~~Move garage from parent directory to submodule~~
* ~~Switch from conda to pipenv~~
* ~~Update garage~~
* ~~Remove all unused imports~~
* ~~Update changed garage imports~~
* ~~Bring up non-embedded launchers~~
* ~~Refactor NPOTaskEmbedding for Runner~~
* ~~Bring up embedded launchers~~
* (Zhanpeng) Bring up .pkl launchers
* (Zhanpeng) Fix MPC launcher
* (Ryan) Fix mujoco_py install
* (Ryan) Fix baselines install
* (Utkarsh) Fix gym-sawyer task space control
* (Utkarsh??) Fix gym-sawyer pusher
* (Ryan) Fix gym-sawyer packaging
* (Ryan) Fix garage data directory config
* (Ryan) Fix garage packaging
* (Ryan) Fix multiworld packaging

### Launchers tracker

* Blocked by broken gym-sawyer (Zhanpeng)
  - trpo_sawyer.py  # task space control broken
  - random_reacher.py  # ??
  - sawyer_push_ppo.py  # pusher missing
* Broken garage sampler import (waiting for Keren's fix)
  - trpo_dm_control_cartpole.py
* Missing code from MPC (Zhanpeng)
  - ppo_point_random_start.py
* Needs pre-trained .pkl file for testing (Zhanpeng)
  - brute_reach_sequence.py
  - ddpg_point_compose.py
  - ddpg_point_seq_compose.py
  - ddpg_point_seq_task_oriented_compose.py
  - ddpg_sawyer_compose.py
  - search_push_sequence.py
  - search_reach_sequence.py
  - ppo_sawyer_reacher_compose.py
  - ppo_point_compose.py
  - play_point_embed.py
  - play_pusher.py
  - play_sawyer_reach_embed.py
  - play_sawyer_reach_trpo.py
  - dqn_sawyer_compose.py
  - mpc_sawyer_sequencing.py
* DONE
  - trpo_point.py
  - trpo_point_embed.py
  - trpo_sawyer_multiworld.py
  - ddpg_sawyer_reach.py
  - sawyer_reach_ppo2.py
  - sawyer_reach_ppo.py
  - sawyer_reach_trpo.py
  - ppo_point_embed.py
  - sawyer_reach_embed.py
  - sawyer_reach_torque_embed.py
  - trpo_point_onehot.py
  - test_fixture.py

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

