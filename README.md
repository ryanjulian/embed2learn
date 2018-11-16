# embed2learn
Embedding to Learn

## Installation

### Step 1
Checkout [garage](https://github.com/rlworkgroup/garage/).

Follow the standard garage [setup instructions](http://rlgarage.readthedocs.io/en/latest/user/installation.html).

If you want to run experiments with Sawyer environments, please also install [sawyer](https://github.com/rlworkgroup/gym-sawyer.git) package in your activated conda environment.

### Step 2
Check out this repository as a submodule of the repository above, into
`sandbox/embed2learn`.

```sh
git submodule add -f git@github.com:ryanjulian/embed2learn.git sandbox/embed2learn
```

### Step 3
```sh
cd sandbox/embed2learn
git submodule init
git submodule update
```

## Running experiements

### Step 1
Activate the anaconda environment for garage
```
conda activate garage
```

### Step 2
```
cd /your/garage/location
export PYTHONPATH=`pwd`
```

### Step3
Train an embedding model and a multi-task policy with point mass environment.
```
python sandbox/embed2learn/launchers/ppo_point_embed.py
```

Train an embedding model and a multi-task policy with sawyer reacher environment.
```
python sandbox/embed2learn/launchers/sawyer_reach_embed.py
``` 

## Citing This Work
If you use this code for scholarly work, please kindly cite our work using one of the Bibtex snippets below.

### General
```
@inproceedings{julian2018scaling,
  title={Scaling simulation-to-real transfer by learning composable robot skills},
  author={Julian, Ryan and Heiden, Eric and He, Zhanpeng and Zhang, Hejia and Schaal, Stefan and Lim, Joseph and Sukhatme, Gaurav and Hausman, Karol},
  booktitle={International Symposium on Experimental Robotics},
  year={2018},
  url={https://arxiv.org/abs/1809.10253}
}
```

### MPC-in-latent space launchers and environments
```
@article{he2018zero,
  title={Zero-Shot Skill Composition and Simulation-to-Real Transfer by Learning Task Representations},
  author={He, Zhanpeng and Julian, Ryan and Heiden, Eric and Zhang, Hejia and Schaal, Stefan and Lim, Joseph and Sukhatme, Gaurav and Hausman, Karol},
  journal={arXiv preprint arXiv:1810.02422},
  year={2018},
  url={https://arxiv.org/abs/1810.02422}
}
```
