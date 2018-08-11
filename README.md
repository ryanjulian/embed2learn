[![CircleCI](https://circleci.com/gh/ryanjulian/embed2learn.svg?style=shield&circle-token=c06ba07f6cec915ee03365f69edf0286b1538be5)](https://circleci.com/gh/ryanjulian/embed2learn)
[![TravisCI](https://travis-ci.com/ryanjulian/embed2learn.svg?token=5Ha2ycwuRnc34dpruRpP&branch=master)](https://travis-ci.com/ryanjulian/embed2learn)

# embed2learn
Embedding to Learn

## Installation

### Step 1
Checkout my fork of [rllab](https://github.com/users/ryanjulian/rllab/) and switch to the [integration](https://github.com/ryanjulian/rllab/tree/integration) branch.

Follow the standard rllab [setup instructions](http://rllab.readthedocs.io/en/latest/user/installation.html).

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
cd external/multiworld
git checkout russell
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```
