## Simulation experiment procedure for ISER 2018

1. Make sure your launcher uses an appropriate `exp_prefix` parameter (e.g. `exp_prefix=trpo_reacher`)
1. Always start from `master`
    ```sh
    cd garage
    git checkout master
    git reset --hard HEAD
    ```
1. Modify parameters (only) in launcher file
1. Make a commit to the `experiments` branch and push to GitHub
    ```sh
    git checkout -b experiments
    git commit -m "Experiment description"
    git push origin experiments  # don't force-push!
    ```
1. Run experiment
1. Manually log the git commit hash to the experiment directory
    ```sh
    export EXP_DIR_SUFFIX="my_launcher_prefix/my_experiment"  # copied from run_experiment output
    echo "$(git rev-parse HEAD)" > data/local/$EXP_DIR_SUFFIX/git_hash
    ```
1. Copy the log directory (in `garage/data/local/<experiment_prefix>/<experiment>`) to the GCS bucket:
    ```sh
    gsutil -m rsync -r garage/data/local/$EXP_DIR_SUFFIX gs://resl-iser2018/experiments/$EXP_DIR_SUFFIX
    ```

## Google Cloud Storage setup
1. [Install Google Cloud SDK](https://cloud.google.com/sdk/)
1. `gcloud init`
1. Choose project `resl-iser2018`
1. Choose zone `us-west-1c`
1. Test
    ```sh
    gsutil cat gs://resl-iser2018/hello_world.txt
    ```
    
 ## Testing
 When testing your scripts or experimenting with gsutil, please use the devel bucket at `gs://resl-iser2018-devel` to avoid polluting the history of the authoritative bucket.
