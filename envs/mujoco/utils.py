from os import path
import tempfile


def mujoco_model_path(file):
    model_dir = path.abspath(
        path.join(path.dirname(__file__), '../../vendor/mujoco_models'))
    return path.join(model_dir, file)
