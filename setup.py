# setup.py
from setuptools import setup, find_packages

setup(
    name='embed2learn',
    packages=[
        package for package in find_packages()
        if package.startswith('embed2learn')
    ],
    version='0.0.0',
)
