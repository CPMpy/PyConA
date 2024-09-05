from setuptools import find_packages, setup
import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

with open("README.md", "r") as descr_file:
    long_description = descr_file.read()

setup(
    name='pycona',
    version=get_version("pycona/__init__.py"),
    author='Dimos Tsouros',
    author_email="dimos.tsouros@kuleuven.be",
    license='Apache 2.0',
    description='A cpmpy-based library for constraint acquisition.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dimosts/",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'cpmpy>=0.9',
    ],
    #extra dependencies.
    extras_require={
        "FULL":  ["scikit-learn", "networkx"],
    },
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)
