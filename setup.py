from setuptools import find_packages
from setuptools import setup

def read(file_name):
    with open(file_name, "r") as f:
        txt = f.read()
    return txt

setup(
    name='genetic_feature_selection',
    license='MIT',
    description='Module for genetic feature selection.',
    long_description=read("README.rst"),
    author='Magnus P. Nytun',
    author_email='magnusnytun92@gmail.com',
    url='https://github.com/MaggiePN92/genetic-algo-feature-selection',
    packages= ["genetic_feature_selection"],#find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],

    keywords=[
       "feature", "selection", "genetic", "algorithm"
    ],
    python_requires='>=3.7',
    install_requires=[
        "tqdm"
    ]
)