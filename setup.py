from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='tfstbd',
    version='1.0.0',
    description='Sentences and tokens boundary detector',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/shkarupa-alex/tfstbd',
    author='Shkarupa Alex',
    author_email='shkarupa.alex@gmail.com',
    license='MIT',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'tfstbd-convert=tfstbd.convert:main',
            'tfstbd-dataset=tfstbd.dataset:main',
            'tfstbd-check=tfstbd.check:main',
            'tfstbd-vocab=tfstbd.vocab:main',
            'tfstbd-train=tfstbd.train:main',
            # 'tfstbd-infer=tfstbd.infer:main',
        ],
    },
    python_requires='>=3.6.0',
    install_requires=[
        'tensorflow>=2.3.0',
        'tfmiss>=0.6.0',
        'nlpvocab>=1.2.0',
        'conllu>=2.3.2',
        'keras-tuner>=1.0.1'
    ],
    setup_requires=['pytest-runner'],
    tests_require=[
        'pytest',
        'ufal.udpipe>=1.2.0.1'
    ]
)
