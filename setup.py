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
        'tensorflow>=2.5.0',
        'tensorflow-addons>=0.13.0',
        'tensorflow-datasets>=4.2.0',
        'tfmiss>=0.13.6',
        'nlpvocab>=1.2.0',
        'conllu>=4.0'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)
