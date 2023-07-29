from setuptools import setup, find_packages

setup(
    name='EasyDeL',
    version='0.0.12',
    author='Erfan Zare Chavoshi',
    author_email='erfanzare82@eyahoo.com',
    description='An open-source library to make training faster and more optimized in Jax/Flax',
    url='https://github.com/erfanzar/EasyDeL',
    packages=find_packages(),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='machine learning, deep learning, pytorch',
    install_requires=[
        'torch>=1.13.0',
        # add any other required dependencies here
    ],
    python_requires='>=3.8',
)
