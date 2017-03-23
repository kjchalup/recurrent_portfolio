"""setup.py for the causehf repository."""
import os
from setuptools import setup

def read(fname):
    """Utiltiy function for README file."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='causehf',
      version="0.0.1",
      description='Stock portfolio generation with recurrent neural networks.',
      # url='',
      author='Krzysztof Chalupka, Alex Teng, Tristan McKinney',
      author_email='krzysztof@cause.ai, alex@cause.ai, tristan@cause.ai',
      license='MIT',
      packages=['linear_hf'],
      install_requires=[
          'pytest==3.0.7',
          'quantiacsToolbox==2.2.11',
          'scipy==0.19.0',
          'sklearn==0.0',
          'tensorflow==1.0.1',
      ],
      long_description=read('README.md'),
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Topic :: Office/Business :: Financial :: Investment",
          "Programming Language :: Python :: 2.7",
          "License :: OSI Approved :: MIT License",
      ],
     )
