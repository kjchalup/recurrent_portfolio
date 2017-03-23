"""setup.py for the causehf repository."""
import os
from setuptools import setup

def read(fname):
    """Utiltiy function for README file."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='causehf',
      # version=__version__,
      description='Stock portfolio generation with recurrent neural networks.',
      # url='',
      # download_url='',
      author='Krzysztof Chalupka, Alex Teng, Tristan McKinney',
      author_email='krzysztof@cause.ai, alex@cause.ai, tristan@cause.ai',
      license='MIT',
      packages=['linear_hf', ],
      include_package_data=True,
      install_requires=[
          'pytest==3.0.7',
          'quantiacsToolbox==2.2.11',
          'scipy==0.19.0',
          'sklearn==0.0',
          'tensorflow==1.0.1',
      ],
      zip_safe=False,
      long_description=read('README.md'),
     )
