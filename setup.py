"""setup.py for the causehf repository."""
import os
from setuptools import setup

def read(fname):
    """Utiltiy function for README file."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='causehf',
      version='0.1.0',
      description='Stock portfolio generation with recurrent neural networks.',
      # url='',
      author='Krzysztof Chalupka, Tristan McKinney, Alex Teng',
      author_email='krzysztof@cause.ai, tristan@cause.ai, alex@cause.ai',
      license='MIT',
      packages=['linear_hf'],
      install_requires=[
          'pytest==3.0.7',
          'quantiacsToolbox==2.2.11',
          'scipy==0.19.0',
          'sklearn==0.0',
          'tensorflow==1.0.1',
      ],
      long_description=read('README.rst'),
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Topic :: Office/Business :: Financial :: Investment',
          'Intended Audience :: Financial and Insurance Industry',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 2.7',
          'License :: OSI Approved :: MIT License',
      ],
      keywords=['finance', 'neural networks', 'machine learning'],
      package_data={'sample': ['tickerData/fake1.txt']}
     )
