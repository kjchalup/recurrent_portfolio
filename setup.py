from setuptools import setup

setup(name='causehf',
      # version=__version__,      
      description='Stock portfolio generation with recurrent neural networks.',
      # url='',
      # download_url='',
      author='Krzysztof Chalupka, Alex Teng, Tristan McKinney',
      author_email='krzysztof@cause.ai, alex@cause.ai, tristan@cause.ai',
      license='MIT',
      # packages=[],
      include_package_data = True,
      install_requires=[
          'pytest==3.0.7',
          'quantiacsToolbox==2.2.11',
          'scipy==0.19.0',
          'sklearn==0.0',
          'tensorflow==1.0.1',
      ],
      zip_safe=False,
     )
