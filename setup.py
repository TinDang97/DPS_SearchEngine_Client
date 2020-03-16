from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='dpssearchclient',
      version='1.0.0',
      description='',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/TinDang97/DPS_SearchEngine_Client',
      author='TinDang',
      author_email='rainstone1029x@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
            'dpsutil>=1.1.13',
            'numpy'
      ])
