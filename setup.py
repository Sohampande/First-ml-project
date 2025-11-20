from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
         This fucntion is responsible for returning the list of requirements
        mentioned in the requirements.txt file
    '''
    with open(file_path) as f:
        require = f.readlines()
        require = [req.replace('\n', '') for req in require]
        if HYPEN_E_DOT in require:
            require.remove(HYPEN_E_DOT)
    return require

setup(
    name='First-ml-project',
    version='0.0.1',
    author='Sohampande',
    author_email='sohampande58@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirments.txt')
)