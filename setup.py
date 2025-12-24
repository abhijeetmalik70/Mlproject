from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Remove new line characters (\n)
        requirements = [req.replace("\n", "") for req in requirements]

        # Remove '-e .' because it's only for triggering setup.py
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Abhijeet',
    author_email='abhijeetmalik70@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)