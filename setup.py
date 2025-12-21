from setuptools import find_packages, setup
from typing import List


constant = "-e ."

def get_requirements(filepath : str) -> List[str] : 
    """
    this function will open the list of requirements
    """
    requirements = []
    with open(filepath,"rb") as f :
        requirements = f.readlines()
        for line in requirements:
            requirements = [line.replace("\n","")]
        if constant in requirements:
            requirements.remove(constant)
    return requirements



setup(

name = "mlproject",
version = "0.0.1",
author = "abhijeet",
author_email = "abhijeetmalik70@gmail.com",
packages = find_packages(),
install_requirements = get_requirements("requirements.txt")
    
)