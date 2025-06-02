'''
the setup.py file is an essential part of packaging and distributing python projects.
it is used by setuptools (or distributing )
'''
from setuptools import find_packages,setup
from typing import List

def get_requirements()->List[str]:
    """
    This Function will return the list of requirements.
    """
    requirement_lst : List[str] = []
    try:
        with open("requirement.txt","r") as file:
            ## read lines from file
            lines = file.readlines()
            ## process ech line
            for line in lines:
                requirement = line.strip()
                ## ignore empty lines and -e .
                if requirement and requirement != '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("No requirements.txt file found")
    return requirement_lst
print(get_requirements())
setup(
    name = "NetworkSecurity",
    version="0.0.1",
    author= "Ali Samad",
    author_email="aliussamd@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements()
)