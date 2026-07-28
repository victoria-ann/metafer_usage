from setuptools import setup, find_packages
from typing import List

def get_requirements() -> List[str]:
    """Reads the requirements from a file and returns them as a list."""
    requirement_list:List[str] = []
    try:
        with open('requirements.txt', 'r') as file:
            # Read lines from the file
            lines = file.readlines()
            # Strip whitespace and ignore empty lines and -e .
            for line in lines:
                requirements = line.strip()
                if requirements and requirements != '-e .':
                    requirement_list.append(requirements)
     
    except FileNotFoundError:
        print("Error: 'requirements.txt' was not found.")
    return requirement_list

setup(
    name='metafer_usage',
    version='0.0.1',
    author='Vicoria Palecek',
    author_email='victoria.palecek@example.com',
    packages=find_packages(),
    install_requires=get_requirements()
)