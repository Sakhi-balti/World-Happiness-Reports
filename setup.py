from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """Read requirements.txt and return a list of dependencies."""
    with open(file_path) as f:
        requirements = f.readlines()
        # Remove empty lines, comments, and '-e .'
        requirements = [r.strip() for r in requirements if r.strip() and r.strip() != '-e .']
    return requirements

setup(
    name='world_happiness_report',
    version='0.0.1',
    author='Sakhawat Hussain',
    author_email='sakhawathussain356@gmail.com',
    packages=find_packages(),  # or packages=find_packages(where="src"), package_dir={"": "src"} if using src/
    install_requires=get_requirements('requirements.txt')
)

