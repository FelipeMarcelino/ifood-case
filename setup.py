
from setuptools import find_packages, setup

# Lê o conteúdo do README para long_description
with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

# Lê as dependências do requirements.txt (se existir)
requirements_path = "requirements.txt"
with open(requirements_path, encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="ifood_case",
    version="0.1.0",
    author="Felipe Marcelino",
    author_email="felipe.marcelino1991@gmail.com",
    description="Projeto de ciência de dados para o case iFood",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FelipeMarcelino/ifood-case",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
)
