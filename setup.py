from setuptools import find_packages, setup

setup(
    name="llm_rec_eval",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "tqdm",
        "transformers",
        "torch",
        "scikit-learn",
        "click",
        "tenacity",
        "accelerate",
        "bitsandbytes",
        "optuna",
    ],
    author="Julia Medina",
    author_email="a.julia.medina@gmail.com",
    description="Evaluation of LLMs as Recommendation Systems",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/curita/llm-as-recsys-evaluation",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
