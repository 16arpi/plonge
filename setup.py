from setuptools import setup


setup(
    name="plonge",
    version="0.0.1",
    description="Python app to calculate HF embeddings",
    url="https://github.com/16arpi/plonge",
    author="César Pichon",
    keywords="embeddings",
    licence="GPL-3.0",
    python_requires=">=3.9",
    install_requires=[
        "casanova",
        "sentence_transformers",
        "accelerate",
        "tqdm"
    ],
    entry_points={"console_scripts": ["plonge=plonge.__main__:main"]},
)

