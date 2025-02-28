from setuptools import setup, find_packages

setup(
    name="arabic-spellchecker",
    version="0.1.0",
    author="Mossab Ibrahim, Pablo Gervás, Gonzalo Méndez",
    author_email="mibrahim@ucm.es",
    description="Semantic-Aware Hybrid Arabic Spell Checker with Domain Adaptation",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/arabic-spellcheck/asc2024",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'torch==2.0.1',
        'transformers==4.28.1',
        'pyarabic==1.0.8',
        'arabic-transformers==0.1.5',
        'peft==0.3.0',  # For QLoRA adaptation
        'pytorch-lightning==2.0.2',
        'tokenizers==0.13.3',
        'datasets==2.11.0',
        'nltk==3.8.1',
        'scikit-learn==1.2.2',
        'numpy==1.23.5',
        'pandas==1.5.3',
        'tqdm==4.65.0',
        'flask==2.2.3',  # Web demo
        'python-Levenshtein==0.21.1'  # For CER calculations
    ],
    extras_require={
        'testing': [
            'pytest==7.3.1',
            'pytest-cov==4.0.0'
        ],
        'gpu': [
            'cudatoolkit==11.8.0',
            'torchvision==0.15.2'
        ]
    },
    entry_points={
        'console_scripts': [
            'asc-train=training.train:main',
            'asc-eval=evaluation.benchmark:main'
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    python_requires=">=3.9",
)
