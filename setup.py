"""
Setup script for Power Market Simulation Package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="power-market-sim",
    version="0.1.0",
    author="Power Market Simulation Team",
    description="Batteriespeicher-Kannibalisierungsanalyse für den deutschen Strommarkt",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/power-market-sim",  # Update with actual URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "pypsa>=0.27.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "networkx>=3.0",
        "scipy>=1.10.0",
        "streamlit>=1.30.0",
        "plotly>=5.18.0",
        "highspy>=1.5.0",
        "pyyaml>=6.0",
        "pyarrow>=14.0.0",
        "openpyxl>=3.1.0",
        "tables>=3.8.0",
        "psutil>=5.9.0",
        "tqdm>=4.66.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "ipython>=8.12.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "power-market-sim=dashboard.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)
