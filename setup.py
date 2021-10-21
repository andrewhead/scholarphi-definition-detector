from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    required = f.read().splitlines()
    required = [str(r) for r in required if r and not r.startswith("#") and not r.startswith("http")]

setup(
    name="heddex",
    version="0.0.5",
    description="Definition detection on scientific text",
    package_dir={"": "."},
    packages=find_packages("."),
    long_description=open("README.md").read(),
    url="https://github.com/dykang/scholarphi_nlp_internal",
    author="Dongyeop Kang",
    author_email="dongyeopk@berkeley.edu",
    keywords="",
    classifiers=[
        "Development Status :: 2 - Beta",
        "Programming Language :: Python :: 3.7.3",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    license="Apache License 2.0",
    install_requires=required,
    extras_require={
        "dev": ["black==20.8b1", "pytest"],
    },
    python_requires=">=3.6",
    zip_safe=False,
)
