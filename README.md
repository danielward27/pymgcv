# pymgcv: Generalized Additive Models in Python

**pymgcv** provides a Pythonic interface to R's powerful [mgcv](https://cran.r-project.org/web/packages/mgcv/index.html) library for fitting Generalized Additive Models (GAMs). It combines the flexibility and statistical rigor of mgcv with the convenience of Python's data science ecosystem.

Currently in development. As this is a multilanguage project (R and Python), we use
[pixi](https://pixi.sh/latest/), a package management tool which supports this (via
conda). For development, the ``pymgcv`` can be installed by installing
[pixi](https://pixi.sh/latest/) and running:

```bash
git clone https://github.com/danielward27/pymgcv.git
cd pymgcv
pixi shell --environment=dev
```
