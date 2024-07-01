A wrapper of TBLIS library
==========================

2021-03-14

* Version 0.1

Install
-------
```
pip install pyscf-tblis
```

This will compile and install TBLIS in addition to the wrapper. If you already
have TBLIS installed on your system (e.g. if you installed the conda-forge
package) you can run

```
CMAKE_CONFIGURE_ARGS="-DVENDOR_TBLIS=off" pip install . --no-deps
```

