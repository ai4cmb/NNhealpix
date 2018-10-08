# HealpixNN

This is a package to apply neural networks and convolutional neural networks to
spherical signal projected on the healpix grid


## Install
The code is under development therefore to install it just use:

```bash
git clone git@github.com:NicolettaK/HealpixNN.git
[sudo] python setup.py develop [--user]
```

After this, you need to compile a small Fortran module. Run the following command:

```bash
cd nnhealpix && f2py -m _maptools -c _maptools.f90
```
