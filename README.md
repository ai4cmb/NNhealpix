# HealpixNN

This is a package to apply neural networks and convolutional neural networks to
spherical signal projected on the healpix grid

## Requirements

- NumPy
- SciPy
- Healpy
- Numba
- Keras
- TensorFlow

## Install
The code is under development therefore to install it just use:

```bash
git clone git@github.com:NicolettaK/HealpixNN.git
[sudo] python setup.py develop [--user]
```

In order to compile the Fortran modules, at the moment you have to do it by hand:

```bash
f2py -m _maptools -c _maptools.f90
```


## Testing

To run a suite of tests, you must have either `nosetests` or
`pytest`. Just run `nosetests` or `pytest` within the `HealpixNN`
folder.
