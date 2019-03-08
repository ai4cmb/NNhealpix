# NNhealpix

[![Documentation Status](https://readthedocs.org/projects/nnhealpix/badge/?version=latest)](https://nnhealpix.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/ai4cmb/NNhealpix.svg?branch=master)](https://travis-ci.com/ai4cmb/NNhealpix)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a package to apply neural networks and convolutional neural
networks to spherical signal projected on the healpix grid.


## Requirements

- Healpy
- Keras
- Matplotlib
- NumPy
- Numba
- SciPy
- TensorFlow

To build the code, you are advised to use
[Filt](https://pypi.org/project/flit/).

## Install

The code is under development therefore to install it just use:

```bash
git clone git@github.com:ai4cmb/NNhealpix.git
flit install --symlink [--python path/to/python]
```

If you do not want to use Flit, you can install the library using the
following command:

```
[sudo] python setup.py develop [--user]
```


## Testing

To run a suite of tests, you must have either `nosetests` or
`pytest`. Just run `nosetests` or `pytest` within the `NNhealpix`
folder.


## License

The library is released under a MIT license. See the file LICENSE for
more information.
