[![Build Status](https://travis-ci.com/ai4cmb/NNhealpix.svg?branch=master)](https://travis-ci.com/ai4cmb/NNhealpix)
# NNhealpix

This is a package to apply neural networks and convolutional neural
networks to spherical signal projected on the healpix grid.


## Requirements

- Healpy
- Matplotlib
- NumPy
- Numba
- SciPy
- Tensorflow 2+

If you prefer to use PyTorch instead of Keras/TensorFlow, [**aasensio**](https://github.com/aasensio) has implemented a port of the library available here: [github.com/aasensio/sphericalCNN](https://github.com/aasensio/sphericalCNN).

## Install

The code is still under development. To install, use the following command:
```bash
[sudo] python setup.py develop [--user]
```

To automatically install the requirements, use the following command:
```bash
pip install -r requirements.txt
```

## Testing

To run a suite of tests, you must have either `nosetests` or
`pytest`. Just run `nosetests` or `pytest` within the `NNhealpix`
folder.

## Citing

If you use this code, please cite the following paper: *Convolutional neural networks on the HEALPix sphere: a pixel-based algorithm and its application to CMB data analysis*, Krachmalnicoff, N. & Tomasi, M. (A&A, Aug 2019).

```
@article{ KrachmalnicoffTomasi2019,
	author = {{Krachmalnicoff, N.} and {Tomasi, M.}},
	title = {Convolutional neural networks on the HEALPix sphere: a pixel-based algorithm and its application to CMB data analysis},
	DOI= "10.1051/0004-6361/201935211",
	url= "https://doi.org/10.1051/0004-6361/201935211",
	journal = {A\&A},
	year = 2019,
	volume = 628,
	pages = "A129",
}
```

## License

The library is released under a MIT license. See the file LICENSE for
more information.
