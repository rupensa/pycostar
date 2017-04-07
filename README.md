# CoStar

CO-clustering for STAR-structured data (CoStar) is a multi-view co-clustering algorithm.
Given a set of different views of the same dataset CoStar provides, at the same time, a clustering on documents and a clustering on features of each view.

This implementation is compatible with the scikit-learn (http://scikit-learn.org/) clustering interface. 


If using, cite the following paper: 

[1] Ienco, Dino, et al., 2013. Parameter-less co-clustering for star-structured heterogeneous data. Data Mining and Knowledge Discovery 26.2: 217-254

## Installation procedure

To use the algorithm, follow these simple steps:

* Install project dependencies with pip

        pip install -r requirements.txt

* Test the code running the file `costar-test.py`

        python costar-test.py

## Future works

Future works are related to:

* Improve the algorithm performances
* Improve code parallelization

## Contributions

Python implementation by [Valentina Rho](https://github.com/valentinarho).

If you want to contribute or comment, write to pensa@di.unito.it.

## Licence 

The software is released under GPL v3 licence. 
