# pyDenStream

![master](https://github.com/MrParosk/pyDenStream/workflows/master/badge.svg?branch=master) [![codecov](https://codecov.io/gh/MrParosk/pyDenStream/branch/master/graph/badge.svg?token=HEKMVIH5WO)](https://codecov.io/gh/MrParosk/pyDenStream)

Implementation of the algorithm [Density-Based Clustering over an Evolving Data Stream with Noise](https://archive.siam.org/meetings/sdm06/proceedings/030caof.pdf) in Python.

## Installation

```Shell
pip install denstream
```

## Example usage

```python
import numpy as np
from denstream import DenStream

# Model parameters
eps = 0.3
lambd = 0.1
beta = 0.2
mu = 10
min_samples = 1

model = DenStream(eps, beta, mu, lambd, min_samples)

x = np.array([[1, 2]])
t = 0

model.partial_fit(x, t)
```

## In depth example

A more in depth example of how to use this package is included in *examples/user_guide.ipynb*.
