# Code for reproducing Ecker et al, *ICLR 2019*

This folder contains the code to reproduce the models and results from the following paper:

Alexander S. Ecker, Fabian H. Sinz, Emmanouil Froudarakis, Paul G. Fahey, Santiago A. Cadena, Edgar Y. Walker, Erick Cobos, Jacob Reimer, Andreas S. Tolias, Matthias Bethge: A rotation-equivariant convolutional neural network model of primary visual cortex. *International Conference on Learning Representations (ICLR 2019)*. https://openreview.net/forum?id=H1fU8iAqKX.


## Reproducing models

If you're just interested in reproducing the various mdoels and baselines described in the paper, check out the Jupyter notebook [models](models.ipynb).
It contains all the relevant models ready-to-use.
All code necessary to construct, train and evaluate the models is contained in the module [cnn_sys_ident.architectures](../../cnn_sys_ident/architectures).


## Reproducing experiments and figures

The Jupyter notebooks in this folder contain the code we used to analyze the results and generate the figrues in the paper.
Note, however, that they won't run out-of-the-box, as they depend on a MySQL database and a data management tool ([DataJoint](https://datajoint.io)) that we use to keep track of our experiments.
Therefore, reproducing the experiments is a bit mode involved than simply running a script, as it requires setting up DataJoint and the MySQL server.
If you're interested in going that route, don't hesitate to touch base; we're happy to help.
