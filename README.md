**Project of Machine Learning for Geometric data**

This project aims to try different architectures to learn linearly invariant em--
bedding to solve the point cloud matching problem. The pipeline of this
the method consists of learning both basis and descriptors in a supervised setting,
to substitute the Laplace Beltrami eigenfunctions and axiomatically chosen de-
scriptures.
In addition, we consider a refinement method, designed to increase the
accuracy of the maps estimated by this data-driven approach. This refinement
algorithm performs an iterative upsampling in a ZoomOut fashion, promoting
the bijectivity of the correspondence.

In this repository, you can find a simple evaluation test for the model based on the SMAL dataset, a folder with utility functions, and 4 folders in which there is the implementation
of the models developed using the 4 feature extractors considered. Each folder contains a data loader, four training files, and a model definition file.

