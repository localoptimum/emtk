#!/bin/zsh

# An empty string environment variable for blas__ldflags will make
# pytensor use numpy's own dot product function instead of trying to
# use one of the blas libraries.  This might remove some of the API
# init errors associated with pytensor and blas, particularly between
# different versions.  The downside is that there is a potential speed
# decrease because of the way blas has been heavily optimised.

export PYTENSOR_FLAGS=blas__ldflags=
