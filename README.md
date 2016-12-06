# Compilation

Install dependencies (minimum version):

1. GCC (5)
2. cmake (3)
3. boost (1.58)

Initialize git submodules:

``` shell
$ git submodule init
$ git submodule update
```

Build using CMake:

``` shell
$ mkdir build
$ cd build
$ cmake ..
$ make -j `grep -c ^processor /proc/cpuinfo`
```

Add the resulting executable `nname` to your path:

``` shell
export PATH=$PATH:`pwd`/build/src/cli
```

or move `nname` to a location already on the path. Check you can
run it:

``` shell
nname --help
```
# Example Usage

See `train_conll_model.sh` and `evaluate_conll_model.sh` and the
scripts in `scripts/` for example usage.
