# Clone PyTorch
cd $FLAMINGO_PREFIX
git clone https://github.com/pytorch/pytorch
cd pytorch
git checkout v1.8.1
git submodule sync
git submodule update --init --recursive --jobs 0

# Build PyTorch
# NOTE: Ensure that the build identifies the OpenBLAS installation
cd "$FLAMINGO_PREFIX/pytorch"
export MAX_JOBS=4
export USE_NUMPY=1
export BLAS=OpenBLAS
export OpenBLAS_HOME=$FLAMINGO_PREFIX/openblas_install_dir
python setup.py bdist_wheel

# Install PyTorch
sudo pip install "$FLAMINGO_PREFIX/pytorch/dist/torch-1.8.0a0+56b43f4-cp38-cp38-linux_aarch64.whl"
