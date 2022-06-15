# Check for installation prefix
if [[ -z ${FLAMINGO_PREFIX+x} ]]
then
	FLAMINGO_PREFIX=$HOME/Software
fi
export $FLAMINGO_PREFIX
mkdir -p $FLAMINGO_PREFIX

# Download & Extract OpenBLAS
OpenBLAS_VERSION="0.3.18"
cd $FLAMINGO_PREFIX
wget "https://github.com/xianyi/OpenBLAS/archive/refs/tags/v$OpenBLAS_VERSION.tar.gz"
tar -xf "v$OpenBLAS_VERSION.tar.gz"

# Build OpenBLAS
cd "OpenBLAS-$OpenBLAS_VERSION"
NUM_PAR_JOBS=4
make -j $NUM_PAR_JOBS

# Install OpenBLAS
mkdir -p "$FLAMINGO_PREFIX/openblas_install_dir"
make install PREFIX="$FLAMINGO_PREFIX/openblas_install_dir"
