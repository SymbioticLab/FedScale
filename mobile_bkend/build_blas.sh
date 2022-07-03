# Check for installation prefix
if [[ -z ${SWAN_PREFIX+x} ]]
then
	SWAN_PREFIX=$HOME/Software
fi
export $SWAN_PREFIX
mkdir -p $SWAN_PREFIX

# Download & Extract OpenBLAS
OpenBLAS_VERSION="0.3.18"
cd $SWAN_PREFIX
wget "https://github.com/xianyi/OpenBLAS/archive/refs/tags/v$OpenBLAS_VERSION.tar.gz"
tar -xf "v$OpenBLAS_VERSION.tar.gz"

# Build OpenBLAS
cd "OpenBLAS-$OpenBLAS_VERSION"
NUM_PAR_JOBS=4
make -j $NUM_PAR_JOBS

# Install OpenBLAS
mkdir -p "$SWAN_PREFIX/openblas_install_dir"
make install PREFIX="$SWAN_PREFIX/openblas_install_dir"
