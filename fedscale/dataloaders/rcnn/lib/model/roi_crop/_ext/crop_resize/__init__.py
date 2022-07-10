
from torch.utils.ffi import _wrap_function

from ._crop_resize import ffi as _ffi
from ._crop_resize import lib as _lib

__all__ = []


def _import_symbols(locals):
    for symbol in dir(_lib):
        fn = getattr(_lib, symbol)
        locals[symbol] = _wrap_function(fn, _ffi)
        __all__.append(symbol)


_import_symbols(locals())
