
import paddle
from paddle import _C_ops
from .. import core

def _acos(*args, **kwargs):
    return _C_ops.acos(*args, **kwargs)

def _amin(*args, **kwargs):
    return _C_ops.amin(*args, **kwargs)

def _acosh(*args, **kwargs):
    return _C_ops.acosh(*args, **kwargs)

def _amax(*args, **kwargs):
    return _C_ops.amax(*args, **kwargs)

def _matmul(*args, **kwargs):
    return _C_ops.matmul(*args, **kwargs)

def _multiply(*args, **kwargs):
    return _C_ops.multiply(*args, **kwargs)

def _log2(*args, **kwargs):
    return _C_ops.log2(*args, **kwargs)

def _log10(*args, **kwargs):
    return _C_ops.log10(*args, **kwargs)

def _log1p(*args, **kwargs):
    return _C_ops.log1p(*args, **kwargs)

def _maximum(*args, **kwargs):
    return _C_ops.maximum(*args, **kwargs)

def _minimum(*args, **kwargs):
    return _C_ops.minimum(*args, **kwargs)

def _greater_than(*args, **kwargs):
    return _C_ops.greater_than(*args, **kwargs)

def _expand_as(*args, **kwargs):
    return _C_ops.expand_as(*args, **kwargs)

def _logical_and(*args, **kwargs):
    return _C_ops.logical_and(*args, **kwargs)

def _logical_or(*args, **kwargs):
    return _C_ops.logical_or(*args, **kwargs)

def _logical_xor(*args, **kwargs):
    return _C_ops.logical_xor(*args, **kwargs)

def _logical_not(*args, **kwargs):
    return _C_ops.logical_not(*args, **kwargs)

def _argmax(*args, **kwargs):
    return _C_ops.argmax(*args, **kwargs)

def _argmin(*args, **kwargs):
    return _C_ops.argmin(*args, **kwargs)

def _ceil(*args, **kwargs):
    return _C_ops.ceil(*args, **kwargs)

def _dot(*args, **kwargs):
    return _C_ops.dot(*args, **kwargs)

def _all(*args, **kwargs):
    return _C_ops.all(*args, **kwargs)

def _bmm(*args, **kwargs):
    return _C_ops.bmm(*args, **kwargs)

def _cos(*args, **kwargs):
    return _C_ops.cos(*args, **kwargs)

def _cosh(*args, **kwargs):
    return _C_ops.cosh(*args, **kwargs)

def _floor(*args, **kwargs):
    return _C_ops.floor(*args, **kwargs)

def _isfinite(*args, **kwargs):
    return _C_ops.isfinite(*args, **kwargs)

def _isinf(*args, **kwargs):
    return _C_ops.isinf(*args, **kwargs)

def _isnan(*args, **kwargs):
    return _C_ops.isnan(*args, **kwargs)

def _log(*args, **kwargs):
    return _C_ops.log(*args, **kwargs)

def _logsumexp(*args, **kwargs):
    return _C_ops.logsumexp(*args, **kwargs)

def _roll(*args, **kwargs):
    return _C_ops.roll(*args, **kwargs)

def _rsqrt(*args, **kwargs):
    return _C_ops.rsqrt(*args, **kwargs)

def _sigmoid(*args, **kwargs):
    return _C_ops.sigmoid(*args, **kwargs)

def _sign(*args, **kwargs):
    return _C_ops.sign(*args, **kwargs)

def _sin(*args, **kwargs):
    return _C_ops.sin(*args, **kwargs)

def _asin(*args, **kwargs):
    return _C_ops.asin(*args, **kwargs)

def _any(*args, **kwargs):
    return _C_ops.any(*args, **kwargs)

def _sqrt(*args, **kwargs):
    return _C_ops.sqrt(*args, **kwargs)

def _tril(*args, **kwargs):
    return _C_ops.tril(*args, **kwargs)

def _triu(*args, **kwargs):
    return _C_ops.triu(*args, **kwargs)

def _gelu(*args, **kwargs):
    return _C_ops.gelu(*args, **kwargs)

def _sum(*args, **kwargs):
    return _C_ops.sum(*args, **kwargs)

def _tanh(*args, **kwargs):
    return _C_ops.tanh(*args, **kwargs)

def _exp(*args, **kwargs):
    return _C_ops.exp(*args, **kwargs)

def _expm1(*args, **kwargs):
    return _C_ops.expm1(*args, **kwargs)

def _diagonal(*args, **kwargs):
    return _C_ops.diagonal(*args, **kwargs)

def _round(*args, **kwargs):
    return _C_ops.round(*args, **kwargs)

def _abs(*args, **kwargs):
    return _C_ops.abs(*args, **kwargs)

def _index_put(*args, **kwargs):
    return _C_ops.index_put(*args, **kwargs)

def _index_put_(*args, **kwargs):
    return _C_ops.index_put_(*args, **kwargs)

def _softplus(*args, **kwargs):
    return _C_ops.softplus(*args, **kwargs)

def _isclose(*args, **kwargs):
    return _C_ops.isclose(*args, **kwargs)

def _grid_sample(*args, **kwargs):
    return _C_ops.grid_sample(*args, **kwargs)

def _atanh(*args, **kwargs):
    return _C_ops.atanh(*args, **kwargs)

def _sinh(*args, **kwargs):
    return _C_ops.sinh(*args, **kwargs)

methods_map = [
  ('acos',_acos),
   ('amin',_amin),
   ('acosh',_acosh),
   ('amax',_amax),
   ('matmul',_matmul),
   ('multiply',_multiply),
   ('log2',_log2),
   ('log10',_log10),
   ('log1p',_log1p),
   ('maximum',_maximum),
   ('minimum',_minimum),
   ('greater_than',_greater_than),
   ('expand_as',_expand_as),
   ('logical_and',_logical_and),
   ('logical_or',_logical_or),
   ('logical_xor',_logical_xor),
   ('logical_not',_logical_not),
   ('argmax',_argmax),
   ('argmin',_argmin),
   ('ceil',_ceil),
   ('dot',_dot),
   ('all',_all),
   ('bmm',_bmm),
   ('cos',_cos),
   ('cosh',_cosh),
   ('floor',_floor),
   ('isfinite',_isfinite),
   ('isinf',_isinf),
   ('isnan',_isnan),
   ('log',_log),
   ('logsumexp',_logsumexp),
   ('roll',_roll),
   ('rsqrt',_rsqrt),
   ('sigmoid',_sigmoid),
   ('sign',_sign),
   ('sin',_sin),
   ('asin',_asin),
   ('any',_any),
   ('sqrt',_sqrt),
   ('tril',_tril),
   ('triu',_triu),
   ('sum',_sum),
   ('tanh',_tanh),
   ('exp',_exp),
   ('expm1',_expm1),
   ('diagonal',_diagonal),
   ('round',_round),
   ('abs',_abs),
   ('index_put',_index_put),
   ('index_put_',_index_put_),
   ('isclose',_isclose),
   ('atanh',_atanh),
   ('sinh',_sinh)
]


funcs_map = [
  ('acos',_acos),
   ('amin',_amin),
   ('acosh',_acosh),
   ('amax',_amax),
   ('matmul',_matmul),
   ('multiply',_multiply),
   ('log2',_log2),
   ('log10',_log10),
   ('log1p',_log1p),
   ('maximum',_maximum),
   ('minimum',_minimum),
   ('greater_than',_greater_than),
   ('expand_as',_expand_as),
   ('logical_and',_logical_and),
   ('logical_or',_logical_or),
   ('logical_xor',_logical_xor),
   ('logical_not',_logical_not),
   ('argmax',_argmax),
   ('argmin',_argmin),
   ('ceil',_ceil),
   ('dot',_dot),
   ('all',_all),
   ('bmm',_bmm),
   ('cos',_cos),
   ('cosh',_cosh),
   ('floor',_floor),
   ('isfinite',_isfinite),
   ('isinf',_isinf),
   ('isnan',_isnan),
   ('log',_log),
   ('logsumexp',_logsumexp),
   ('roll',_roll),
   ('rsqrt',_rsqrt),
   ('sigmoid',_sigmoid),
   ('sign',_sign),
   ('sin',_sin),
   ('asin',_asin),
   ('any',_any),
   ('sqrt',_sqrt),
   ('tril',_tril),
   ('triu',_triu),
   ('sum',_sum),
   ('tanh',_tanh),
   ('exp',_exp),
   ('expm1',_expm1),
   ('diagonal',_diagonal),
   ('round',_round),
   ('abs',_abs),
   ('index_put',_index_put),
   ('index_put_',_index_put_),
   ('isclose',_isclose),
   ('atanh',_atanh),
   ('sinh',_sinh)
]


nn_funcs_map = [
  ('sigmoid',_sigmoid),
   ('gelu',_gelu),
   ('tanh',_tanh),
   ('softplus',_softplus),
   ('grid_sample',_grid_sample)
]


def monkey_patch_generated_methods_for_tensor():

    # set methods for paddle.Tensor in dygraph
    local_tensor = core.eager.Tensor
    for method_name, method in methods_map:
        setattr(local_tensor, method_name, method)
        setattr(paddle.tensor, method_name, method)


    # set functions for paddle
    for method_name, method in funcs_map:
        setattr(paddle, method_name, method)


    # set functions for paddle.nn.functional
    for method_name, method in nn_funcs_map:
        setattr(paddle.nn.functional, method_name, method)
