# class taylor_softmax(nn.Module):
#     def __init__(self, dim):
#         super(taylor_softmax, self).__init__():
#         self.dim = dim

#     def forward(self, input: Tensor): -> Tensor:
#         return

def _get_softmax_dim(name, ndim, stacklevel):
    # type: (str, int, int) -> int
    warnings.warn("Implicit dimension choice for {} has been deprecated. "
                  "Change the call to include dim=X as an argument.".format(name), stacklevel=stacklevel)
    if ndim == 0 or ndim == 1 or ndim == 3:
        ret = 0
    else:
        ret = 1
    return ret


def taylor_softmax(input, dim=None, _stacklevel=3, dtype=None):
    # if not torch.jit.is_scripting():
    #     if type(input) is not Tensor and has_torch_function((input,)):
    #         return handle_torch_function(
    #             softmax, (input,), input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
    if dim is None:
        dim = _get_softmax_dim('softmax', input.dim(), _stacklevel)
    if dtype is None:
        ret = input.softmax(dim)
    else:
        ret = input.softmax(dim, dtype=dtype)
    return ret