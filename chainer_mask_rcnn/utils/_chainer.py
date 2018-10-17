import copy

import chainer
from chainer.backends import cuda


def copyparams_link(dst_link, src_link, copy_persistent=True):
    src = src_link.__dict__
    dst = dst_link.__dict__
    for name in dst_link._params:
        dst[name].copydata(src[name])
    if copy_persistent:
        array_types = chainer.get_array_types()
        for name in dst_link._persistent:
            d = dst[name]
            s = src[name]
            if isinstance(d, array_types) and isinstance(s, array_types):
                xp = cuda.get_array_module(d)
                xp.copyto(d, s)
            else:
                dst[name] = copy.deepcopy(s)


def copyparams(dst_chain, src_chain, copy_persistent=True):
    for dst_link in dst_chain.children():
        src_link = getattr(src_chain, dst_link.name)
        if isinstance(src_link, chainer.Chain):
            copyparams(dst_link, src_link, copy_persistent)
        else:
            assert isinstance(src_link, chainer.Link)
            copyparams_link(dst_link, src_link, copy_persistent)
