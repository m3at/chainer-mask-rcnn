import chainer
import chainer.links as L

from .. import links


def batch_normalization_to_affine_link(bn):
    assert isinstance(bn, L.BatchNormalization)
    channels = bn.gamma.size
    bn_mean = bn.avg_mean
    bn_var = bn.avg_var
    scale = bn.gamma.data
    bias = bn.beta.data
    xp = chainer.cuda.get_array_module(bn_var)
    std = xp.sqrt(bn_var + 1e-5)
    new_scale = scale / std
    new_bias = bias - bn_mean * new_scale
    affine = links.AffineChannel2D(channels)
    affine.W.data[:] = new_scale[:]
    affine.b.data[:] = new_bias[:]
    return affine


def batch_normalization_to_affine(chain):
    assert isinstance(chain, chainer.Chain)
    for name, link in chain.namedlinks():
        if not isinstance(link, L.BatchNormalization):
            continue
        for key in name.split('/')[:-1]:
            if key == '':
                parent = chain
            else:
                parent = getattr(parent, key)
        key = name.split('/')[-1]
        delattr(parent, key)
        link_affine = batch_normalization_to_affine_link(link)
        parent.add_link(key, link_affine)
