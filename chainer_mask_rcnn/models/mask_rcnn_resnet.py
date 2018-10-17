# Modified works:
# --------------------------------------------------------
# Copyright (c) 2017 - 2018 Kentaro Wada.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# This is modified work of FasterRCNNVGG16:
# --------------------------------------------------------
# Copyright (c) 2017 Preferred Networks, Inc.
# Licensed under The MIT License [see LICENSE for details]
# https://github.com/chainer/chainercv
# --------------------------------------------------------

import functools

import chainer
import chainer.functions as F
import chainer.links as L
import chainercv
import numpy as np

from .. import functions
from ..utils import copyparams
from .batch_normalization_to_affine import batch_normalization_to_affine_chain
from .mask_rcnn import MaskRCNN
from .region_proposal_network import RegionProposalNetwork


class MaskRCNNResNet(MaskRCNN):

    feat_stride = 16

    def __init__(self,
                 n_layers,
                 n_fg_class,
                 pretrained_model=None,
                 min_size=600,
                 max_size=1000,
                 ratios=(0.5, 1, 2),
                 anchor_scales=(4, 8, 16, 32),
                 mean=(123.152, 115.903, 103.063),
                 res_initialW=None,
                 rpn_initialW=None,
                 loc_initialW=None,
                 score_initialW=None,
                 mask_initialW=None,
                 proposal_creator_params=dict(
                     min_size=0,
                     n_test_pre_nms=6000,
                     n_test_post_nms=1000,
                 ),
                 pooling_func=functions.roi_align_2d,
                 rpn_hidden=1024,
                 roi_size=7,
                 ):
        if loc_initialW is None:
            loc_initialW = chainer.initializers.Normal(0.001)
        if score_initialW is None:
            score_initialW = chainer.initializers.Normal(0.01)
        if mask_initialW is None:
            mask_initialW = chainer.initializers.Normal(0.01)
        if rpn_initialW is None:
            rpn_initialW = chainer.initializers.Normal(0.01)
        if res_initialW is None and pretrained_model:
            res_initialW = chainer.initializers.constant.Zero()

        if pretrained_model:
            kwargs = dict(n_class=1)
        else:
            kwargs = dict(pretrained_model='imagenet')
        cls = getattr(chainercv.links, 'ResNet{:d}'.format(n_layers))
        assert isinstance(mean, tuple) and len(mean) == 3
        extractor = cls(
            arch='he', mean=np.array(mean)[:, None, None], **kwargs
        )
        extractor.pool1 = functools.partial(
            F.max_pooling_2d, ksize=3, stride=2, pad=1
        )

        res5 = chainercv.links.model.resnet.ResBlock(
            n_layer=3,
            in_channels=1024,
            mid_channels=512,
            out_channels=2048,
            stride=roi_size // 7,
            initialW=res_initialW,
            stride_first=True,
        )
        copyparams(res5, extractor.res5)

        extractor.pick = ('res2', 'res4')
        extractor.remove_unused()

        rpn = RegionProposalNetwork(
            1024, rpn_hidden,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
            initialW=rpn_initialW,
            proposal_creator_params=proposal_creator_params,
        )
        head = ResNetRoIHead(
            res5=res5,
            n_class=n_fg_class + 1,
            roi_size=roi_size,
            spatial_scale=1. / self.feat_stride,
            res_initialW=res_initialW,
            loc_initialW=loc_initialW,
            score_initialW=score_initialW,
            mask_initialW=mask_initialW,
            pooling_func=pooling_func,
        )

        super(MaskRCNNResNet, self).__init__(
            extractor,
            rpn,
            head,
            min_size=min_size,
            max_size=max_size
        )

        batch_normalization_to_affine_chain(self)

        if pretrained_model:
            chainer.serializers.load_npz(pretrained_model, self)


class ResNetRoIHead(chainer.Chain):

    mask_size = 14  # Size of the predicted mask.

    def __init__(self, res5, n_class, roi_size, spatial_scale,
                 res_initialW=None, loc_initialW=None, score_initialW=None,
                 mask_initialW=None, pooling_func=functions.roi_align_2d,
                 ):
        # n_class includes the background
        super(ResNetRoIHead, self).__init__()
        with self.init_scope():
            self.res5 = res5
            self.cls_loc = L.Linear(2048, n_class * 4, initialW=loc_initialW)
            self.score = L.Linear(2048, n_class, initialW=score_initialW)

            # 7 x 7 x 2048 -> 14 x 14 x 256
            self.deconv6 = L.Deconvolution2D(
                2048, 256, 2, stride=2, initialW=mask_initialW)
            # 14 x 14 x 256 -> 14 x 14 x 20
            n_fg_class = n_class - 1
            self.mask = L.Convolution2D(
                256, n_fg_class, 1, initialW=mask_initialW)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.pooling_func = pooling_func

    def __call__(self, x, rois, roi_indices, pred_bbox=True, pred_mask=True):
        roi_indices = roi_indices.astype(np.float32)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)
        pool = self.pooling_func(
            x,
            indices_and_rois,
            outh=self.roi_size,
            outw=self.roi_size,
            spatial_scale=self.spatial_scale,
            axes='yx',
        )

        res5 = self.res5(pool)

        roi_cls_locs = None
        roi_scores = None
        roi_masks = None

        if pred_bbox:
            pool5 = F.average_pooling_2d(res5, 7, stride=7)
            roi_cls_locs = self.cls_loc(pool5)
            roi_scores = self.score(pool5)

        if pred_mask:
            deconv6 = F.relu(self.deconv6(res5))
            roi_masks = self.mask(deconv6)

        return roi_cls_locs, roi_scores, roi_masks
