import torch

__all__ = ['PointTensor']

from .base_utils import clone, to_device


class PointTensor:
    def __init__(self, feat, coords, idx_query=None, weights=None):
        self.F = feat
        self.C = coords
        self.idx_query = idx_query if idx_query is not None else {}
        self.weights = weights if weights is not None else {}
        self.additional_features = {}
        self.additional_features['idx_query'] = {}
        self.additional_features['counts'] = {}

    def detach(self):
        assert type(self.F) == torch.Tensor
        assert type(self.C) == torch.Tensor
        self.F = self.F.detach()
        self.C = self.C.detach()
        return self

    def to(self, device, non_blocking=True):
        pt = self.clone()
        assert type(pt.F) == torch.Tensor
        assert type(pt.C) == torch.Tensor
        pt.F = to_device(pt.F, device=device, non_blocking=non_blocking)
        pt.C = to_device(pt.C, device=device, non_blocking=non_blocking)
        pt.idx_query = to_device(pt.idx_query, device=device, non_blocking=non_blocking)
        pt.weights = to_device(pt.weights, device=device, non_blocking=non_blocking)
        pt.additional_features = to_device(pt.additional_features, device=device, non_blocking=non_blocking)
        return pt

    def cuda(self, non_blocking=True):
        return self.to(device='cuda', non_blocking=non_blocking)

    def cpu(self):
        return self.to('cpu')

    def __add__(self, other):
        tensor = PointTensor(self.F + other.F, self.C, self.idx_query,
                             self.weights)
        tensor.additional_features = self.additional_features
        return tensor

    @property
    def device(self):
        return self.F.device

    def clone(self):
        F = clone(self.F)
        C = clone(self.C)
        idx_query = clone(self.idx_query)
        weights = clone(self.weights)
        additional_features = clone(self.additional_features)
        pt = PointTensor(F, C, idx_query, weights)
        pt.additional_features = additional_features
        return pt
