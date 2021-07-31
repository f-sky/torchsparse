import torch.nn.functional as F
import torch

__all__ = ['SparseTensor']

from .base_utils import clone, to_device


class SparseTensor:
    def __init__(self, feats, coords, stride=1):
        self.F = feats
        self.C = coords
        self.s = stride
        self.coord_maps = {}
        self.kernel_maps = {}

    def check(self):
        if self.s not in self.coord_maps:
            self.coord_maps[self.s] = self.C

    def detach(self):
        assert type(self.F) == torch.Tensor
        assert type(self.C) == torch.Tensor
        self.F = self.F.detach()
        self.C = self.C.detach()
        return self

    def to(self, device, non_blocking=True):
        st = self.clone()
        assert type(st.F) == torch.Tensor
        assert type(st.C) == torch.Tensor
        st.F = to_device(st.F, device)
        st.C = to_device(st.C, device)
        st.s = to_device(st.s, device)
        st.kernel_maps = to_device(st.kernel_maps, device)
        st.coord_maps = to_device(st.coord_maps, device)
        return st

    def cuda(self, non_blocking=True):
        return self.to(device='cuda', non_blocking=non_blocking)

    def cpu(self, non_blocking=True):
        return self.to(device='cpu', non_blocking=non_blocking)

    def __add__(self, other):
        tensor = SparseTensor(self.F + other.F, self.C, self.s)
        tensor.coord_maps = self.coord_maps
        tensor.kernel_maps = self.kernel_maps
        return tensor

    @property
    def shape(self):
        return self.F.shape

    @property
    def coords(self):
        if isinstance(self.C, torch.Tensor):
            C = self.C.float()
        else:
            C = self.C
        return C * self.s

    @property
    def device(self):
        return self.F.device

    def clone(self):
        F = clone(self.F)
        C = clone(self.C)
        s = clone(self.s)
        coord_maps = clone(self.coord_maps)
        kernel_maps = clone(self.kernel_maps)
        st = SparseTensor(F, C, s)
        st.coord_maps = coord_maps
        st.kernel_maps = kernel_maps
        return st

    @property
    def is_cuda(self):
        return self.F.is_cuda

    def dim(self):
        return 2
