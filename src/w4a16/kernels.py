"""w4a16 GEMV kernels for A100 (sm_80): weight-only int4, fp16 activations.

The CUDA source (kernels/w4a16_kernels.cu) is JIT-compiled with
torch.utils.cpp_extension.load() on first use. No setup.py build step (see README).

This module exposes the full optimization ladder plus the shipped Marlin kernel.
Two packings are used and MUST be matched to the kernel:
  - pack_int4             (simple hi/lo)             -> byte, u32, u32_v4, champ, h2f32, uint4
  - pack_int4_interleaved (slot [0,2,4,6,1,3,5,7])   -> marlin (SHIPPED)

The dequant reference (dequant_ref) works from the quantized codes directly and
is therefore packing-independent, every kernel is checked against it.
"""
import functools
import os
import torch
import torch.nn as nn

_CU_SOURCE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "kernels", "w4a16_kernels.cu")
)


@functools.lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load
    if not os.path.exists(_CU_SOURCE):
        raise FileNotFoundError(f"CUDA source not found at {_CU_SOURCE}")
    return load(name="w4a16_a100", sources=[_CU_SOURCE], verbose=False)


# ---- ladder wrappers (simple packing) ----
def matvec_byte(packed, scale, vec):   return _ext().byte_cuda(packed, scale, vec)
def matvec_u32(packed, scale, vec):    return _ext().u32_cuda(packed, scale, vec)
def matvec_u32_v4(packed, scale, vec): return _ext().u32_v4_cuda(packed, scale, vec)
def matvec_champ(packed, scale, vec):  return _ext().champ_cuda(packed, scale, vec)
def matvec_uint4(packed, scale, vec):  return _ext().uint4_cuda(packed, scale, vec)   # rejected
def matvec_h2f32(packed, scale, vec):  return _ext().h2f32_cuda(packed, scale, vec)   # lossy opt-in
# ---- shipped kernel (interleaved packing) ----
def w4a16_gemv(packed, scale, vec):    return _ext().marlin_cuda(packed, scale, vec)
# ---- fp16 dense reference ----
def matvec_fp16(mat, vec):             return _ext().fp16_cuda(mat, vec)


# ---- int4 quantization ----
def quantize_int4_symmetric(w):
    scale = w.abs().max(dim=1, keepdim=True).values / 7.0
    q = torch.round(w / scale).clamp(-8, 7).to(torch.int8)
    return q, scale.squeeze(1)


# ---- packings ----
def pack_int4(q):
    """Simple hi/lo packing: byte c = (col 2c << 4) | col 2c+1."""
    q_u = (q + 8).to(torch.uint8)
    rows, cols = q_u.shape
    q_u = q_u.view(rows, cols // 2, 2)
    packed = (q_u[:, :, 0] << 4) | q_u[:, :, 1]
    return packed.to(torch.uint8)


_INTERLEAVE = [0, 2, 4, 6, 1, 3, 5, 7]  # slot i within each 8-col group holds this column


def pack_int4_interleaved(q):
    """Interleaved packing for the marlin kernel. Within each group of 8 columns,
    nibble slot i (bit offset 4*i of the little-endian uint32) holds column
    _INTERLEAVE[i], so four lop3 ops decode nibbles back into column order."""
    rows, cols = q.shape
    assert cols % 8 == 0, "interleaved packing requires cols % 8 == 0"
    qu = (q + 8).to(torch.int64)
    g = cols // 8
    qg = qu.view(rows, g, 8)
    val = torch.zeros(rows, g, dtype=torch.int64, device=q.device)
    for slot, col in enumerate(_INTERLEAVE):
        val |= (qg[:, :, col] & 0xF) << (4 * slot)
    b0 = (val >> 0) & 0xFF; b1 = (val >> 8) & 0xFF
    b2 = (val >> 16) & 0xFF; b3 = (val >> 24) & 0xFF
    packed = torch.stack([b0, b1, b2, b3], dim=2).reshape(rows, g * 4)
    return packed.to(torch.uint8)


def dequant_ref(q, scale, vec):
    """Packing-independent reference: dequantize codes -> dense fp32 matvec."""
    return (q.float() * scale.unsqueeze(1)) @ vec.float()


# ---- nn.Linear drop-in: marlin kernel on decode (seq==1), dense fp16 on prefill ----
class W4A16Linear(nn.Module):
    def __init__(self, weight_fp16, bias, packed, scale):
        super().__init__()
        self.register_buffer("dense_weight", weight_fp16)
        self.register_buffer("packed", packed)      # interleaved packing (marlin)
        self.register_buffer("scale", scale)
        self.bias = bias

    def forward(self, x):
        orig_shape = x.shape
        in_features = self.dense_weight.shape[1]
        x_flat = x.reshape(-1, in_features)
        if x_flat.shape[0] == 1:
            vec = x_flat[0].half().contiguous()
            out = w4a16_gemv(self.packed, self.scale, vec)   # marlin
        else:
            out = x_flat.float() @ self.dense_weight.float().T
        if self.bias is not None:
            out = out + self.bias.float()
        return out.to(x.dtype).view(*orig_shape[:-1], -1)


def replace_linear_with_w4a16(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            w = child.weight.data.float()
            q, scale = quantize_int4_symmetric(w)
            packed = pack_int4_interleaved(q).cuda()      # marlin packing
            new_layer = W4A16Linear(child.weight.data.half(), child.bias, packed, scale.float().cuda())
            setattr(module, name, new_layer)
        else:
            replace_linear_with_w4a16(child)
    return module
