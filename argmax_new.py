import torch
import triton
import triton.language as tl

@triton.jit
def kernel0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    _tmp1_index = tl.zeros([XBLOCK, RBLOCK], tl.int32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        tmp0 = tl.load(in_ptr0 + rindex, rmask).to(tl.float32)  # 移除了 .evict_last
        _tmp1_index = tl.where(xmask & rmask & (_tmp1 < tmp0), rindex, _tmp1_index)
        _tmp1 = tl.where(xmask & rmask & (_tmp1 < tmp0), tmp0, _tmp1)
    _tmp1_index_reduce = tl.reshape(tl.argmax(_tmp1, 1), [XBLOCK, 1]).to(tl.int32)
    _tmp1_index_mask = (tl.arange(0, RBLOCK)[None, :] == _tmp1_index_reduce)
    outptr_index = tl.arange(0, XBLOCK)[:, None] + tl.zeros((RBLOCK,), tl.int32)[None, :]
    tl.store(out_ptr0 + outptr_index, value=_tmp1_index, mask=_tmp1_index_mask & xmask)

def call():
    x = torch.randn((1, 8192), device="cuda")
    y = torch.zeros((1, 128), device="cuda")  # 修改了 y 的形状以匹配 RBLOCK
    kernel0[(1,)](x, y, 1, 8192, XBLOCK=1, RBLOCK=128)
    print(torch.argmax(x))
    print(y)

call()