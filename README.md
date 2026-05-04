# LeetGPU Challenges

LeetGPU 编程练习题解。

## 题目列表

### Easy

| 题目 | CUDA | Triton | PyTorch |
|------|------|--------|---------|
| Vector Addition | [.cu](easy/vector-addition/vector-addition.cu) | [.py](easy/vector-addition/vector-addition.triton.py) | [.py](easy/vector-addition/vector-addition.pytorch.py) |
| Matrix Multiplication | [.cu](easy/matrix-multiplication/matrix-multiplication.cu) | [.py](easy/matrix-multiplication/matrix_multiplication.triton.py) | [.py](easy/matrix-multiplication/matrix_multiplication.pytorch.py) |
| Matrix Transpose | [.cu](easy/matrix-transpose/matrix_transpose.cu) | [.py](easy/matrix-transpose/matrix_transpose.triton.py) | [.py](easy/matrix-transpose/matrix_transpose.pytorch.py) |
| Color Inversion | [.cu](easy/color-inversion/color_inversion.cu) | [.py](easy/color-inversion/color_inversion.triton.py) | [.py](easy/color-inversion/color_inversion.pytorch.py) |
| Matrix Addition | [.cu](easy/matrix-addition/matrix_addition.cu) | [.py](easy/matrix-addition/matrix_addition.triton.py) | [.py](easy/matrix-addition/matrix_addition.pytorch.py) |
| 1D Convolution | [.cu](easy/1d-convolution/1d_convolution.cu) | [.py](easy/1d-convolution/1d_convolution.triton.py) | [.py](easy/1d-convolution/1d_convolution.pytorch.py) |
| Reverse Array | [.cu](easy/reverse-array/reverse_array.cu) | [.py](easy/reverse-array/reverse_array.triton.py) | [.py](easy/reverse-array/reverse_array.pytorch.py) |
| ReLU | [.cu](easy/relu/relu.cu) | [.py](easy/relu/relu.triton.py) | [.py](easy/relu/relu.pytorch.py) |
| Leaky ReLU | [.cu](easy/leaky-relu/leaky_relu.cu) | [.py](easy/leaky-relu/leaky_relu.triton.py) | [.py](easy/leaky-relu/leaky_relu.pytorch.py) |
| Simple Inference | - | - | [.py](easy/simple-inference/simple_inference.pytorch.py) |
| Sigmoid | [.cu](easy/sigmoid/sigmoid.cu) | [.py](easy/sigmoid/sigmoid.triton.py) | [.py](easy/sigmoid/sigmoid.pytorch.py) |
| SiLU | [.cu](easy/silu/silu.cu) | [.py](easy/silu/silu.triton.py) | [.py](easy/silu/silu.pytorch.py) |
| SwiGLU | [.cu](easy/swiglu/swiglu.cu) | [.py](easy/swiglu/swiglu.triton.py) | [.py](easy/swiglu/swiglu.pytorch.py) |
| GEGLU | [.cu](easy/geglu/geglu.cu) | [.py](easy/geglu/geglu.triton.py) | [.py](easy/geglu/geglu.pytorch.py) |
| Interleave | [.cu](easy/interleave/interleave.cu) | [.py](easy/interleave/interleave.triton.py) | [.py](easy/interleave/interleave.pytorch.py) |
| Matrix Copy | [.cu](easy/matrix-copy/matrix_copy.cu) | [.py](easy/matrix-copy/matrix_copy.triton.py) | [.py](easy/matrix-copy/matrix_copy.pytorch.py) |
| Value Clipping | [.cu](easy/value-clipping/value_clipping.cu) | [.py](easy/value-clipping/value_clipping.triton.py) | [.py](easy/value-clipping/value_clipping.pytorch.py) |

### Medium

| 题目 | CUDA | Triton | PyTorch |
|------|------|--------|---------|
| Reduction | [.cu](medium/reduction/reduction.cu) | [.py](medium/reduction/reduction.triton.py) | [.py](medium/reduction/reduction.pytorch.py) |
| Dot Product | [.cu](medium/dot-product/dot_product.cu) | [.py](medium/dot-product/dot_product.triton.py) | [.py](medium/dot-product/dot_product.pytorch.py) |
| FP16 Dot Product | [.cu](medium/fp16_dot_product/fp16_dot_product.cu) | [.py](medium/fp16_dot_product/fp16_dot_product.triton.py) | [.py](medium/fp16_dot_product/fp16_dot_product.pytorch.py) |
| Softmax | [.cu](medium/softmax/softmax.cu) | [.py](medium/softmax/softmax.triton.py) | [.py](medium/softmax/softmax.pytorch.py) |
| RMS Normalization | [.cu](medium/rms_normalization/rms_normalization.cu) | [.py](medium/rms_normalization/rms_normalization.triton.py) | [.py](medium/rms_normalization/rms_normalization.pytorch.py) |

> Note: Softmax 的 CUDA/Triton 当前为单 block / 单 program 学习版，`N = 500000` 的多 block 高性能实现尚未完成。
