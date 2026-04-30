# LLM Infra Challenge Plan

## Goal

从 `AlphaGPU/leetgpu-challenges` 中筛选与 LLM inference、serving、kernel infra 最相关的题目，按 `challenges/easy`、`challenges/medium`、`challenges/hard` 分类，并组织成可执行的刷题路线。

筛选优先级：

- 高优先级：GEMM、batched matmul、attention、KV cache、RoPE、RMSNorm、SwiGLU、quantization、sampling、MoE、LoRA。
- 中优先级：reduction、prefix sum、top-k、stream compaction、数据布局变换。
- 低优先级：图像处理、图算法、传统 CV 算子、与 LLM serving 迁移价值弱的通用算法题。

## Easy

### 必做

| 题目 | 价值 |
|---|---|
| `21_relu` | 基础 activation，练 elementwise kernel |
| `23_leaky_relu` | 基础 activation，练分支和向量化 |
| `68_sigmoid` | gate 和 activation 基础 |
| `52_silu` | LLM MLP 常用激活 |
| `54_swiglu` | Llama/Qwen 类 MLP 核心 |
| `65_geglu` | gated activation，与 SwiGLU 同类 |
| `63_interleave` | 数据布局变换，对 QKV/KV cache layout 有帮助 |
| `2_matrix_multiplication` | matmul 入门，后续 GEMM 和 attention 的基础 |
| `41_simple_inference` | 最贴近端到端推理的小题 |

### 选做

| 题目 | 价值 |
|---|---|
| `31_matrix_copy` | memory bandwidth、coalescing 基础 |
| `62_value_clipping` | logits 和数值稳定性相关，优先级较低 |
| `8_matrix_addition` | elementwise 基础 |

### 可跳过

- `color_inversion`
- `rgb_to_grayscale`
- `rainbow_table`
- `reverse_array`

这些题可以练 CUDA 基础，但和 LLM infra 的直接关联较弱。

## Medium

### 必做主线

| 题目 | 价值 |
|---|---|
| `4_reduction` | softmax、norm、top-k 的基础 |
| `58_fp16_dot_product` | reduction 和 fp16 数值处理 |
| `5_softmax` | attention 和 sampling 基础 |
| `50_rms_normalization` | 现代 LLM 常用 normalization |
| `22_gemm` | LLM 最核心算子 |
| `57_fp16_batched_matmul` | batched inference 和 attention projection |
| `6_softmax_attention` | attention 基础版 |
| `61_rope_embedding` | Llama/Qwen 类位置编码 |
| `80_grouped_query_attention` | GQA/MQA serving 核心 |
| `64_weight_dequantization` | 量化权重推理基础 |
| `32_int8_quantized_matmul` | INT8 推理核心 |
| `81_int4_matmul` | INT4/低比特推理核心 |
| `96_int8_kv_cache_attention` | KV cache + quantized attention，贴近推理系统 |
| `85_lora_linear` | LoRA adapter serving |
| `67_moe_topk_gating` | MoE routing |
| `60_top_p_sampling` | decode sampling 直接相关 |

### 支撑能力

| 题目 | 价值 |
|---|---|
| `16_prefix_sum` | compaction、sampling、scan 基础 |
| `17_dot_product` | reduction 基础 |
| `29_top_k_selection` | sampling 和 MoE routing 相关 |
| `30_batched_matrix_multiplication` | batch 维度处理 |
| `70_segmented_prefix_sum` | ragged batch 和 variable length 相关 |
| `72_stream_compaction` | filter、动态 token 处理相关 |
| `75_sparse_matrix_dense_matrix_multiplication` | sparse、MoE、稀疏算子可选 |
| `82_linear_recurrence` | state-space 和 scan 类模型可选 |

### 训练相关，低优先级

| 题目 | 价值 |
|---|---|
| `25_categorical_cross_entropy_loss` | 训练 loss |
| `27_mean_squared_error` | 通用 loss |
| `40_batch_normalization` | LLM 不常用，偏 CV/传统 DL |

### 可跳过

卷积、pooling、图像、FFT、最短路、Monte Carlo 等题目除非目标是系统练 CUDA 通用能力，否则不放进 LLM infra 主线。

## Hard

### 必做

| 题目 | 价值 |
|---|---|
| `12_multi_head_attention` | 标准 Transformer attention |
| `53_casual_attention` | repo 名称是 casual，语义应是 causal attention；decode/prefill 核心 |
| `56_linear_attention` | attention 变体，长上下文方向 |
| `59_sliding_window_attn` | 长上下文和局部 attention |
| `74_gpt2_block` | 综合题，把 norm、attention、MLP 串起来 |

### 选做

| 题目 | 价值 |
|---|---|
| `15_sorting` | top-k、sampling、MoE 可迁移 |
| `36_radix_sort` | 高性能排序基础 |
| `39_Fast_Fourier_transform` | 不直接 LLM，但能练复杂 memory pattern |

### 可跳过

- `14_multi_agent_sim`
- `20_kmeans_clustering`
- `46_bfs_shortest_path`
- `73_all_pairs_shortest_paths`

这些题和 LLM infra 主线关联较弱。

## Execution Plan

### Phase 1: Easy 打底

目标：建立 memory copy、elementwise、activation、layout、简单 matmul 的基础直觉。先做被后续激活函数依赖的基础题，再做 gated activation。

1. `31_matrix_copy`
2. `62_value_clipping`
3. `68_sigmoid`
4. `52_silu`
5. `54_swiglu`
6. `65_geglu`
7. `63_interleave`
8. `2_matrix_multiplication`
9. `41_simple_inference`

### Phase 2: Medium 基础核心

目标：掌握后续 attention、norm、quantization 依赖的基础 kernel 模式。先做 reduction/dot，再做 softmax/norm 和 GEMM 系列。

1. `4_reduction`
2. `17_dot_product`
3. `5_softmax`
4. `50_rms_normalization`
5. `22_gemm`
6. `30_batched_matrix_multiplication`
7. `57_fp16_batched_matmul`
8. `58_fp16_dot_product`

### Phase 3: Attention 主线

目标：覆盖基础 attention、MHA、causal mask、RoPE、GQA、长上下文相关核心路径。KV cache 的量化版本放到量化阶段之后。

1. `6_softmax_attention`
2. `12_multi_head_attention`
3. `53_casual_attention`
4. `61_rope_embedding`
5. `80_grouped_query_attention`
6. `59_sliding_window_attn`
7. `56_linear_attention`

### Phase 4: Serving 和 Decode

目标：补齐 sampling、动态 token、ragged batch 相关能力。Top-p 依赖排序/选择和累计概率，放在 prefix/top-k 之后。

1. `16_prefix_sum`
2. `70_segmented_prefix_sum`
3. `29_top_k_selection`
4. `15_sorting`
5. `60_top_p_sampling`
6. `72_stream_compaction`

### Phase 5: Quantization、Adapter、MoE

目标：覆盖真实推理系统中的低比特、adapter 和 MoE 路径。先做 dequant 和量化 matmul，再做依赖 attention/quantization 的 KV cache attention。

1. `64_weight_dequantization`
2. `32_int8_quantized_matmul`
3. `81_int4_matmul`
4. `96_int8_kv_cache_attention`
5. `85_lora_linear`
6. `67_moe_topk_gating`

### Phase 6: 综合收口

目标：用 block 级题目把前面的核心算子串起来。

1. `74_gpt2_block`

## Minimal Track

如果时间有限，优先完成下面题目，形成 LLM inference kernel 的最小闭环。这里把概念依赖也列进来，避免先做 `SwiGLU` 却没做 `Sigmoid/SiLU`。

1. `31_matrix_copy`
2. `68_sigmoid`
3. `52_silu`
4. `54_swiglu`
5. `2_matrix_multiplication`
6. `4_reduction`
7. `5_softmax`
8. `50_rms_normalization`
9. `22_gemm`
10. `57_fp16_batched_matmul`
11. `6_softmax_attention`
12. `53_casual_attention`
13. `61_rope_embedding`
14. `80_grouped_query_attention`
15. `64_weight_dequantization`
16. `96_int8_kv_cache_attention`

## Dependency Notes

- `68_sigmoid` should precede `52_silu`, because SiLU is `x * sigmoid(x)`.
- `52_silu` should precede `54_swiglu`, because SwiGLU applies SiLU to the gate branch.
- `65_geglu` should be learned alongside or after `54_swiglu`; it uses the same `[x, gate]` split pattern but replaces SiLU with GELU.
- `4_reduction` should precede `5_softmax` and `50_rms_normalization`, because both require reductions.
- `5_softmax` should precede attention and sampling tasks.
- `22_gemm` should precede batched/quantized/adapter linear variants.
- `16_prefix_sum` and `29_top_k_selection` should precede `60_top_p_sampling`.
- `64_weight_dequantization`, `32_int8_quantized_matmul`, and attention basics should precede `96_int8_kv_cache_attention`.
- `74_gpt2_block` should stay last because it combines norm, attention, projection, and MLP concepts.

## Suggested Cadence

- 每题先读题目描述和 reference，再自己写 CUDA/Triton 实现。
- 每题记录三件事：kernel 设计、性能瓶颈、下一版优化点。
- 每个 phase 完成后做一次复盘，重点比较 memory bandwidth、occupancy、shared memory、register pressure、numeric stability。
- 对 attention、GEMM、quantization 题目至少做两版：先正确，再优化。
