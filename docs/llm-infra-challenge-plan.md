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
| `2_matrix_multiplication` | matmul 入门，后续 GEMM 和 attention 的基础 |
| `21_relu` | 基础 activation，练 elementwise kernel |
| `23_leaky_relu` | 基础 activation，练分支和向量化 |
| `41_simple_inference` | 最贴近端到端推理的小题 |
| `52_silu` | LLM MLP 常用激活 |
| `54_swiglu` | Llama/Qwen 类 MLP 核心 |
| `65_geglu` | gated activation，与 SwiGLU 同类 |
| `68_sigmoid` | gate 和 activation 基础 |
| `63_interleave` | 数据布局变换，对 QKV/KV cache layout 有帮助 |

### 选做

| 题目 | 价值 |
|---|---|
| `31_matrix_copy` | memory bandwidth、coalescing 基础 |
| `8_matrix_addition` | elementwise 基础 |
| `62_value_clipping` | logits 和数值稳定性相关，优先级较低 |

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
| `22_gemm` | LLM 最核心算子 |
| `57_fp16_batched_matmul` | batched inference 和 attention projection |
| `58_fp16_dot_product` | reduction 和 fp16 数值处理 |
| `5_softmax` | attention 和 sampling 基础 |
| `6_softmax_attention` | attention 基础版 |
| `50_rms_normalization` | 现代 LLM 常用 normalization |
| `60_top_p_sampling` | decode sampling 直接相关 |
| `61_rope_embedding` | Llama/Qwen 类位置编码 |
| `64_weight_dequantization` | 量化权重推理基础 |
| `32_int8_quantized_matmul` | INT8 推理核心 |
| `81_int4_matmul` | INT4/低比特推理核心 |
| `67_moe_topk_gating` | MoE routing |
| `80_grouped_query_attention` | GQA/MQA serving 核心 |
| `85_lora_linear` | LoRA adapter serving |
| `96_int8_kv_cache_attention` | KV cache + quantized attention，贴近推理系统 |

### 支撑能力

| 题目 | 价值 |
|---|---|
| `4_reduction` | softmax、norm、top-k 的基础 |
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

目标：建立 elementwise、简单 matmul、activation、layout 的基础直觉。

1. `2_matrix_multiplication`
2. `31_matrix_copy`
3. `52_silu`
4. `68_sigmoid`
5. `54_swiglu`
6. `65_geglu`
7. `41_simple_inference`

### Phase 2: Medium 基础核心

目标：掌握后续 attention、norm、quantization 依赖的基础 kernel 模式。

1. `4_reduction`
2. `17_dot_product`
3. `5_softmax`
4. `22_gemm`
5. `57_fp16_batched_matmul`
6. `50_rms_normalization`

### Phase 3: Attention 主线

目标：覆盖 prefill、decode、GQA、KV cache、长上下文相关核心路径。

1. `61_rope_embedding`
2. `6_softmax_attention`
3. `12_multi_head_attention`
4. `53_casual_attention`
5. `80_grouped_query_attention`
6. `59_sliding_window_attn`
7. `96_int8_kv_cache_attention`

### Phase 4: Serving 和 Decode

目标：补齐 sampling、动态 token、ragged batch 相关能力。

1. `29_top_k_selection`
2. `60_top_p_sampling`
3. `16_prefix_sum`
4. `70_segmented_prefix_sum`
5. `72_stream_compaction`

### Phase 5: Quantization、Adapter、MoE

目标：覆盖真实推理系统中的低比特、adapter 和 MoE 路径。

1. `64_weight_dequantization`
2. `32_int8_quantized_matmul`
3. `81_int4_matmul`
4. `85_lora_linear`
5. `67_moe_topk_gating`

### Phase 6: 综合收口

目标：用 block 级题目把前面的核心算子串起来。

1. `74_gpt2_block`

## Minimal Track

如果时间有限，优先完成下面 12 个题，形成 LLM inference kernel 的最小闭环：

1. `2_matrix_multiplication`
2. `22_gemm`
3. `57_fp16_batched_matmul`
4. `5_softmax`
5. `50_rms_normalization`
6. `54_swiglu`
7. `61_rope_embedding`
8. `6_softmax_attention`
9. `53_casual_attention`
10. `80_grouped_query_attention`
11. `64_weight_dequantization`
12. `96_int8_kv_cache_attention`

## Suggested Cadence

- 每题先读题目描述和 reference，再自己写 CUDA/Triton 实现。
- 每题记录三件事：kernel 设计、性能瓶颈、下一版优化点。
- 每个 phase 完成后做一次复盘，重点比较 memory bandwidth、occupancy、shared memory、register pressure、numeric stability。
- 对 attention、GEMM、quantization 题目至少做两版：先正确，再优化。
