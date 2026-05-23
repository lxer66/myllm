# 从零搭建大语言模型

参考 MiniMind 仓库，从零实现大语言模型的模型结构、数据处理、训练与推理评测。

## 整体阶段

1. 模型搭建
2. 数据准备（含预 tokenize）
3. 预训练
4. SFT / 微调
5. 推理评测

## 当前进度

模型搭建、数据预处理、预 tokenize、预训练、SFT 微调与推理评测均已完成。

## 文件说明

### model/config.py

模型配置类 `Config`，继承 HF `PretrainedConfig`，model_type 为 `"MicroLM"`。

默认超参：`hidden_size=768`，`num_hidden_layers=8`，`num_attention_heads=8`（GQA kv=4），`vocab_size=6400`，`max_position_embeddings=32768`，`rope_theta=1e6`。

### model/model.py

完整模型实现，所有模块从零使用 PyTorch 编写，命名对齐 `MicroLM`。

- `RMSNorm` — Root Mean Square Layer Normalization
- `precompute_freqs_cis` / `apply_rotary_pos_emb` — RoPE 位置编码（含 YaRN 推演支持，推理时启用）
- `Attention` — GQA + RoPE + Flash Attention + QK Norm
- `FeedForward` — SwiGLU 前馈网络
- `MOEFeedForward` — 混合专家前馈，top-1 路由（4 experts），含负载均衡辅助损失
- `TransformerBlock` — 单个 Transformer 层（Pre-Norm 结构）
- `MicroLMModel` — 骨干网络：Embedding → N 层 TransformerBlock → RMSNorm
- `MicroLMForCausalLM` — 完整因果语言模型，继承 `PreTrainedModel` + `GenerationMixin`，含 loss 计算、weight tying 和自回归 `generate()`

### model/tokenizer.py

独立实现的 BPE + ByteLevel 分词器，完全复刻 HF tokenizer 的编码/解码行为。

- `MyTokenizer` — encode / decode / batch_decode / apply_chat_template / save_pretrained
- `BatchEncoding` — 分词输出容器，支持 `.to(device)`

**已知问题**：自研 tokenizer 的 `_bpe()` 为纯 Python 实现，长文本（>5000 字符）性能极差（O(n²)），不适用于在线数据加载。训练时推荐使用预 tokenize 后的 `.bin` 数据。

### model/tokenizer.json / tokenizer_config.json

6400 词表及分词器配置，BOS 为 `<|im_start|>`、EOS 为 `<|im_end|>`、PAD/UNK 为 `<|endoftext|>`。chat template 支持 `tools` / `open_thinking` 参数。

### dataset/lm_dataset.py

所有训练阶段的 Dataset 类。

- `PretrainDataset` — 预训练数据集，读取 JSONL，ByteLevel BPE 编码后拼接 BOS/EOS，pad 到 `max_seq_len`。文本在 tokenize 前截断到 `max_length * 4` 字符，避免长文本卡死
- `BinDataset` — 预 tokenize 后的二进制数据集，mmap 读取，**零 CPU、零随机 I/O**。从 `pretrain_t2t.bin` 直接按索引取 token ids，无需 JSON 解析和 BPE 分词
- `SFTDataset` — 对话微调数据集，`generate_labels` 只对 assistant 回复区间计算 loss
- `DPODataset` / `RLAIFDataset` / `AgentRLDataset` — 其他训练阶段数据集
- `pre_processing_chat` / `post_processing_chat` — 数据增强辅助函数

### dataset/pre_tokenize.py

**一次性预 tokenize 脚本**，将 JSONL 转为 `.bin` 二进制张量文件。

- 使用 HuggingFace 原生 `tokenizers` 库（Rust 后端），处理速度 8000-15000 it/s
- 多进程并行（默认 8 进程），846 万条数据约 10 分钟完成
- 输出格式：`int16` 二进制文件（~11.5 GB）+ 形状元数据 JSON
- 用法：`cd dataset && python pre_tokenize.py`

### trainer/train_pretrain.py

预训练脚本，遵循标准 9 步骨架。

- 支持在线 tokenize（`PretrainDataset`，读 JSONL）和预 tokenize（`BinDataset`，读 `.bin`），通过 `--data_path` 后缀自动切换
- 本地 seaborn 绘图：每 1000 步自动生成 loss/ppl/lr/grad 四张曲线图，保存至 `trainer/plots/`
- 数据点持久化至 `plots/metrics.json`，续训时自动恢复历史数据
- 默认 `batch_size=128`、`accumulation_steps=2`（有效 batch=256）、`epochs=1`、`save_interval=20000`、`lr=5e-4`
- Step 0 记录初始 loss，便于对比训练效果

### trainer/train_full_sft.py

全参数 SFT 脚本，与预训练脚本结构对齐。

- 基于 `from_weight=pretrain` 的预训练权重进行微调
- 同样使用 seaborn 本地绘图，图表保存至 `trainer/plots_sft/`
- 默认 `batch_size=32`、`accumulation_steps=1`（有效 batch=32）、`epochs=1`、`lr=1.5e-5`、`max_seq_len=768`
- 对话数据通过 `SFTDataset.generate_labels` 只对 assistant 回复区间计算 loss

### trainer/trainer_utils.py

训练公共工具。

- `init_model` — 创建 tokenizer + 模型，可选加载预训练权重
- `lm_checkpoint` — 续训检查点读写（含 GPU 数量变化时 step 自动缩放）
- `get_lr` — 余弦学习率调度
- `init_distributed_mode` / `setup_seed` / `is_main_process` / `Logger` — 分布式训练基础设施
- `SkipBatchSampler` — 支持跳过 batch 的采样器（续训用）
- `LMForRewardModel` — 基于外部 RM 的打分接口（RL 阶段用）
- `get_model_params` — 参数量统计（区分总参/激活参数）

### eval_pretrain.py

**预训练模型推理测试脚本**。加载 `out/pretrain_768.pth`，使用自回归补全模式生成文本。支持手动输入和预设提示词自动测试。

### eval_sft.py

**SFT 模型对话测试脚本**。加载 `out/full_sft_768.pth`，使用 chat template 进行多轮对话。支持携带历史对话（`--historys N`）和预设提示词自动测试。

## 训练命令

```bash
cd ./homework/trainer
```

### 预 tokenize（首次训练前执行一次即可）

```bash
cd ../dataset
python pre_tokenize.py                         # 生成 pretrain_t2t.bin（~11.5 GB，约 10 分钟）
```

### 预训练

```bash
cd ../trainer

# 预 tokenize 训练（推荐，零 CPU/IO 瓶颈）
python train_pretrain.py --data_path ../dataset/pretrain_t2t.bin

# 在线 tokenize 训练（备用，长文本数据会卡顿）
python train_pretrain.py

# MoE 变体
python train_pretrain.py --data_path ../dataset/pretrain_t2t.bin --use_moe 1

# 断点续训
python train_pretrain.py --data_path ../dataset/pretrain_t2t.bin --from_resume 1
```

### SFT 微调

```bash
# 基于预训练权重
python train_full_sft.py

# 多 GPU（DDP）
torchrun --nproc_per_node N train_full_sft.py

# 断点续训
python train_full_sft.py --from_resume 1
```

### 推理测试

```bash
cd ./homework

# 预训练模型（补全模式）
python eval_pretrain.py                           # 自动加载 out/pretrain_768.pth

# SFT 模型（对话模式）
python eval_sft.py                                # 自动加载 out/full_sft_768.pth

# SFT 多轮对话
python eval_sft.py --historys 4
```

预训练模型使用补全模式（非对话），提示词如 "中国的首都是"、"机器学习是"。

## 预 tokenize 数据说明

| 项目 | 值 |
|------|-----|
| 源数据 | `dataset/pretrain_t2t.jsonl`（7.8 GB，846 万条） |
| 输出 | `dataset/pretrain_t2t.bin`（11.5 GB，int16） |
| 元数据 | `dataset/pretrain_t2t.bin.json` |
| 耗时 | ~10 分钟（8 进程，Rust tokenizer） |
| 截断策略 | BPE 编码后取前 `max_seq_len - 2` 个 token，再拼接 BOS/EOS |

## 训练曲线

训练过程中自动生成，实时覆盖更新：

| 图表 | 路径 |
|------|------|
| 预训练 | `trainer/plots/loss.png` / `ppl.png` / `lr.png` / `grad.png` |
| SFT | `trainer/plots_sft/loss.png` / `ppl.png` / `lr.png` / `grad.png` |

数据点保存在同目录 `metrics.json` 中，可随时用原始数据重新画图。
