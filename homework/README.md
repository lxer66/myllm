# 从零搭建大语言模型

参考 MiniMind 仓库，从零实现大语言模型的模型结构、数据处理、训练与推理评测。

## 整体阶段

1. 模型搭建
2. 数据准备
3. 预训练
4. SFT / 微调
5. 推理评测

## 当前进度

模型搭建、数据预处理与预训练脚本已完成，预训练已启动。

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

### model/tokenizer.json / tokenizer_config.json

6400 词表及分词器配置，BOS 为 `<|im_start|>`、EOS 为 `<|im_end|>`、PAD/UNK 为 `<|endoftext|>`。chat template 支持 `tools` / `open_thinking` 参数。

### dataset/lm_dataset.py

所有训练阶段的 Dataset 类。

- `PretrainDataset` — 预训练数据集，读取 JSONL，ByteLevel BPE 编码后拼接 BOS/EOS，pad 到 `max_seq_len`，labels 中 pad 位置为 -100
- `SFTDataset` — 对话微调数据集，`generate_labels` 只对 assistant 回复区间计算 loss
- `DPODataset` — 偏好对齐数据集
- `RLAIFDataset` — RL 对齐数据集
- `AgentRLDataset` — Agent 工具调用 RL 数据集
- `pre_processing_chat` / `post_processing_chat` — 数据增强辅助函数

### trainer/train_pretrain.py

预训练脚本，遵循标准 9 步骨架：DDP 初始化 → 配置/续训检查 → 混合精度 → wandb → 模型/数据/优化器 → 恢复状态 → compile/DDP 包装 → 训练循环 → 清理进程组。

默认训练 1 个 epoch，有效 batch=256，每 50000 步保存两份产物：`out/` 下的 half 精度纯权重 `.pth` 和 `checkpoints/` 下的完整续训包。训练记录通过 WandB 实时上报 loss / ppl / grad_norm / tokens_per_sec 等指标。

### trainer/trainer_utils.py

训练公共工具。

- `init_model` — 创建 tokenizer + 模型，可选加载预训练权重
- `lm_checkpoint` — 续训检查点读写
- `get_lr` — 余弦学习率调度
- `init_distributed_mode` / `setup_seed` / `is_main_process` / `Logger` — 分布式训练基础设施
- `SkipBatchSampler` — 支持跳过 batch 的采样器（续训用）
- `LMForRewardModel` — 基于外部 RM 的打分接口（RL 阶段用）
- `get_model_params` — 参数量统计（区分总参/激活参数）

## 训练命令

```bash
cd ./homework/trainer

# 单 GPU 预训练
python train_pretrain.py

# MoE 变体
python train_pretrain.py --use_moe 1

# 多 GPU（DDP）
torchrun --nproc_per_node N train_pretrain.py

# 从预训练权重继续训练
python train_pretrain.py --from_weight pretrain

# 断点续训
python train_pretrain.py --from_resume 1

# 开启实验记录
python train_pretrain.py --use_wandb
```

训练数据默认路径为 `../dataset/pretrain_t2t.jsonl`，可通过 `--data_path` 指定。
