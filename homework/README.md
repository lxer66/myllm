# 从零搭建大语言模型

参考 MiniMind 仓库，从零实现大语言模型的模型结构、数据处理、训练与推理评测。

## 整体阶段

1. 模型搭建
2. 数据准备
3. 训练
4. 推理评测

## 当前进度

模型搭建已完成，数据准备进行中。

## 文件说明

### model/config.py

模型配置类 `Config`，继承 HF `PretrainedConfig`，支持 `save_pretrained` / `from_pretrained`。

### model/model.py

完整模型实现，所有模块从零使用 PyTorch 编写。

- `RMSNorm`、`Attention`（GQA + RoPE + Flash Attention）、`FeedForward`（SwiGLU）、`MOEFeedForward`（top-k 路由）
- `TransformerBlock` — 单个 Transformer 层
- `MyLLMModel` — 骨干网络：Embedding → N 层 TransformerBlock → RMSNorm
- `MyLLMForCausalLM` — 完整因果语言模型，继承 `PreTrainedModel` + `GenerationMixin`，含 loss 计算和自回归 `generate()`

### model/tokenizer.py

独立实现的 BPE + ByteLevel 分词器。

- `MyTokenizer` — encode / decode / batch_decode / apply_chat_template
- `BatchEncoding` — 分词输出容器

### model/tokenizer.json / tokenizer_config.json

6400 词表及分词器配置，BOS 为 `<|im_start|>`、EOS 为 `<|im_end|>`。
