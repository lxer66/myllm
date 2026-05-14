# CLAUDE.md

本文档用于指导 Claude Code（claude.ai/code）在本仓库中协作时的行为。

## 沟通语言

**所有回复、代码注释、提交信息、PR 描述以及与用户的全部沟通一律使用中文。** 本项目的 README、代码注释、CLI 帮助说明本身就是中文，请保持一致的语言风格，避免中英文混杂。仅在需要保留原始术语（如 `RoPE`、`MoE`、`DDP`）或引用代码标识符时才使用英文。

## 项目概览

MiniMind 是一个从 0 开始训练大语言模型的项目，主线 Dense 模型约 64M 参数（MoE 变体约 198M-A64M），核心算法全部用 PyTorch 原生实现，不依赖 `trl`、`peft`、`transformers.Trainer` 这类高层封装。模型结构对齐 `Qwen3 / Qwen3-MoE` 生态，训练完的权重可以转换到 `transformers / llama.cpp / vllm / ollama`。项目演示了 LLM 训练的完整链路：Pretrain → SFT → LoRA / DPO / PPO / GRPO / CISPO / 蒸馏 / Agentic RL。

文档主要以中文为主（见 [README.md](README.md) / [README_en.md](README_en.md)），代码注释和 CLI 帮助说明也都是中文。

## 常用命令

所有训练脚本必须 `cd ./trainer` 后执行；推理和格式转换脚本从仓库根目录或 `./scripts/` 执行。

```bash
# --- 推理 ---
python eval_llm.py --load_from ./minimind-3                       # Transformers 格式模型
python eval_llm.py --load_from ./model --weight full_sft          # ./out/ 下的原生 .pth
python eval_llm.py --weight pretrain                              # 测试预训练检查点
python eval_llm.py --weight full_sft --lora_weight lora_medical   # 基模 + LoRA 适配器

# --- 训练（先 cd ./trainer） ---
python train_pretrain.py                                          # next-token 预测，默认 pretrain_t2t_mini.jsonl
python train_full_sft.py                                          # 全参 SFT，默认 from_weight=pretrain
python train_lora.py                                              # PEFT LoRA，在 full_sft 之上叠加
python train_dpo.py                                               # 偏好优化
python train_ppo.py / train_grpo.py                               # RLAIF（使用 rollout_engine）
python train_agent.py                                             # 多轮 Tool Call 的 Agentic RL
python train_distillation.py                                      # 白盒蒸馏（CE + KL）

# --- 分布式训练（DDP） ---
torchrun --nproc_per_node N train_xxx.py                          # N 为 GPU 数量

# --- 断点续训（所有训练脚本通用） ---
python train_xxx.py --from_resume 1                               # 自动检测 ./checkpoints/<weight>_<dim>_resume.pth

# --- 训练可视化 ---
python train_xxx.py --use_wandb                                   # 实际使用的是 SwanLab（接口与 WandB 兼容）

# --- 可选：sglang 作为 RL 的 rollout 后端 ---
python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998

# --- 权重转换与服务 ---
python scripts/convert_model.py                                   # .pth ↔ HF transformers（也支持转换为 Qwen3 / Qwen3MoE 配置）
python scripts/serve_openai_api.py                                # OpenAI 协议兼容的 API 服务（FastAPI）
cd scripts && streamlit run web_demo.py                           # Streamlit 聊天 WebUI
```

项目没有单元测试、Linter 配置或构建步骤，训练脚本本身就是验证手段。

## 仓库结构

- [model/model_minimind.py](model/model_minimind.py) — 单文件模型实现：`MiniMindConfig`、`RMSNorm`、RoPE（可选 YaRN scaling）、`Attention`（GQA + flash）、`FeedForward`、`MOEFeedForward`、`MiniMindBlock`、`MiniMindModel`、`MiniMindForCausalLM`。继承自 HF `PreTrainedModel + GenerationMixin`。
- [model/model_lora.py](model/model_lora.py) — 从 0 实现的 LoRA：`apply_lora` 给所有方阵 `nn.Linear` 旁路 monkey-patch 一个低秩 `A·B` 分支；配套有 `save_lora / load_lora / merge_lora`。
- [model/tokenizer.json](model/tokenizer.json) / [tokenizer_config.json](model/tokenizer_config.json) — 6400 词表，基于 BPE + ByteLevel，包含 `<|im_start|>`、`<|im_end|>`、`<tool_call>`、`<tool_response>`、`<think>` 等特殊 token。chat template 写在 `tokenizer_config.json` 中，支持 `open_thinking` 与 `tools` 参数。
- [dataset/lm_dataset.py](dataset/lm_dataset.py) — 所有 Dataset 类：`PretrainDataset`、`SFTDataset`、`DPODataset`、`RLAIFDataset`、`AgentRLDataset`。`SFTDataset.generate_labels` 会把 `<|im_start|>assistant\n ... <|im_end|>\n` 区间以外的所有位置都 mask 成 `-100`。
- [trainer/](trainer/) — 每个训练阶段一个脚本，全部遵循相同的 9 步骨架（见下文"训练脚本骨架"）。
- [trainer/trainer_utils.py](trainer/trainer_utils.py) — 公共工具：`init_model`、`lm_checkpoint`（续训读写）、`init_distributed_mode`、`get_lr`（余弦学习率）、`SkipBatchSampler`、`LMForRewardModel`。
- [trainer/rollout_engine.py](trainer/rollout_engine.py) — 抽象基类 `RolloutEngine` 及 HF、sglang 两个后端，被 PPO/GRPO/agent 训练器使用。`compute_per_token_logps` 也在这里。
- [scripts/](scripts/) — 推理与转换相关脚本（不含训练逻辑）。
- `out/`（已 gitignore）— `.pth` 权重文件，命名约定：`<weight>_<hidden_size>[_moe].pth`（例如 `full_sft_768.pth`、`pretrain_768_moe.pth`）。
- `checkpoints/` — 续训打包：`<weight>_<hidden_size>[_moe]_resume.pth`，里面是 `{model, optimizer, scaler, epoch, step, world_size, wandb_id}`。
- `dataset/*.jsonl`（已 gitignore，需单独下载）— `pretrain_t2t_mini.jsonl`、`sft_t2t_mini.jsonl`、`dpo.jsonl`、`rlaif.jsonl`、`agent_rl.jsonl` 等。

## 架构与约定

**模型超参。** 默认配置 `hidden_size=768, num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4`（GQA），`vocab_size=6400`，`max_position_embeddings=32768`，`rope_theta=1e6`。MoE 额外有 `num_experts=4, num_experts_per_tok=1`（top-1 路由，无 shared expert）。所有训练脚本都支持 `--hidden_size`、`--num_hidden_layers`、`--use_moe`，并且**跨阶段必须保持一致** —— 检查点文件名里硬编码了 `hidden_size` 与可选的 `_moe` 后缀。

**训练脚本骨架。** 每个 `trainer/train_*.py` 都遵循同一套带编号的 9 步：初始化 DDP/随机种子 → 构造 config 并检查续训 → 设置 autocast 混合精度 → 初始化 wandb → 构造模型/数据/优化器 → 从续训 ckp 恢复状态 → `torch.compile` 与 DDP 包装 → 训练循环 → 销毁分布式进程组。新增训练脚本时请对齐这套骨架，[trainer/train_pretrain.py](trainer/train_pretrain.py) 与 [trainer/train_full_sft.py](trainer/train_full_sft.py) 是最干净的参考。

**检查点协议。** `lm_checkpoint()` 在每个 save_interval 会落两份产物：`out/` 下的 half 精度纯权重 `.pth`（用于推理）以及 `checkpoints/` 下的完整续训包（用于 `--from_resume 1`）。续训时如果 world_size 变了，会自动按比例缩放保存的 `step`。`--from_resume 1` 的时候续训包是在**优化器构造之前**读出来的，所以 model/optimizer/scaler 的 state 是在对象构造之后再 load 进去的。

**RoPE 与 YaRN。** 位置编码在 `MiniMindModel.__init__` 中通过 `precompute_freqs_cis` 一次性预计算。推理时如需长上下文外推，给 `eval_llm.py` 传 `--inference_rope_scaling`，会启用 `factor=16` 的 YaRN，从 `original_max_position_embeddings=2048` 外推。**训练阶段不使用 YaRN。**

**Chat template 与 label mask。** `SFTDataset` 通过 `tokenizer.apply_chat_template(messages, tools=...)` 构造输入。Label 通过扫描 `bos_id` / `eos_id` 序列 `<|im_start|>assistant\n` 与 `<|im_end|>\n` 计算 —— 只有 assistant 的 span 参与 loss。`pre_processing_chat` 以 20% 的概率随机在对话头部插入 system prompt；`post_processing_chat` 以 80% 的概率去掉空的 `<think>\n\n</think>\n\n` 块。

**Tool Call 数据已混入 SFT 主线。** `sft_t2t.jsonl` / `sft_t2t_mini.jsonl` 里已经包含工具调用样本，所以 `full_sft` 训练完就具备基础 Tool Call 能力，无需独立的工具调用 SFT 阶段。可以用 [scripts/chat_api.py](scripts/chat_api.py) 与 [scripts/eval_toolcall.py](scripts/eval_toolcall.py) 验证。

**Agentic / RL 数据流。** `train_ppo.py`、`train_grpo.py`、`train_agent.py` 通过 `rollout_engine` 生成 completion，然后用 `rollout_engine.py` 中的 `compute_per_token_logps` 计算 policy / ref 的 log-prob。Reward 由 `LMForRewardModel.get_score`（Qwen 风格的 RM）加上规则化奖励组成：长度窗口、`<think>` 是否存在以及长度、n-gram 重复惩罚。默认 rollout 后端是 HF Transformers；`--use_sglang` 可切换到 sglang，需要先按上面的命令把 sglang server 起起来。

**SwanLab 替代 WandB。** 脚本里写的是 `import swanlab as wandb`，CLI 参数仍叫 `--use_wandb`，project 名是 `"MiniMind-Pretrain"` 等。SwanLab 接口与 WandB 兼容，但每个 trainer 的第 4 步会硬编码 project 字符串与 run name 模板。

**日志规范。** 所有 `print` 都走 `Logger()`，内部会用 `is_main_process()` 过滤，DDP 下只有 rank 0 会输出。不要用裸 `print` 替代它。

## 易踩的坑

- **[requirements.txt](requirements.txt) 的依赖处理。** `torch` 被注释掉了（要根据自己的 CUDA 版本单独装），`peft` 也被注释掉了（项目自己实现了 LoRA），FastAPI 才是当前主推的服务端。
- **Tokenizer 已冻结。** [trainer/train_tokenizer.py](trainer/train_tokenizer.py) 仅作为学习参考 —— 仓库自带的 6400 词表是开源权重的统一前提，改了 tokenizer 会破坏所有已发布的 checkpoint。
- **不要误解 `add_special_tokens`。** 当前 MiniMind tokenizer 的 HF 真实行为就是：`encode(..., add_special_tokens=True)` **不会**自动加 BOS / EOS。这不是仓库自研 tokenizer 的缺陷，而是当前配置本来如此：`tokenizer.json` 的 `post_processor = None`，`tokenizer_config.json` 的 `add_bos_token = false`、`add_eos_token = false`。HF 版和仓库自研 `MiniMindTokenizer` 在这点上保持一致。预训练数据与基座补全推理如果需要起止符，必须由上层手动添加；可参考 [dataset/lm_dataset.py](dataset/lm_dataset.py) 中预训练样本在编码后手动拼接 `bos_token_id` / `eos_token_id` 的做法，以及 [eval_llm.py](eval_llm.py) 中 `pretrain` 模式手动在 prompt 前加 `tokenizer.bos_token` 的做法。对话场景不依赖这个参数，而是依赖 `apply_chat_template()` 直接把 `<|im_start|>` / `<|im_end|>` 写入文本。
- **所有对话权重都使用 `<|im_start|>` / `<|im_end|>`。** `2025-04-26` 之前的旧权重用的是 `<s></s>`，当前主线已经不再支持直接加载。
- **`max_seq_len` 单位是 token，不是字符。** README 给出的换算大概是中文 1.5~1.7 字符 / token、英文 4~5 字符 / token。推荐值：`pretrain_t2t_mini` 约 340，`sft_t2t_mini` 约 768，都是按数据长度分布调过的。
- **MoE 训练比 Dense 慢约 50%。** 当前 `4 experts / top-1` 的实现没有 fused MoE kernel，循环内是按 expert 逐个 forward 的，因此训练吞吐反而下降。
- **续训时 GPU 数量变化的支持是有限的。** `SkipBatchSampler` 与 step 自动重缩放支持 dataloader 跳过，但优化器状态和学习率调度**不会**被重新缩放，跨 `--nproc_per_node` 续训请留意这个边界。
- **Windows + bash 环境。** 仓库 checkout 在 Windows 上，但 shell 是 bash —— 路径用正斜杠，空设备用 `/dev/null` 而不是 `NUL`。
