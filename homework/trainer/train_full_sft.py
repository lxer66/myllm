import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import math
import time
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark")
# 注意：Windows 上必须先导入 datasets（pyarrow）再导入 torch，否则 DLL 冲突会导致 segfault
from dataset.lm_dataset import SFTDataset
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.config import Config
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def save_metrics(steps, losses, ppls, lrs, grads):
    plot_dir = os.path.join(os.path.dirname(__file__), 'plots_sft')
    os.makedirs(plot_dir, exist_ok=True)
    with open(os.path.join(plot_dir, 'metrics.json'), 'w') as f:
        json.dump({'steps': steps, 'loss': losses, 'ppl': ppls, 'lr': lrs, 'grad': grads}, f)


def draw_plots(steps, losses, ppls, lrs, grads):
    plot_dir = os.path.join(os.path.dirname(__file__), 'plots_sft')
    os.makedirs(plot_dir, exist_ok=True)
    save_metrics(steps, losses, ppls, lrs, grads)
    for name, values, ylabel, color in [
        ('loss', losses, 'Loss', 'blue'),
        ('ppl', ppls, 'Perplexity', 'red'),
        ('lr', lrs, 'Learning Rate', 'green'),
        ('grad', grads, 'Gradient Norm', 'purple'),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, values, color=color, linewidth=0.8)
        ax.set_xlabel('Step')
        ax.set_ylabel(ylabel)
        ax.set_title(f'SFT {ylabel} — step {steps[-1]}')
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f'{name}.png'), dpi=120)
        plt.close(fig)


def train_epoch(epoch, loader, iters, start_step=0, metrics=None):
    start_time = time.time()
    last_step = start_step
    cached_grad = 0.0

    if start_step == 0 and epoch * iters == 0:
        input_ids, labels = loader.dataset[0]
        input_ids = input_ids.unsqueeze(0).to(args.device)
        labels = labels.unsqueeze(0).to(args.device)
        with torch.no_grad():
            with autocast_ctx:
                res = model(input_ids, labels=labels)
                loss0 = res.loss + (res.aux_loss or 0)
        lr0 = get_lr(epoch * iters, args.epochs * iters, args.learning_rate)
        Logger(f'[Step 0] 初始loss: {loss0.item():.4f}')
        if metrics is not None:
            metrics['steps'].append(0)
            metrics['losses'].append(loss0.item())
            metrics['ppls'].append(math.exp(loss0.item()))
            metrics['lrs'].append(lr0)
            metrics['grads'].append(0.0)
        del input_ids, labels, res, loss0

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            cached_grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        global_step = epoch * iters + step
        if global_step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_ppl = math.exp(current_loss)
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, ppl: {current_ppl:.2f}, aux: {current_aux_loss:.4f}, lr: {current_lr:.2e}, grad: {cached_grad:.2f}, eta: {eta_min:.1f}min')
            if metrics is not None and is_main_process():
                metrics['steps'].append(global_step)
                metrics['losses'].append(current_loss)
                metrics['ppls'].append(current_ppl)
                metrics['lrs'].append(current_lr)
                metrics['grads'].append(cached_grad)
                if global_step % 1000 == 0:
                    draw_plots(metrics['steps'], metrics['losses'], metrics['ppls'], metrics['lrs'], metrics['grads'])

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, save_dir='../checkpoints')
            model.train()
            del state_dict

        del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MicroLM Full SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=768, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t.jsonl", help="训练数据路径")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = Config(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 绘图目录准备 ==========
    os.makedirs(os.path.join(os.path.dirname(__file__), 'plots_sft'), exist_ok=True)

    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        Logger(f"续训检查点已加载: {lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}")
        Logger(f"  ├─ 文件: checkpoints/{args.save_weight}_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}_resume.pth")
        Logger(f"  ├─ Epoch: {start_epoch + 1}/{args.epochs}")
        Logger(f"  └─ Step:  {start_step}")
    else:
        Logger("从头开始训练 (未检测到续训检查点)")

    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    Logger('=' * 60)
    Logger(f'  训练配置')
    Logger(f'  ├─ 模型: MicroLM-{lm_config.hidden_size}{" MoE" if lm_config.use_moe else " Dense"}')
    Logger(f'  ├─ 数据: {args.data_path} ({len(train_ds):,} 条)')
    Logger(f'  ├─ 序列: max_seq_len={args.max_seq_len}')
    Logger(f'  ├─ 批次: batch_size={args.batch_size} × accumulation={args.accumulation_steps} (有效={args.batch_size * args.accumulation_steps})')
    Logger(f'  ├─ 设备: {args.device} | 精度: {args.dtype} | workers: {args.num_workers}')
    Logger(f'  └─ Epoch: {start_epoch + 1}→{args.epochs} | Step: {start_step + 1}→?')
    Logger('=' * 60)

    metrics = {'steps': [], 'losses': [], 'ppls': [], 'lrs': [], 'grads': []}
    metrics_file = os.path.join(os.path.dirname(__file__), 'plots_sft', 'metrics.json')
    if os.path.exists(metrics_file) and start_step > 0:
        with open(metrics_file) as f:
            old = json.load(f)
        metrics['steps'] = old['steps']
        metrics['losses'] = old['loss']
        metrics['ppls'] = old['ppl']
        metrics['lrs'] = old['lr']
        metrics['grads'] = old['grad']
        Logger(f'已加载 {len(metrics["steps"])} 个历史数据点')
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        total_steps = len(loader) + skip
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}步，从 step {start_step + 1} 开始')
        train_epoch(epoch, loader, total_steps, skip, metrics)
    if metrics['steps']:
        draw_plots(metrics['steps'], metrics['losses'], metrics['ppls'], metrics['lrs'], metrics['grads'])
        Logger(f'最终图表已保存至 trainer/plots_sft/ (共{len(metrics["steps"])}个数据点)')

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
