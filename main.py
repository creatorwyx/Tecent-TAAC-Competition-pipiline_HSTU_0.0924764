import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.amp import GradScaler, autocast
from dataset import MyDataset
from model import BaselineModel
def get_args():
    parser = argparse.ArgumentParser()
    # Train params
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    # Baseline Model construction
    parser.add_argument('--embsize', default=32, type=int, help='Embedding dimension for discrete features')
    parser.add_argument('--hiddendim', default=32, type=int, help='Hidden dimension for model layers')
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--num_heads', default=2, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--item_emb_dropout_rate', default=0, type=float, help='Dropout rate for item embeddings')
    parser.add_argument('--seq_emb_dropout_rate', default=0.2, type=float, help='Dropout rate for sequence embeddings (after position encoding)')
    parser.add_argument('--attention_dropout_rate', default=0.2, type=float, help='Dropout rate for attention layers')
    parser.add_argument('--feedforward_dropout_rate', default=0.2, type=float, help='Dropout rate for feedforward layers')
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')
    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])
    # SL超参数
    parser.add_argument('--sl_alpha', type=float, default=2.0,help='Stochastic Length alpha (1.6~2.0)')
    # 新增：优化器相关参数
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay for AdamW optimizer')
    parser.add_argument('--max_grad_norm', type=float, default=2.0, help='Max gradient norm for clipping')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio of total training steps')
    parser.add_argument('--target_lr', type=float, default=0.001, help='Target learning rate after warmup')
    # 混合精度训练参数
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use automatic mixed precision training')
    parser.add_argument('--amp_dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16'],
                       help='AMP data type')
    args = parser.parse_args()
    return args
def get_warmup_cosine_scheduler(optimizer, num_warmup_steps, num_training_steps, target_lr=0.001, last_epoch=-1):
    """
    Warmup到0.001 + 余弦退火学习率调度器
    """
    initial_lr = optimizer.param_groups[0]['lr']
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # 线性预热到目标学习率
            return float(current_step) / float(max(1, num_warmup_steps)) * (target_lr / initial_lr)
        else:
            # 余弦退火
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress))) * (target_lr / initial_lr)
    return LambdaLR(optimizer, lr_lambda, last_epoch)
def compute_batch_hitrate_at_k(seq_embs, pos_embs, neg_embs, loss_mask, k=10):
    """
    计算batch内的hitrate@k
    Args:
        seq_embs: 序列embedding，形状为 [batch_size, maxlen, hidden_size]
        pos_embs: 正样本embedding，形状为 [batch_size, maxlen, hidden_size]
        neg_embs: 负样本embedding，形状为 [batch_size, maxlen, hidden_size]
        loss_mask: 损失掩码，形状为 [batch_size, maxlen]
        k: 计算hitrate@k的k值

    Returns:
        hitrate: batch内的hitrate@k
    """
    batch_size, maxlen, hidden_size = seq_embs.shape
    # 归一化所有embedding
    seq_embs_norm = seq_embs / (seq_embs.norm(p=2, dim=-1, keepdim=True) + 1e-8)
    pos_embs_norm = pos_embs / (pos_embs.norm(p=2, dim=-1, keepdim=True) + 1e-8)
    neg_embs_norm = neg_embs / (neg_embs.norm(p=2, dim=-1, keepdim=True) + 1e-8)
    # 只计算有效的item token位置
    valid_mask = loss_mask.bool()  # [batch_size, maxlen]
    if not valid_mask.any():
        return torch.tensor(0.0, device=seq_embs.device)
    # 提取有效的用户表征和正样本表征
    valid_seq_embs = seq_embs_norm[valid_mask]  # [num_valid, hidden_size]
    valid_pos_embs = pos_embs_norm[valid_mask]  # [num_valid, hidden_size]
    # 计算正样本相似度
    pos_scores = torch.sum(valid_seq_embs * valid_pos_embs, dim=-1)  # [num_valid]
    # 计算负样本相似度（使用batch内所有负样本）
    neg_embs_flat = neg_embs_norm.view(-1, hidden_size)  # [batch_size * maxlen, hidden_size]
    neg_scores = torch.matmul(valid_seq_embs, neg_embs_flat.t())  # [num_valid, batch_size * maxlen]
    # 拼接正负样本分数
    all_scores = torch.cat([pos_scores.unsqueeze(-1), neg_scores], dim=-1)  # [num_valid, 1 + batch_size * maxlen]
    # 计算排名
    sorted_indices = torch.argsort(all_scores, dim=-1, descending=True)  # [num_valid, 1 + batch_size * maxlen]
    # 找到正样本的位置（正样本在第一个位置）
    pos_ranks = torch.where(sorted_indices == 0, torch.arange(sorted_indices.size(1), device=sorted_indices.device),
                            torch.tensor(sorted_indices.size(1), device=sorted_indices.device))
    pos_ranks = pos_ranks.min(dim=-1)[0]  # [num_valid]
    # 计算hitrate@k
    hit_at_k = (pos_ranks < k).float()  # [num_valid]
    hitrate = hit_at_k.mean()
    return hitrate
def print_dropout_config(args):
    """
    打印所有dropout配置信息
    """
    print("\n" + "="*60)
    print("DROPOUT CONFIGURATION")
    print("="*60)
    print(f"Item Embedding Dropout Rate:     {args.item_emb_dropout_rate}")
    print(f"Sequence Embedding Dropout Rate: {args.seq_emb_dropout_rate}")
    print(f"Attention Dropout Rate:          {args.attention_dropout_rate}")
    print(f"FeedForward Dropout Rate:        {args.feedforward_dropout_rate}")
    print("="*60 + "\n")
if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    data_path = os.environ.get('TRAIN_DATA_PATH')
    args = get_args()
    # 设置混合精度训练
    if args.use_amp:
        if args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
        scaler = GradScaler('cuda')
        print(f"Using automatic mixed precision training with {args.amp_dtype}")
    else:
        amp_dtype = torch.float32
        scaler = None
        print("Using full precision training")
    dataset = MyDataset(data_path, args)
    # train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    total = len(dataset)
    train_size = int(0.99 * total)
    valid_size = total - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    # 打开训练标志
    train_dataset.dataset.training = True
    print(f"[Debug] Dataset length: {len(dataset)}")
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0
    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')
    # 使用AdamW优化器，支持weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        betas=(0.9, 0.98), 
        eps=1e-8, 
        weight_decay=args.weight_decay
    )
    # 计算总训练步数和预热步数
    num_training_steps = args.num_epochs * len(train_loader)
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    # 创建warmup到0.001+余弦退火学习率调度器
    scheduler = get_warmup_cosine_scheduler(
        optimizer, 
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        target_lr=args.target_lr
    )
    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    print("Start training")
    print(f"Learning rate schedule: Warmup to {args.target_lr} + Cosine Annealing")
    print(f"Total training steps: {num_training_steps}")
    print(f"Warmup steps: {num_warmup_steps}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Max gradient norm: {args.max_grad_norm}")
    # 打印dropout配置
    print_dropout_config(args)
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            optimizer.zero_grad()
            # 使用混合精度训练
            with autocast('cuda', dtype=amp_dtype, enabled=args.use_amp):
                seq_embs, pos_embs, neg_embs, loss_mask = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                )
                loss = model.compute_infonce_loss(seq_embs, pos_embs, neg_embs, loss_mask)
                # L2正则化
                for param in model.item_emb.parameters():
                    loss += args.l2_emb * torch.norm(param)
            # 计算batch内的hitrate@10
            with torch.no_grad():
                hitrate_at_10 = compute_batch_hitrate_at_k(seq_embs, pos_embs, neg_embs, loss_mask, k=10)
                writer.add_scalar('Metrics/hitrate@10', hitrate_at_10.item(), global_step)
            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time(), 'temp': model.temp, 'hitrate@10': hitrate_at_10.item()}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], global_step)
            global_step += 1
            # 混合精度反向传播
            if args.use_amp:
                scaler.scale(loss).backward()
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                # 优化器步进
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()
            # 在每个step后更新学习率（warmup+余弦退火）
            scheduler.step()
        model.eval()
        valid_loss_sum = 0
        for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            with autocast('cuda', dtype=amp_dtype, enabled=args.use_amp):
                seq_embs, pos_embs, neg_embs, loss_mask = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                )
                loss = model.compute_infonce_loss(seq_embs, pos_embs, neg_embs, loss_mask)
            valid_loss_sum += loss.item()
        valid_loss_sum /= len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)
        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")
    print("Done")
    writer.close()
    log_file.close()