from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.init as init
from dataset import save_emb

# ==================== HSTU Attention ====================
class HSTUAttention(torch.nn.Module):
    """HSTU Self-Attention严格对齐历史checkpoint实现"""
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super().__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.u_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)
    def forward(self, query, key, value, attn_mask=None):
        B, L, _ = query.size()
        Q = self.q_linear(query)
        K = self.k_linear(key)
        U = self.u_linear(query)
        V = self.v_linear(value)
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        U = U.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        Q, K = self.apply_rope(Q, K)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask.unsqueeze(1), -1e9)
        weights = F.silu(scores)
        attn_output = torch.matmul(weights, V)
        output = attn_output * U
        output = output.transpose(1, 2).contiguous().view(B, L, self.hidden_units)
        output = self.out_linear(output)
        output = self.dropout(output)
        return output, weights
    @staticmethod
    def apply_rope(Q, K):
        B, H, L, D = Q.size()
        device = Q.device
        pos = torch.arange(L, device=device)
        freqs = 1.0 / (10000 ** (torch.arange(0, D, 2, device=device) / D))
        angles = pos[:, None] * freqs[None, :]
        sin, cos = angles.sin(), angles.cos()
        q1, q2 = Q[..., ::2], Q[..., 1::2]
        k1, k2 = K[..., ::2], K[..., 1::2]
        Q_rot = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).flatten(-2)
        K_rot = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).flatten(-2)
        return Q_rot, K_rot
# ==================== HSTU FeedForward (SwiGLU) ====================
class HSTUFeedForward(torch.nn.Module):
    """SwiGLU风格FFN，严格对齐历史checkpoint实现"""
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.linear1 = torch.nn.Linear(hidden_units, hidden_units)
        self.linear2 = torch.nn.Linear(hidden_units, hidden_units)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.dropout2 = torch.nn.Dropout(dropout_rate)
    def forward(self, x):
        x_proj = self.linear1(x)
        x_gate = F.silu(x_proj)
        x = self.linear2(self.dropout1(x_proj * x_gate))
        x = self.dropout2(x)
        return x
class BaselineModel(torch.nn.Module):
    """
    Args:
        user_num: 用户数量
        item_num: 物品数量
        feat_statistics: 特征统计信息，key为特征ID，value为特征数量
        feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        args: 全局参数

    Attributes:
        user_num: 用户数量
        item_num: 物品数量
        dev: 设备
        norm_first: 是否先归一化
        maxlen: 序列最大长度
        item_emb: Item Embedding Table
        user_emb: User Embedding Table
        sparse_emb: 稀疏特征Embedding Table
        emb_transform: 多模态特征的线性变换
        userdnn: 用户特征拼接后经过的全连接层
        itemdnn: 物品特征拼接后经过的全连接层
    """
    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):  #
        super(BaselineModel, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.maxlen = args.maxlen
        self.embsize = args.embsize
        self.hiddendim = args.hiddendim
        # 温度系数改为固定常量
        self._temp = 0.03
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.embsize, padding_idx=0)
        init.kaiming_uniform_(self.item_emb.weight, mode='fan_in', nonlinearity='relu')
        self.item_emb.weight.data[0] = 0
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.embsize, padding_idx=0)
        init.kaiming_uniform_(self.user_emb.weight, mode='fan_in', nonlinearity='relu')
        self.user_emb.weight.data[0] = 0
        self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, args.hiddendim, padding_idx=0)
        init.uniform_(self.pos_emb.weight, a=-0.1, b=0.1)
        self.pos_emb.weight.data[0] = 0
        # 序列embedding dropout（在位置编码之后）
        self.seq_emb_dropout = torch.nn.Dropout(p=args.seq_emb_dropout_rate)
        # Item embedding dropout
        self.item_emb_dropout = torch.nn.Dropout(p=args.item_emb_dropout_rate)
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()
        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self._init_feat_info(feat_statistics, feat_types)
        userdim = args.embsize * (len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT)) + len(
            self.USER_CONTINUAL_FEAT
        )
        itemdim = (
            args.embsize * (len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT))
            + len(self.ITEM_CONTINUAL_FEAT)
            + args.embsize * len(self.ITEM_EMB_FEAT)
        )
        self.userdnn = torch.nn.Linear(userdim, args.hiddendim)
        self.itemdnn = torch.nn.Linear(itemdim, args.hiddendim)
        self.item_emb_norm = torch.nn.LayerNorm(args.hiddendim, eps=1e-8)  # New LayerNorm after item DNN
        self.user_emb_norm = torch.nn.LayerNorm(args.hiddendim, eps=1e-8)  # New LayerNorm after user DNN
        self.combined_emb_norm = torch.nn.LayerNorm(args.hiddendim, eps=1e-8)  # New LayerNorm after combined embeddings
        self.last_layernorm = torch.nn.LayerNorm(args.hiddendim, eps=1e-8)
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hiddendim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            # 使用历史checkpoint中的HSTUAttention
            new_attn_layer = HSTUAttention(
                args.hiddendim, args.num_heads, args.attention_dropout_rate
            )
            self.attention_layers.append(new_attn_layer)
            new_fwd_layernorm = torch.nn.LayerNorm(args.hiddendim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            # 使用历史checkpoint中的HSTUFeedForward
            new_fwd_layer = HSTUFeedForward(args.hiddendim, args.feedforward_dropout_rate)
            self.forward_layers.append(new_fwd_layer)
        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, args.embsize, padding_idx=0)
            init.kaiming_uniform_(self.sparse_emb[k].weight, mode='fan_in', nonlinearity='relu')
            self.sparse_emb[k].weight.data[0] = 0
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, args.embsize, padding_idx=0)
            init.kaiming_uniform_(self.sparse_emb[k].weight, mode='fan_in', nonlinearity='relu')
            self.sparse_emb[k].weight.data[0] = 0
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.embsize, padding_idx=0)
            init.kaiming_uniform_(self.sparse_emb[k].weight, mode='fan_in', nonlinearity='relu')
            self.sparse_emb[k].weight.data[0] = 0
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, args.embsize, padding_idx=0)
            init.kaiming_uniform_(self.sparse_emb[k].weight, mode='fan_in', nonlinearity='relu')
            self.sparse_emb[k].weight.data[0] = 0
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], args.embsize)
            init.xavier_uniform_(self.emb_transform[k].weight)
            init.zeros_(self.emb_transform[k].bias)
    @property
    def temp(self):
        """获取温度系数值"""
        return self._temp
    @temp.setter
    def temp(self, value):
        """设置温度系数值"""
        self._temp = float(value)
    def _init_feat_info(self, feat_statistics, feat_types):
        """
        将特征统计信息（特征数量）按特征类型分组产生不同的字典，方便声明稀疏特征的Embedding Table

        Args:
            feat_statistics: 特征统计信息，key为特征ID，value为特征数量
            feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        """
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}  # 记录的是不同多模态特征的维度
    def feat2tensor(self, seq_feature, k):
        """
        Args:
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典，形状为 [batch_size, maxlen]
            k: 特征ID

        Returns:
            batch_data: 特征值的tensor，形状为 [batch_size, maxlen, max_array_len(if array)]
        """
        batch_size = len(seq_feature)
        if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT:
            # 如果特征是Array类型，需要先对array进行padding，然后转换为tensor
            max_array_len = 0
            max_seq_len = 0
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                max_seq_len = max(max_seq_len, len(seq_data))
                max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data))
            batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                for j, item_data in enumerate(seq_data):
                    actual_len = min(len(item_data), max_array_len)
                    batch_data[i, j, :actual_len] = item_data[:actual_len]
            return torch.from_numpy(batch_data).to(self.dev)
        else:
            # 如果特征是Sparse类型，直接转换为tensor
            max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
            batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                batch_data[i] = seq_data
            return torch.from_numpy(batch_data).to(self.dev)
    def feat2emb(self, seq, feature_array, mask=None, include_user=False):
        """
        Args:
            seq: 序列ID
            feature_array: 特征list，每个元素为当前时刻的特征字典
            mask: 掩码，1表示item，2表示user
            include_user: 是否处理用户特征，在两种情况下不打开：1) 训练时在转换正负样本的特征时（因为正负样本都是item）;2) 生成候选库item embedding时。

        Returns:
            seqs_emb: 序列特征的Embedding
        """
        seq = seq.to(self.dev)
        if include_user:
            user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            user_embedding = self.user_emb(user_mask * seq)
            item_embedding = self.item_emb(item_mask * seq)
            item_embedding = self.item_emb_dropout(item_embedding)
            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]
        else:
            item_embedding = self.item_emb(seq)
            item_embedding = self.item_emb_dropout(item_embedding)
            item_feat_list = [item_embedding]
        if isinstance(feature_array, dict):
            all_feat_types = [
                (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),
                (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),
                (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list),
            ]
            if include_user:
                all_feat_types.extend([
                    (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list),
                    (self.USER_ARRAY_FEAT, 'user_array', user_feat_list),
                    (self.USER_CONTINUAL_FEAT, 'user_continual', user_feat_list),
                ])
            for feat_dict, feat_type, feat_list in all_feat_types:
                if not feat_dict:
                    continue
                for k in feat_dict:
                    tensor_feature = feature_array[k]
                    if feat_type.endswith('sparse'):
                        feat_list.append(self.sparse_emb[k](tensor_feature))
                    elif feat_type.endswith('array'):
                        feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                    elif feat_type.endswith('continual'):
                        # === 修改：移除批内归一化，直接使用 ===
                        feat_list.append(tensor_feature.float().unsqueeze(2))
            for k in self.ITEM_EMB_FEAT:
                tensor_feature = feature_array[k]
                item_feat_list.append(self.emb_transform[k](tensor_feature))
        else:
            all_feat_types = [
                (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),
                (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),
                (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list),
            ]
            if include_user:
                all_feat_types.extend([
                    (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list),
                    (self.USER_ARRAY_FEAT, 'user_array', user_feat_list),
                    (self.USER_CONTINUAL_FEAT, 'user_continual', user_feat_list),
                ])
            for feat_dict, feat_type, feat_list in all_feat_types:
                if not feat_dict:
                    continue
                for k in feat_dict:
                    tensor_feature = self.feat2tensor(feature_array, k)
                    if feat_type.endswith('sparse'):
                        feat_list.append(self.sparse_emb[k](tensor_feature))
                    elif feat_type.endswith('array'):
                        feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                    elif feat_type.endswith('continual'):
                        feat_list.append(tensor_feature.unsqueeze(2))
            for k in self.ITEM_EMB_FEAT:
                batch_size = len(feature_array)
                emb_dim = self.ITEM_EMB_FEAT[k]
                seq_len = len(feature_array[0])
                batch_emb_data = np.zeros((batch_size, seq_len, emb_dim), dtype=np.float32)
                for i, seq in enumerate(feature_array):
                    for j, item in enumerate(seq):
                        if k in item:
                            batch_emb_data[i, j] = item[k]
                tensor_feature = torch.from_numpy(batch_emb_data).to(self.dev)
                item_feat_list.append(self.emb_transform[k](tensor_feature))
        all_item_emb = torch.cat(item_feat_list, dim=2)
        all_item_emb = self.item_emb_norm(torch.relu(self.itemdnn(all_item_emb)))
        if include_user:
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = self.user_emb_norm(torch.relu(self.userdnn(all_user_emb)))
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb
        return seqs_emb
    def log2feats(self, log_seqs, mask, seq_feature):
        """
        Args:
            log_seqs: 序列ID
            mask: token类型掩码，1表示item token，2表示user token
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            time_diff_buckets: 时间差分桶特征，形状为 [batch_size, maxlen]

        Returns:
            seqs_emb: 序列的Embedding，形状为 [batch_size, maxlen, hidden_units]
        """
        batch_size = log_seqs.shape[0]
        maxlen = log_seqs.shape[1]
        seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True)
        seqs = self.combined_emb_norm(seqs)
        seqs *= self.hiddendim**0.5
        poss = torch.arange(1, maxlen + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1).clone()
        poss *= log_seqs != 0
        seqs += self.pos_emb(poss)
        seqs = self.seq_emb_dropout(seqs)
        maxlen = seqs.shape[1]
        ones_matrix = torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones_matrix)
        attention_mask_pad = (mask != 0).to(self.dev)
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)
        for i in range(len(self.attention_layers)):
            x = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
            seqs = seqs + mha_outputs
            y = self.forward_layernorms[i](seqs)
            seqs = seqs + self.forward_layers[i](y)
        log_feats = self.last_layernorm(seqs)
        return log_feats
    def forward(
        self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type, seq_feature, pos_feature, neg_feature
    ):
        """
        训练时调用，计算正负样本的embedding
        Args:
            user_item: 用户序列ID
            pos_seqs: 正样本序列ID
            neg_seqs: 负样本序列ID
            mask: token类型掩码，1表示item token，2表示user token
            next_mask: 下一个token类型掩码，1表示item token，2表示user token
            next_action_type: 下一个token动作类型，0表示曝光，1表示点击
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            pos_feature: 正样本特征list，每个元素为当前时刻的特征字典
            neg_feature: 负样本特征list，每个元素为当前时刻的特征字典

        Returns:
            seq_embs: 序列embedding，形状为 [batch_size, maxlen, hidden_size]
            pos_embs: 正样本embedding，形状为 [batch_size, maxlen, hidden_size]
            neg_embs: 负样本embedding，形状为 [batch_size, maxlen, hidden_size]
            loss_mask: 损失掩码，形状为 [batch_size, maxlen]
        """
        seq_feature = {k: v.to(self.dev) for k, v in seq_feature.items()} if isinstance(seq_feature, dict) else seq_feature
        pos_feature = {k: v.to(self.dev) for k, v in pos_feature.items()} if isinstance(pos_feature, dict) else pos_feature
        neg_feature = {k: v.to(self.dev) for k, v in neg_feature.items()} if isinstance(neg_feature, dict) else neg_feature
        log_feats = self.log2feats(user_item, mask, seq_feature)
        loss_mask = (next_mask == 1).to(self.dev)
        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)
        neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False)
        return log_feats, pos_embs, neg_embs, loss_mask
    def predict(self, log_seqs, seq_feature, mask):
        """
        计算用户序列的表征
        Args:
            log_seqs: 用户序列ID
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            mask: token类型掩码，1表示item token，2表示user token
            time_diff_buckets: 时间差分桶特征，形状为 [batch_size, maxlen]
        Returns:
            final_feat: 用户序列的表征，形状为 [batch_size, hidden_units]
        """
        seq_feature = {k: v.to(self.dev) for k, v in seq_feature.items()} if isinstance(seq_feature, dict) else seq_feature
        log_feats = self.log2feats(log_seqs, mask, seq_feature)
        final_feat = log_feats[:, -1, :]
        # 归一化embedding用于推理
        final_feat = final_feat / (final_feat.norm(dim=-1, keepdim=True) + 1e-8)
        return final_feat
    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        """
        生成候选库item embedding，用于检索

        Args:
            item_ids: 候选item ID（re-id形式）
            retrieval_ids: 候选item ID（检索ID，从0开始编号，检索脚本使用）
            feat_dict: 训练集所有item特征字典，key为特征ID，value为特征值
            save_path: 保存路径
            batch_size: 批次大小
        """
        all_embs = []
        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))
            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)
            batch_feat = []
            for i in range(start_idx, end_idx):
                batch_feat.append(feat_dict[i])
            batch_feat = np.array(batch_feat, dtype=object)
            batch_emb = self.feat2emb(item_seq, [batch_feat], include_user=False).squeeze(0)
            # 归一化embedding用于推理
            batch_emb = batch_emb / (batch_emb.norm(dim=-1, keepdim=True) + 1e-8)
            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))
        # 合并所有批次的结果并保存
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))
    def compute_infonce_loss(self, seq_embs, pos_embs, neg_embs, loss_mask):
        """
        计算InfoNCE loss（向量化优化版本）
        
        Args:
            seq_embs: 序列embedding，形状为 [batch_size, maxlen, hidden_size]
            pos_embs: 正样本embedding，形状为 [batch_size, maxlen, hidden_size]
            neg_embs: 负样本embedding，形状为 [batch_size, maxlen, hidden_size]
            loss_mask: 损失掩码，形状为 [batch_size, maxlen]
            
        Returns:
            loss: InfoNCE loss
        """
        batch_size, maxlen, hidden_size = seq_embs.shape
        # 1. 批量归一化所有embedding（向量化操作）
        seq_embs_norm = seq_embs / (seq_embs.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        pos_embs_norm = pos_embs / (pos_embs.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        neg_embs_norm = neg_embs / (neg_embs.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        # 2. 计算正样本相似度（向量化）
        pos_logits = torch.sum(seq_embs_norm * pos_embs_norm, dim=-1, keepdim=True)
        # 3. 计算负样本相似度（向量化矩阵乘法）
        # 将neg_embs重塑为 [batch_size * maxlen, hidden_size]
        neg_embs_flat = neg_embs_norm.view(-1, hidden_size)
        # 计算所有seq_embs与所有neg_embs的相似度
        neg_logits = torch.matmul(seq_embs_norm.view(-1, hidden_size), neg_embs_flat.t())
        # 重塑回 [batch_size, maxlen, batch_size * maxlen]
        neg_logits = neg_logits.view(batch_size, maxlen, -1)
        # 4. 拼接正负样本logits
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        # 5. 应用mask和温度系数（向量化）
        logits = logits[loss_mask.bool()] / self._temp
        # 6. 创建标签（向量化）
        labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.long)
        # 7. 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)
        return loss