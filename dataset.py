import json
import pickle
import random
import struct
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集

    Args:
        data_dir: 数据文件目录
        args: 全局参数

    Attributes:
        data_dir: 数据文件目录
        maxlen: 最大长度
        item_feat_dict: 物品特征字典
        mm_emb_ids: 激活的mm_emb特征ID
        mm_emb_dict: 多模态特征字典
        itemnum: 物品数量
        usernum: 用户数量
        indexer_i_rev: 物品索引字典 (reid -> item_id)
        indexer_u_rev: 用户索引字典 (reid -> user_id)
        indexer: 索引字典
        feature_default_value: 特征缺省值
        feature_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feat_statistics: 特征统计信息，包括user和item的特征数量
    """

    def __init__(self, data_dir, args):
        """
        初始化数据集
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self._load_data_and_offsets()
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id
        self.sl_alpha = getattr(args, "sl_alpha", 1.8)  # 新增：SL 超参数 α
        self.sl_maxlen = int(self.maxlen ** (2.0 / self.sl_alpha))  # 计算最大子序列长度
        self.training = False  # 默认推理阶段
        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer
        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()
        self.emb_shape = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}

    def add_time_features(self, user_sequence, tau=86400):
        ts_array = np.array([r[5] for r in user_sequence], dtype=np.int64)  # 提取 timestamp
        # 时间差计算
        prev_ts_array = np.roll(ts_array, 1)
        prev_ts_array[0] = ts_array[0]  # 第一个元素无 prev，使用自身
        time_gap = ts_array - prev_ts_array
        time_gap[0] = 0
        log_gap = np.log1p(time_gap)  # log(1 + time_gap)
        # 小时、星期、月份
        ts_utc = ts_array + 8 * 3600  # 假设 UTC+8 时区调整，根据数据调整
        hours = (ts_utc % 86400) // 3600
        weekdays = ((ts_utc % 86400 + 4) % 7).astype(np.int32)  # 周几 (0=周日, 1=周一...)
        months = pd.to_datetime(ts_array, unit='s').month.to_numpy()  # 月份 (1-12)
        # 时间衰减 (delta_t scaled)
        last_ts = ts_array[-1]
        delta_t = last_ts - ts_array
        delta_scaled = np.log1p(delta_t / tau)
        # 归一化时间连续特征
        time_gap_max = np.max(time_gap)
        log_gap_max = np.max(log_gap)
        delta_scaled_max = np.max(delta_scaled)
        time_gap = time_gap / (time_gap_max + 1e-8)
        log_gap = log_gap / (log_gap_max + 1e-8)
        delta_scaled = delta_scaled / (delta_scaled_max + 1e-8)
        # 新序列
        new_sequence = []
        for idx, record in enumerate(user_sequence):
            u, i, user_feat, item_feat, action_type, ts = record
            if user_feat is None:
                user_feat = {}  # 如果为空，初始化
            if item_feat is None:
                item_feat = {}  # 如果为空，初始化
            # 添加到 user_feat（先加这里）
            user_feat["200"] = int(hours[idx])
            user_feat["201"] = int(weekdays[idx])
            user_feat["202"] = int(time_gap[idx])  # 如果作为 continual，可 float
            user_feat["203"] = float(log_gap[idx])
            user_feat["204"] = int(months[idx])
            user_feat["205"] = float(delta_scaled[idx])
            new_sequence.append((u, i, user_feat, item_feat, action_type, ts))
        return ts_array, new_sequence  # 返回 ts_array 以防后续用
    def _transfer_context_features(self, user_feat: dict, item_feat: dict, cols_to_trans: list):
        for col in cols_to_trans:
            item_feat[col] = user_feat[col]
        return item_feat
    def _load_data_and_offsets(self):
        """
        加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        """
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)
    def _load_user_data(self, uid):
        """
        从数据文件中加载单个用户的数据

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        with open(self.data_dir / "seq.jsonl", 'rb') as data_file:
            data_file.seek(self.seq_offsets[uid])
            line = data_file.readline()
            data = json.loads(line)
        return data
    def _random_neq(self, l, r, s):
        """
        生成一个不在序列s中的随机整数, 用于训练时的负采样

        Args:
            l: 随机整数的最小值
            r: 随机整数的最大值
            s: 序列

        Returns:
            t: 不在序列s中的随机整数
        """
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def _apply_stochastic_length(self, seq):
        """
        对序列应用 Stochastic Length 采样。
        Args:
            seq: 原始序列，list 类型
        Returns:
            采样子序列
        """
        L = len(seq)
        if L <= self.sl_maxlen:
            return seq  # 序列较短，不采样
        # 随机均匀采样子序列
        indices = sorted(random.sample(range(L), self.sl_maxlen))
        return [seq[i] for i in indices]

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户ID(reid)

        Returns:
            seq: 用户序列ID
            pos: 正样本ID（即下一个真实访问的item）
            neg: 负样本ID
            token_type: 用户序列类型，1表示item，2表示user
            next_token_type: 下一个token类型，1表示item，2表示user
            next_action_type: 下一个action类型
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            pos_feat: 正样本特征，每个元素为字典，key为特征ID，value为特征值
            neg_feat: 负样本特征，每个元素为字典，key为特征ID，value为特征值
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        # === 训练阶段：应用 Stochastic Length ===
        if hasattr(self, 'sl_alpha') and self.training:  # 训练阶段启用
            user_sequence = self._apply_stochastic_length(user_sequence)
        # 添加时间特征到 user_sequence（先加到 user_feat）
        _, user_sequence = self.add_time_features(user_sequence)
        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, timestamp = record_tuple
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type))
            if i and item_feat:
                # 只在序列特征中添加时间特征，不在pos/neg样本中添加
                cols_to_trans = ['200', '201', '202', '203', '204', '205']
                item_feat = self._transfer_context_features(user_feat or {}, item_feat, cols_to_trans)
                ext_user_sequence.append((i, item_feat, 1, action_type))
        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.maxlen + 1], dtype=object)
        nxt = ext_user_sequence[-1]
        idx = self.maxlen
        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])
        # left-padding, 从后往前遍历，将用户序列填充到maxlen+1的长度
        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type = record_tuple
            next_i, next_feat, next_type, next_act_type = nxt
            feat = self.fill_missing_feat(feat, i)
            next_feat = self.fill_missing_feat(next_feat, next_i)
            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            seq_feat[idx] = feat
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                #  使用原始item特征，不添加时间特征
                original_item_feat = self.item_feat_dict.get(str(next_i), {})
                pos_feat[idx] = self.fill_missing_feat(original_item_feat, next_i)
                neg_id = self._random_neq(1, self.itemnum + 1, ts)
                original_neg_feat = self.item_feat_dict.get(str(neg_id), {})
                neg[idx] = neg_id
                neg_feat[idx] = self.fill_missing_feat(original_neg_feat, neg_id)
            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break
        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)
        all_feat_keys = []
        for ft in self.feature_types.values():
            all_feat_keys.extend(ft)
        seq_feature_list = [seq_feat.tolist()]
        pos_feature_list = [pos_feat.tolist()]
        neg_feature_list = [neg_feat.tolist()]
        seq_tensors = {}
        pos_tensors = {}
        neg_tensors = {}
        for k in set(all_feat_keys) - set(self.feature_types['item_emb']):
            seq_tensors[k] = self.feat2tensor(seq_feature_list, k).squeeze(0)
            pos_tensors[k] = self.feat2tensor(pos_feature_list, k).squeeze(0)
            neg_tensors[k] = self.feat2tensor(neg_feature_list, k).squeeze(0)
        for k in self.feature_types['item_emb']:
            emb_dim = self.emb_shape[k]
            seq_emb_data = np.zeros((self.maxlen + 1, emb_dim), dtype=np.float32)
            for j in range(self.maxlen + 1):
                if seq_feat[j] is not None and k in seq_feat[j]:
                    seq_emb_data[j] = seq_feat[j][k]
            seq_tensors[k] = torch.from_numpy(seq_emb_data)
            pos_emb_data = np.zeros((self.maxlen + 1, emb_dim), dtype=np.float32)
            for j in range(self.maxlen + 1):
                if pos_feat[j] is not None and k in pos_feat[j]:
                    pos_emb_data[j] = pos_feat[j][k]
            pos_tensors[k] = torch.from_numpy(pos_emb_data)
            neg_emb_data = np.zeros((self.maxlen + 1, emb_dim), dtype=np.float32)
            for j in range(self.maxlen + 1):
                if neg_feat[j] is not None and k in neg_feat[j]:
                    neg_emb_data[j] = neg_feat[j][k]
            neg_tensors[k] = torch.from_numpy(neg_emb_data)
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_tensors, pos_tensors, neg_tensors

    def __len__(self):
        """
        返回数据集长度，即用户数量

        Returns:
            usernum: 用户数量
        """
        return len(self.seq_offsets)

    def _init_feat_info(self):
        """
        初始化特征信息, 包括特征缺省值和特征类型

        Returns:
            feat_default_value: 特征缺省值，每个元素为字典，key为特征ID，value为特征缺省值
            feat_types: 特征类型，key为特征类型名称，value为包含的特征ID列表
            feat_statistics: 特征统计信息，包括user和item的特征数量
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        # 原始特征类型定义
        feat_types['user_sparse'] = ['103', '104', '105', '109']
        feat_types['item_sparse'] = [
            '100', '117', '111', '118', '101', '102', '119', '120',
            '114', '112', '121', '115', '122', '116',
        ]
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = []
        feat_types['item_continual'] = []
        # 添加时间特征到特征类型
        time_sparse_features = ['200', '201', '204']
        feat_types['user_sparse'].extend(time_sparse_features)
        feat_types['item_sparse'].extend(time_sparse_features)
        time_continual_features = ['202', '203', '205']
        feat_types['user_continual'].extend(time_continual_features)
        feat_types['item_continual'].extend(time_continual_features)
        # 时间特征统计信息
        time_feat_stats = {
            '200': 24,  # 小时 (0-23)
            '201': 7,  # 星期几 (0-6)
            '204': 12  # 月份 (1-12)
        }
        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = time_feat_stats.get(feat_id, len(self.indexer['f'][feat_id]))
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = time_feat_stats.get(feat_id, len(self.indexer['f'][feat_id]))
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0.0 if feat_id in ['202', '203', '205'] else 0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0.0 if feat_id in ['202', '203', '205'] else 0
        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = np.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
            )
        return feat_default_value, feat_types, feat_statistics

    def feat2tensor(self, seq_feature, k):
        batch_size = len(seq_feature)
        if k in self.feature_types['item_array'] or k in self.feature_types['user_array']:
            max_array_len = 0
            max_seq_len = 0
            for i in range(batch_size):
                seq_data = [item.get(k, self.feature_default_value[k]) for item in seq_feature[i]]
                max_seq_len = max(max_seq_len, len(seq_data))
                if seq_data:
                    max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data))
            batch_data = np.zeros((batch_size, max_seq_len, max_array_len if max_array_len > 0 else 1), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item.get(k, self.feature_default_value[k]) for item in seq_feature[i]]
                for j, item_data in enumerate(seq_data[:max_seq_len]):
                    actual_len = min(len(item_data), max_array_len)
                    batch_data[i, j, :actual_len] = item_data[:actual_len]
            return torch.from_numpy(batch_data)
        else:
            max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
            dtype = np.int64 if k not in self.feature_types['item_continual'] and k not in self.feature_types['user_continual'] else np.float32
            batch_data = np.zeros((batch_size, max_seq_len), dtype=dtype)
            for i in range(batch_size):
                seq_data = [item.get(k, self.feature_default_value[k]) for item in seq_feature[i]]
                batch_data[i, :len(seq_data)] = seq_data
            return torch.from_numpy(batch_data)

    def fill_missing_feat(self, feat, item_id):
        """
        对于原始数据中缺失的特征进行填充缺省值

        Args:
            feat: 特征字典
            item_id: 物品ID

        Returns:
            filled_feat: 填充后的特征字典
        """
        if feat == None:
            feat = {}
        filled_feat = {}
        for k in feat.keys():
            filled_feat[k] = feat[k]
        all_feat_ids = []
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)
        missing_fields = set(all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]
        for feat_id in self.feature_types['item_emb']:
            if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                if type(self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]) == np.ndarray:
                    filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]
        return filled_feat

    @staticmethod
    def collate_fn(batch):
        """
        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            pos: 正样本ID, torch.Tensor形式
            neg: 负样本ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            next_token_type: 下一个token类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            pos_feat: 正样本特征, list形式
            neg_feat: 负样本特征, list形式
        """
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_tensors, pos_tensors, neg_tensors = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        array_keys = ['106', '107', '108', '110']
        def pad_array_tensors(tensors_dict_list, k):
            max_array_len = max(t.shape[1] if len(t.shape) > 1 else 1 for d in tensors_dict_list for t in [d[k]] if k in d)
            padded = []
            for d in tensors_dict_list:
                t = d[k]
                if len(t.shape) == 1:
                    t = t.unsqueeze(1)
                current_len = t.shape[1]
                if current_len < max_array_len:
                    pad = torch.zeros(t.shape[0], max_array_len - current_len, dtype=t.dtype)
                    t = torch.cat([t, pad], dim=1)
                padded.append(t)
            return torch.stack(padded, dim=0)
        seq_feat = {}
        pos_feat = {}
        neg_feat = {}
        for k in seq_tensors[0]:
            if k in array_keys:
                seq_feat[k] = pad_array_tensors(seq_tensors, k)
                pos_feat[k] = pad_array_tensors(pos_tensors, k)
                neg_feat[k] = pad_array_tensors(neg_tensors, k)
            else:
                seq_feat[k] = torch.stack([d[k] for d in seq_tensors], dim=0)
                pos_feat[k] = torch.stack([d[k] for d in pos_tensors], dim=0)
                neg_feat[k] = torch.stack([d[k] for d in neg_tensors], dim=0)
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat


class MyTestDataset(MyDataset):
    """
    测试数据集
    """

    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)
    def _load_data_and_offsets(self):
        self.data_file = open(self.data_dir / "predict_seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)
    def _load_user_data(self, uid):
        """
        从数据文件中加载单个用户的数据（测试集专用）
        
        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        self.data_file.seek(self.seq_offsets[uid])
        line = self.data_file.readline()
        data = json.loads(line)
        return data

    def _process_cold_start_feat(self, feat):
        """
        处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
        """
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if type(feat_value) == list:
                value_list = []
                for v in feat_value:
                    if type(v) == str:
                        value_list.append(0)
                    else:
                        value_list.append(v)
                processed_feat[feat_id] = value_list
            elif type(feat_value) == str:
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        return processed_feat

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户在self.data_file中储存的行号
        Returns:
            seq: 用户序列ID
            token_type: 用户序列类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            time_diff_buckets: 时间差分桶特征
            user_id: user_id eg. user_xxxxxx ,便于后面对照答案
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据
        # === 加入时间特征，保持和训练一致 ===
        _, user_sequence = self.add_time_features(user_sequence)
        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, timestamp = record_tuple
            if u:
                if type(u) == str:  # 如果是字符串，说明是user_id
                    user_id = u
                else:  # 如果是int，说明是re_id
                    user_id = self.indexer_u_rev[u]
            if u and user_feat:
                if type(u) == str:
                    u = 0
                if user_feat:
                    user_feat = self._process_cold_start_feat(user_feat)
                ext_user_sequence.insert(0, (u, user_feat, 2, None))
            if i and item_feat:
                # 序列对于训练时没见过的item，不会直接赋0，而是保留creative_id，creative_id远大于训练时的itemnum
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                # === 关键修改：把时间特征从 user_feat 转移到 item_feat ===
                cols_to_trans = ['200', '201', '202', '203', '204', '205']
                item_feat = self._transfer_context_features(user_feat or {}, item_feat, cols_to_trans)
                ext_user_sequence.append((i, item_feat, 1, None))
        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        idx = self.maxlen
        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])
        for record_tuple in reversed(ext_user_sequence):
            i, feat, type_, _ = record_tuple
            feat = self.fill_missing_feat(feat, i)
            seq[idx] = i
            token_type[idx] = type_
            seq_feat[idx] = feat
            idx -= 1
            if idx == -1:
                break
        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        all_feat_keys = []
        for ft in self.feature_types.values():
            all_feat_keys.extend(ft)
        seq_feature_list = [seq_feat.tolist()]
        seq_tensors = {}
        for k in set(all_feat_keys) - set(self.feature_types['item_emb']):
            seq_tensors[k] = self.feat2tensor(seq_feature_list, k).squeeze(0)
        for k in self.feature_types['item_emb']:
            emb_dim = self.emb_shape[k]
            seq_emb_data = np.zeros((self.maxlen + 1, emb_dim), dtype=np.float32)
            for j in range(self.maxlen + 1):
                if seq_feat[j] is not None and k in seq_feat[j]:
                    seq_emb_data[j] = seq_feat[j][k]
            seq_tensors[k] = torch.from_numpy(seq_emb_data)
        return seq, token_type, seq_tensors, user_id

    def __len__(self):
        """
        Returns:
            len(self.seq_offsets): 用户数量
        """
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            temp = pickle.load(f)
        return len(temp)

    @staticmethod
    def collate_fn(batch):
        """
        将多个__getitem__返回的数据拼接成一个batch

        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            time_diff_buckets: 时间差分桶特征, torch.Tensor形式
            user_id: user_id, str
        """
        seq, token_type, seq_tensors, user_id = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        array_keys = ['106', '107', '108', '110']
        def pad_array_tensors(tensors_dict_list, k):
            max_array_len = max(t.shape[1] if len(t.shape) > 1 else 1 for d in tensors_dict_list for t in [d[k]] if k in d)
            padded = []
            for d in tensors_dict_list:
                t = d[k]
                if len(t.shape) == 1:
                    t = t.unsqueeze(1)
                current_len = t.shape[1]
                if current_len < max_array_len:
                    pad = torch.zeros(t.shape[0], max_array_len - current_len, dtype=t.dtype)
                    t = torch.cat([t, pad], dim=1)
                padded.append(t)
            return torch.stack(padded, dim=0)
        seq_feat = {}
        for k in seq_tensors[0]:
            if k in array_keys:
                seq_feat[k] = pad_array_tensors(seq_tensors, k)
            else:
                seq_feat[k] = torch.stack([d[k] for d in seq_tensors], dim=0)
        return seq, token_type, seq_feat, user_id


def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)

def load_mm_emb(mm_path, feat_ids):
    """
    加载多模态特征Embedding

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
        if not base_path.exists():
            print(f"[Warning] Path not found: {base_path}")
            continue
        for json_file in base_path.glob('part-*'):
            try:
                with open(json_file, 'r', encoding='utf-8') as file:
                    for line in file:
                        try:
                            data = json.loads(line.strip())
                            if 'emb' not in data or 'anonymous_cid' not in data:
                                continue  # 跳过异常行
                            cid = str(data['anonymous_cid'])
                            emb = np.array(data['emb'], dtype=np.float32)
                            emb_dict[cid] = emb
                        except Exception as e:
                            continue  # 静默跳过
            except Exception as e:
                print(f"[Error] Failed to read file {json_file}: {e}")
                continue
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb with {len(emb_dict)} items')
    return mm_emb_dict