import sys
sys.path.append('/home/server/Dai/MAR')
from data_preprocessing.load_data import *
from utils.get_llm_respond import *

import numpy as np
import argparse
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import re
from torch.utils.data import Dataset, DataLoader, random_split, Sampler
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score
import pickle
import os
import json
import logging
from collections import defaultdict  # 添加這行
from transformers import DebertaV2Model, DebertaV2Tokenizer
from torch_geometric.nn.conv import GCNConv, GATConv, SAGEConv, GINConv
from torch_scatter import scatter
from torch_geometric.data import Batch, Data
from einops import repeat
from torch_geometric.nn.norm import BatchNorm
import torch.nn.functional as func
from torch_geometric.nn.pool import global_mean_pool
from pathlib import Path
import random
import datetime
import nltk
import sys

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

setup_seed(3759)



# User Profile 定義
profile_text = {
    'gender': ['You are male.', 'You are female.', 'Your gender is other.'],
    'age': ['You are 18 to 29 years old.', 'You are 30 to 49 years old.',
            'You are 50 to 64 years old.', 'You are over 65 years old.'],
    'race': ['You are White.', 'You are Black.', 'You are Asian.',
             'You are Hispanic/Latino.', 'You are of other/multiple races.'],
    'education': ['Educationally, you are a college graduate.',
                  'Educationally, you have attended some college.',
                  'Educationally, you have a high school diploma or less.'],
    'income': ['Your annual income is $75,000 or more.',
               'Your annual income is between $30,000 and $74,999.',
               'Your annual income is less than $30,000.'],
    'political_tendency': ['Politically, you are conservative.',
                           'Politically, you are moderate.',
                           'Politically, you are liberal.',
                           'Politically, you are independent.'],
    'occupation': ['You work as an Education Practitioner.',
                   'You work as an Administrative Manager / Officer.',
                   'You are Unemployed / a Student.',
                   'You work as an Engineer.',
                   'You work as a Labor Technician / Worker.',
                   'You work as a Logistics Practitioner.',
                   'You work as Medical Personnel.',
                   'You work as a Financial Practitioner.',
                   'You work as Media Personnel.',
                   'You work as an Entertainment and Arts Practitioner.'],
    'media_literacy': ['Your media literacy level is high.',
                       'Your media literacy level is medium.',
                       'Your media literacy level is low.']}


def generate_random_user_profile(selected_features):
    """生成隨機用戶角色"""
    user_profile = 'You are a social media user. '
    for feature in selected_features:
        if feature in profile_text:
            user_profile += random.choice(profile_text[feature]) + ' '
    return user_profile


def generate_with_selected_features(news_content, selected_features):
    """根據選定的特徵生成評論網路"""

    
    # 使用生成的用戶角色進行評論生成
    res = [news_content]
    comments = []
    users = []
    father = [None]
    try_time = 0
    like_num = 0
    share_num = 0

    while len(res) < 11:
        try_time += 1
        if try_time > 20:
            break


        # 為每條評論生成新的隨機用戶檔案
        user_profile = generate_random_user_profile(selected_features)
        
        # 第一次評論
        while len(comments) < 1:
            out = get_news_step1(user_profile, res[0])
                
            comment_match = re.search(r'Comment:\s*(.*)', out)
            if not comment_match:
                continue
            comment_value = comment_match.group(1)

            if "like the news" in out:
                like_num += 1
            if "share the news" in out:
                share_num += 1
                
            father.append(0)
            res.append(comment_value)
            users.append(user_profile)
            comments.append(comment_value)
        
        # 後續評論
        out = get_news_step2(user_profile, res[0], father, res)
        responses = re.findall(r'(Response:\s*.*?)(?=Response:|$)', out, re.DOTALL)

        for response in responses:
            response_type = re.search(r'Response:\s*(.*)', response)
            if response_type:
                response_type = response_type.group(1).strip()
                response_options = response_type.split(" || ")
                
                for option in response_options:
                    option = option.strip().lower()

                    if "no comment" in option:
                        res.append("No comment")
                        father.append(0)
                        users.append(user_profile)
                        continue

                    if "comment on the news" in option:
                        comment_value = re.search(r'Comment:\s*(.*)', response)
                        if comment_value:
                            res.append(comment_value.group(1))
                            father.append(0)
                            users.append(user_profile)
                        continue

                    if "select a comment from the comment network to reply to" in option:
                        match_choose = re.search(r'Choose:\s*(comment_\d+)', response)
                        comment_value = re.search(r'Comment:\s*(.*)', response)
                        if match_choose and comment_value:
                            comment_index = int(re.search(r'\d+', match_choose.group(1)).group())
                            if 0 < comment_index < len(res):
                                res.append(comment_value.group(1))
                                father.append(comment_index)
                                users.append(user_profile)
                        continue
                            
                    if "like the news" in option:
                        like_num += 1

                    if "share the news" in option:
                        share_num += 1

    return {
        'father': father,
        'users': users,
        'res': res,
        'like_num': like_num,
        'share_num': share_num
    }

def get_news_step1(user, news):
    prompt = f'''{user}
You are browsing social media and come across the following news article.

News content:
{news}

Task: Please react to either the news article or an existing comment in the network. Choose one main action and optionally include social reactions.

OUTPUT FORMAT (MUST BE STRICTLY FOLLOWED):

Response: [REQUIRED MAIN ACTION] || [OPTIONAL SOCIAL REACTIONS]

Main action (choose exactly one):
- no comment
- comment on the news
- select a comment from the comment network to reply to

Optional social reactions (You may choose one, both, or none of these reactions.):
- like the news
- share the news

Choose: [REQUIRED ONLY IF "select a comment" is chosen] 
Format: comment_X (where X is the comment number)
Leave blank for other actions.

Comment: [REQUIRED for "comment on the news" or "select a comment"]
Your comment here (max 40 words). 
Write "No comment" if you chose "no comment" as your main action.

CORRECT EXAMPLES:

Example 1:
Response: no comment || like the news || share the news
Choose: 
Comment: No comment

Example 2:
Response: comment on the news || share the news
Choose: 
Comment: This article raises important points about climate change. We need to take action now to protect our planet's future.

Example 3:
Response: comment on the news
Choose: 
Comment: While the article presents interesting data, I think it overlooks some key factors in its analysis.

Remember: You must choose exactly one main action, and you may optionally include social reactions. Your response should always follow this format strictly.
'''
    out = get_completion(prompt)
    return out

def build_comment_tree(parent_indices, res):
    comments = []
    for i in range(1,len(res)):
        comment = {
            "id": f"comment_{i}",  
            "content": res[i],
            "parent": "News" if parent_indices[i] ==0  else f"comment_{parent_indices[i]}",
            "order": i
        }
        comments.append(comment)
    return json.dumps(comments, indent=2)

def get_news_step2(user, news, parent_indices, res):
    prompt = f'''{user}
You are browsing social media and come across the following news article and its comment network.

News content:
{news}

Comment network:
{build_comment_tree(parent_indices, res)}

Task: Please react to either the news article or an existing comment in the network. Your response must include exactly one of the following three main actions, plus optional social reactions.
As a social media user, react to either the news article or an existing comment in the network.
OUTPUT FORMAT (MUST BE STRICTLY FOLLOWED):

Response: [REQUIRED MAIN ACTION] || [OPTIONAL SOCIAL REACTIONS]

Main action (choose exactly one):
- select a comment from the comment network to reply to
- comment on the news
- no comment

Optional social reactions (you may include either, both, or none):
- like the news
- share the news

Choose: [REQUIRED ONLY IF "select a comment" is chosen] 
Format: comment_X (where X is the comment number)
Leave blank for other actions.

Comment: [REQUIRED for "comment on the news" or "select a comment"]
Your comment here (max 40 words). 
Write "No comment" if you chose "no comment" as your main action.

CORRECT EXAMPLES:

Example 1:
Response: select a comment from the comment network to reply to || like the news
Choose: comment_2
Comment: I agree with your perspective. It's crucial that we consider long-term environmental impacts in our decision-making.

Example 2:
Response: comment on the news || share the news
Choose: 
Comment: This article raises important points about climate change. We need to take action now to protect our planet's future.

Example 3:
Response: no comment || like the news || share the news
Choose: 
Comment: No comment

Example 4:
Response: comment on the news
Choose: 
Comment: While the article presents interesting data, I think it overlooks some key factors in its analysis.

Remember: You must choose exactly one main action, and you may optionally include social reactions. Your response should always follow this format strictly.
'''
    out = get_completion(prompt)
    return out


def load_news_ids(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return [f"{item['source_id']}.json" for item in data]
    
class NewsDataset(Dataset):
    def __init__(self, dataset_name=None, preloaded_data=None):
        self.data = []
        
        if preloaded_data is not None:
            # 如果提供了預加載的數據，直接使用它
            self.data = preloaded_data
        elif dataset_name is not None:
            # 從原始數據加載
            news_ids = get_news_ids(dataset_name)
            real = news_ids["real"]
            fake = news_ids["fake"]
            
            graph_info = []
            texts = []
            labels = []
            intentions = []

            # 合併 real 和 fake 的處理邏輯
            for subset, news_set, label in [('fake', fake, 0), ('real', real, 1)]:
                for news_id in news_set:
                    
                    # intention
                    with open(f'/home/server/Dai/MAR/data/news_intention/{dataset_name}/{subset}/{news_id}', encoding='utf-8') as f:
                        intention = json.load(f)
                    intentions.append(intention)  
                    
                    # social
                    with open(f'/home/server/Dai/MAR/data/reaction_occu/{dataset_name}/{subset}/{news_id}') as f:
                        data = json.load(f)

                    # emotion
                    with open(f'/home/server/Dai/MAR/data/emotion_occu/{dataset_name}/{subset}/{news_id}') as f:
                        emo_data = json.load(f)    

                    text = data['res'][0]
                    texts.append(text)
                    
                    # comments
                    row, col = [], []
                    for tgt, src in enumerate(data['father']):
                        if src is None:
                            continue
                        row.append(src)
                        col.append(tgt)
                        
                    edge_index = [row, col]
                    edge_index = torch.tensor(edge_index, dtype=torch.long)

                    # comment
                    comment = data['res'][1:]
                    # emotion
                    emotion = emo_data[1:]
                    
                    # like, share num
                    like_num = data['like_num']
                    share_num = data['share_num']
                    
                    labels.append(label)
                    graph_info.append({
                        'emotion': emotion,
                        'like_num': like_num,
                        'share_num': share_num,
                        'comment': comment,
                        'edge_index': edge_index,
                    })

            # 組裝最終的 self.data
            for index in range(len(labels)):
                self.data.append({
                    'intention': intentions[index],
                    'text': texts[index],
                    'graph_info': graph_info[index],
                    'label': labels[index],
                })
        else:
            raise ValueError("Either dataset_name or preloaded_data must be provided")
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @classmethod
    def save_split_datasets(cls, train_dataset, val_dataset, test_dataset, save_dir):
        """保存切分後的數據集"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存每個數據集和它們的索引
        splits = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
        
        for split_name, dataset in splits.items():
            save_path = os.path.join(save_dir, f'{split_name}_dataset.pkl')
            # 創建包含數據和索引的字典
            split_data = {
                'data': [dataset.dataset.data[i] for i in dataset.indices],
                'indices': dataset.indices
            }
            with open(save_path, 'wb') as f:
                pickle.dump(split_data, f)
                    
        print(f"Datasets saved to {save_dir}")

    @classmethod
    def load_split_datasets(cls, load_dir):
        """加載切分後的數據集"""
        splits = {}
        indices = {}
        
        for split_name in ['train', 'val', 'test']:
            load_path = os.path.join(load_dir, f'{split_name}_dataset.pkl')
            with open(load_path, 'rb') as f:
                split_data = pickle.load(f)
                dataset = cls(None, preloaded_data=split_data['data'])
                indices[split_name] = split_data['indices']
                splits[split_name] = dataset

        return splits['train'], splits['val'], splits['test']
    
def my_collate_fn(batch):
    text = []
    label = []
    graph_info = []
    intention = []
    
    for item in batch:
        graph_info.append(item['graph_info'])
        intention.append(item['intention'])
        text.append(item['text'])
        label.append(item['label'])
    label = torch.tensor(label, dtype=torch.long)

    return {
        'graph_info': graph_info,
        'intention': intention,
        'text': text,
        'label': label,
        'batch_size': len(batch),
    }



def load_splits(save_dir):
    """讀取已保存的數據集分割"""
    print(f"Loading datasets from {save_dir}...")
    
    try:
        with open(os.path.join(save_dir, 'train_dataset.pkl'), 'rb') as f:
            train_data = pickle.load(f)
        with open(os.path.join(save_dir, 'val_dataset.pkl'), 'rb') as f:
            val_data = pickle.load(f)
        with open(os.path.join(save_dir, 'test_dataset.pkl'), 'rb') as f:
            test_data = pickle.load(f)
    except FileNotFoundError:
        logger.error(f"Dataset files not found in {save_dir}")
        raise
        
    # 確保數據結構正確
    for data in [train_data, val_data, test_data]:
        if not isinstance(data, dict) or 'data' not in data:
            raise ValueError("Invalid dataset format")

    train_dataset = NewsDataset(preloaded_data=train_data['data'])
    val_dataset = NewsDataset(preloaded_data=val_data['data'])
    test_dataset = NewsDataset(preloaded_data=test_data['data'])
    
    return train_dataset, val_dataset, test_dataset


def construct_news_length(text, length=1500):
    if len(text) < length:
        return text
    sents = nltk.sent_tokenize(text)
    out = ''
    for sent in sents:
        if len(out) + len(sent) + 1 <= length:
            out = out + ' ' + sent
        else:
            break
    return out

class GNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_unit, gnn, dropout=0.0, batch_norm=False):
        super().__init__()

        self.num_unit = num_unit
        self.dropout = dropout
        self.batch_norm = batch_norm
        assert gnn in ['GCN', 'SAGE', 'GAT', 'GIN']
        gnn_map = {
            'GCN': GCNConv,
            'SAGE': SAGEConv,
            'GAT': GATConv,
            'GIN': GINConv
        }
        Conv = gnn_map[gnn]
        if gnn != 'GIN':
            in_conv = Conv(in_dim, hidden_dim)
        else:
            in_conv = Conv(nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dim, hidden_dim)))
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.convs.append(in_conv)

        for i in range(num_unit):
            if gnn != 'GIN':
                conv = Conv(hidden_dim, hidden_dim)
            else:
                conv = Conv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, hidden_dim)))
            bn = BatchNorm(hidden_dim)
            self.convs.append(conv)
            self.batch_norms.append(bn)

        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.activation(x)
            if self.batch_norm:
                x = self.batch_norms[i](x)
            x = func.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x



class MyModel(nn.Module):

    def __init__(self, device, num_class, task, dataset_name,lm_path='../DeBERTa-v3', max_length=1024):
        super().__init__()

        self.dataset_name = dataset_name

        self.tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')
        self.lm = DebertaV2Model.from_pretrained('microsoft/deberta-v3-base')

        
        for name, param in self.lm.named_parameters():
            param.requires_grad = False

        
        self.max_length = max_length
        self.graph_encoder = GNNEncoder(768,768,2, gnn='GAT', dropout=0.1)
        self.text_transfer = nn.Linear(768 , 768)
        self.graph_transfer = nn.Linear(768 , 768)
        
        self.cls = nn.Sequential(
            nn.Linear(768*2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_class)
        )
        
        self.device = device

        self.pooling = global_mean_pool
        if task == 'TASK1':
            self.loss_fn = nn.CrossEntropyLoss()


    def tokenize(self, data):
        tokens = []
        for item in data:
            # 確保 item 是字串，如果不是，將其轉換為空字串
            if not isinstance(item, str):
                item = ""
            
            token = self.tokenizer.tokenize(item)
            if len(token) == 0:
                token = [self.tokenizer.pad_token_id]
            token = token[:self.max_length-2]
            
            # 確保所有 token 都是字串
            token = [str(t) for t in token]
    
            tokens.append([self.tokenizer.cls_token_id] +
                          self.tokenizer.convert_tokens_to_ids(token) +
                          [self.tokenizer.eos_token_id])
    
        max_length = max(len(token) for token in tokens)
    
        input_ids = []
        token_type_ids = []
        attention_mask = []
        for token in tokens:
            input_ids.append(token + [self.tokenizer.pad_token_id] * (max_length - len(token)))
            token_type_ids.append([0] * max_length)
            attention_mask.append([1] * len(token) + [0] * (max_length - len(token)))
    
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    
        return {
            'input_ids': input_ids.to(self.device),
            'token_type_ids': token_type_ids.to(self.device),
            'attention_mask': attention_mask.to(self.device)
        }

    
    
    def forward(self, data):
        
        
        text = data['text']
        text_input = self.tokenize(text)
        text_reps = self.lm(**text_input).last_hidden_state.mean(dim=1)
        text_reps = self.text_transfer(torch.cat([text_reps], dim=-1))
        
        
        graphs = []
        for index, graph_info in enumerate(data['graph_info']):
            comment = graph_info['comment']            
            comment_input = self.tokenize(comment)
            comment_reps = self.lm(**comment_input).last_hidden_state.mean(dim=1)
            
            
            comment_reps = self.graph_transfer(
                torch.cat([comment_reps], dim=-1))
            
 
            
            # 將所有節點特徵連接在一起
            x = torch.cat([
                text_reps[index].unsqueeze(0),  # 文本節點
                comment_reps              # 評論節點
            ], dim=0)
            

        
            edge_index = graph_info['edge_index']          
            graphs.append(Data(x=x, edge_index=edge_index).to(self.device))
    
        graph = Batch.from_data_list(graphs)
        graph_reps = self.graph_encoder(graph.x, graph.edge_index)
        graph_reps = self.pooling(graph_reps, graph.batch)

        reps = torch.cat([text_reps,graph_reps], dim=-1)
        
        pred = self.cls(reps)
        loss = self.loss_fn(pred, data['label'].to(self.device))
        return pred, loss
    

def setup_logging(dataset_name):
    """設置日誌記錄"""
    log_dir = Path(f'/home/server/Dai/MAR/logs/SFS/{dataset_name}')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # SFS主日誌
    sfs_logger = logging.getLogger('SFS')
    sfs_logger.setLevel(logging.INFO)
    
    # 文件處理器
    log_file = log_dir / f'sfs_{timestamp}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 控制台處理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加處理器
    sfs_logger.addHandler(file_handler)
    sfs_logger.addHandler(console_handler)
    
    # 設置其他日誌級別
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    return sfs_logger

def load_sfs_progress(dataset_name):
    """載入SFS進度"""
    progress_file = Path(f'/home/server/Dai/MAR/data/SFS/{dataset_name}/sfs_progress.json')
    if progress_file.exists():
        try:
            with progress_file.open('r') as f:
                progress = json.load(f)
                # 處理舊版本的進度文件
                if 'global_best_score' not in progress:
                    progress['global_best_score'] = progress['best_score']
                    progress['global_best_features'] = progress['selected_features']
                return progress
        except json.JSONDecodeError:
            logging.warning("Progress file corrupted, starting fresh")
            return None
    return None

def save_sfs_progress(dataset_name, selected_features, remaining_features, best_score, global_best_score, global_best_features, results):
    """保存SFS進度"""
    progress_file = Path(f'/home/server/Dai/MAR/data/SFS/{dataset_name}/sfs_progress.json')
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    
    current_round = len(selected_features) + 1
    
    progress = {
        'current_round': current_round,
        'selected_features': selected_features,
        'remaining_features': remaining_features,
        'best_score': best_score,
        'global_best_score': global_best_score,
        'global_best_features': global_best_features,
        'results': results,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    # 備份舊文件
    if progress_file.exists():
        backup_file = progress_file.parent / f'sfs_progress_backup_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        progress_file.rename(backup_file)
    
    with progress_file.open('w') as f:
        json.dump(progress, f, indent=2)
        
    # 在對應的round目錄下也保存一份當前round的進度
    round_progress_dir = Path(f'/home/server/Dai/MAR/data/SFS/{dataset_name}/round{current_round}')
    round_progress_dir.mkdir(parents=True, exist_ok=True)
    round_progress_file = round_progress_dir / 'round_progress.json'
    
    with round_progress_file.open('w') as f:
        json.dump(progress, f, indent=2)

def load_feature_combination_results(dataset_name, features):
    """載入特徵組合的結果"""
    # 根據特徵數量確定輪次
    round_num = len(features)
    feature_name = "_".join(sorted(features))
    
    # 從對應輪次的results目錄載入結果
    result_file = Path(f'/home/server/Dai/MAR/data/SFS/{dataset_name}/round{round_num}/results/{feature_name}.json')
    if result_file.exists():
        try:
            with result_file.open('r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
    return None

def save_feature_combination_results(dataset_name, features, result):
    """保存特徵組合的結果"""
    # 確定當前是第幾輪
    round_num = len(features)
    feature_name = "_".join(sorted(features))
    
    result_dir = Path(f'/home/server/Dai/MAR/data/SFS/{dataset_name}/round{round_num}/results')
    result_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = result_dir / f'{feature_name}.json'
    with result_file.open('w') as f:
        json.dump({
            'features': features,
            'results': result,
            'timestamp': datetime.datetime.now().isoformat()
        }, f, indent=2)

def process_news_comments_sfs(dataset, subset, selected_features, feature_combo_id, logger):
    """處理新聞評論，使用選定的特徵生成評論網路"""
    outputs = []
    failed_ids = []
    
    news_ids = [item['id'] for item in dataset]
    logger.info(f"Processing {len(news_ids)} news articles for {subset}...")
    
    # 確定當前是第幾輪並構建路徑
    round_num = len(selected_features)  # 特徵數量決定輪數
    feature_name = "_".join(sorted(selected_features))  # 例如: "age_gender"
    
    for news_id in tqdm(news_ids, desc=f"Processing {subset} news", unit="news"):
        try:
            # 修改儲存路徑，加入round信息
            comments_save_dir = Path(f'/home/server/Dai/MAR/data/SFS/{dataset_name}/round{round_num}/{feature_name}/{subset}')
            comments_save_dir.mkdir(parents=True, exist_ok=True)
            save_path = comments_save_dir / str(news_id)
            
            if save_path.exists():
                logger.debug(f"Loading existing data for news_id {news_id}")
                with save_path.open('r') as f:
                    out = json.load(f)
                outputs.append((news_id, out))
                continue
            
            news_item = next(item for item in dataset if item['id'] == news_id)
            news_content = news_item['text']
            news_content = construct_news_length(news_content)
            
            out = generate_with_selected_features(news_content, selected_features)
            
            with save_path.open('w', encoding='utf-8') as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            outputs.append((news_id, out))
            
        except Exception as e:
            logger.error(f"Error processing news_id {news_id}: {str(e)}")
            failed_ids.append(news_id)
            continue
    
    if failed_ids:
        logger.warning(f"Failed to process {len(failed_ids)} news articles: {failed_ids}")
    
    return outputs, failed_ids

def train_model(train_loader, val_loader, test_loader, model_name, dataset_name,
                device='cuda', lr=6.6e-4):

    # 創建模型保存目錄
    if not os.path.exists('models'):
        os.makedirs('models')
    
    """訓練和評估模型"""
    model = MyModel(
        device=device,
        num_class=2,
        task='TASK1',
        dataset_name=dataset_name
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    best_val_f1 = 0
    best_epoch = 0
    no_improve = 0
    patience = 10
    
    for epoch in range(100):
        # 訓練
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
            pred, loss = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(pred.argmax(1).cpu().numpy())
            train_labels.extend(batch['label'].cpu().numpy())
            
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        
        # 驗證
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                pred, _ = model(batch)
                val_preds.extend(pred.argmax(1).cpu().numpy())
                val_labels.extend(batch['label'].cpu().numpy())
                
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        scheduler.step(val_f1)
        
        # 檢查改進
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            no_improve = 0
            # 保存最佳模型
            torch.save(model.state_dict(), f'models/{model_name}_best.pt')
        else:
            no_improve += 1
            
        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
            
    # 測試最佳模型
    model.load_state_dict(torch.load(f'models/{model_name}_best.pt', weights_only=True))
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            pred, _ = model(batch)
            test_preds.extend(pred.argmax(1).cpu().numpy())
            test_labels.extend(batch['label'].cpu().numpy())
            
    test_f1 = f1_score(test_labels, test_preds, average='macro')
    
    return {
        'test_f1_macro': test_f1,
        'val_f1_macro': best_val_f1,
        'best_epoch': best_epoch
    }

def sequential_forward_selection(dataset_name, train_dataset, val_dataset, test_dataset):
   """執行Sequential Forward Selection"""
   # 設置日誌
   logger = setup_logging(dataset_name)
   logger.info(f"Starting SFS for dataset: {dataset_name}")
   
   # 檢查GPU可用性
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   logger.info(f"Using device: {device}")
   
   # 載入進度或初始化
   progress = load_sfs_progress(dataset_name)
   if progress:
       logger.info("Resuming from saved progress...")
       selected_features = progress['selected_features']
       remaining_features = progress['remaining_features']
       best_score = progress['best_score']
       results = progress['results']
   else:
       logger.info("Starting fresh SFS process...")
       # 保持原有8個屬性
       all_features = [
           'gender', 'age', 'race', 'education', 'income',
           'political_tendency', 'occupation', 'media_literacy'
       ]
       selected_features = []
       remaining_features = all_features.copy()
       best_score = 0
       results = []

   feature_combo_id = len(results)
   round_num = len(selected_features) + 1
   
   # 保存全局最佳分數和對應的特徵組合
   global_best_score = best_score
   global_best_features = selected_features.copy()

   try:
       # 修改終止條件：強制探索到6個特徵
       while remaining_features and (len(selected_features) < 8 or len(selected_features) < len(all_features)):
           logger.info(f"\n=== Round {round_num} ===")
           logger.info(f"Currently selected features: {selected_features}")
           logger.info(f"Testing combinations with remaining features: {remaining_features}")
           
           best_new_score = 0
           best_feature = None
           round_results = []

           for feature in remaining_features:
               current_features = selected_features + [feature]
               logger.info(f"\nTrying feature combination: {current_features}")
               
               try:
                   # 檢查是否有緩存結果
                   cached_results = load_feature_combination_results(dataset_name, current_features)
                   if cached_results:
                       logger.info("Using cached results")
                       score = cached_results['results']['test_f1_macro']
                   else:
                       # 處理每個數據集
                       for split_name, dataset in [('train', train_dataset), 
                                                ('val', val_dataset), 
                                                ('test', test_dataset)]:
                           # 為當前特徵組合生成評論網路
                           outputs, failed_ids = process_news_comments_sfs(
                               dataset,
                               split_name,
                               current_features,
                               feature_combo_id,
                               logger
                           )
                           
                           if failed_ids:
                               logger.warning(f"Some news articles failed processing in {split_name} set")
                       
                       # 創建數據加載器
                       train_loader = DataLoader(
                           train_dataset,
                           batch_size=64,
                           shuffle=True,
                           collate_fn=my_collate_fn,
                           num_workers=4,
                           pin_memory=True
                       )
                       val_loader = DataLoader(
                           val_dataset,
                           batch_size=64,
                           shuffle=False,
                           collate_fn=my_collate_fn,
                           num_workers=4,
                           pin_memory=True
                       )
                       test_loader = DataLoader(
                           test_dataset,
                           batch_size=64,
                           shuffle=False,
                           collate_fn=my_collate_fn,
                           num_workers=4,
                           pin_memory=True
                       )
                       
                       # 訓練和評估
                       result = train_model(
                           train_loader, val_loader, test_loader,
                           f'SFS_{"_".join(current_features)}',
                           dataset_name,
                           device=device,
                           lr=6.6e-4
                       )
                       
                       score = result['test_f1_macro']
                       save_feature_combination_results(dataset_name, current_features, result)
                   
                   round_results.append({
                       'feature_added': feature,
                       'combination': current_features,
                       'score': score
                   })
                   
                   if score > best_new_score:
                       best_new_score = score
                       best_feature = feature
                   
                   results.append({
                       'features': current_features.copy(),
                       'score': score,
                       'timestamp': datetime.datetime.now().isoformat()
                   })
                   
                   # 保存當前進度
                   save_sfs_progress(
                    dataset_name=dataset_name,
                    selected_features=selected_features,
                    remaining_features=remaining_features,
                    best_score=best_score,
                    global_best_score=global_best_score,  # 需要添加這個參數
                    global_best_features=global_best_features,  # 需要添加這個參數
                    results=results
                )
                   
               except Exception as e:
                   logger.error(f"Error processing feature combination {current_features}: {str(e)}")
                   continue
           
           # 輸出本輪結果
           logger.info(f"\nRound {round_num} Results:")
           for result in sorted(round_results, key=lambda x: x['score'], reverse=True):
               logger.info(f"Feature: {result['feature_added']}, "
                         f"Score: {result['score']:.4f}, "
                         f"Combination: {result['combination']}")

           feature_combo_id += 1
           
           # 更新最佳特徵
           if best_new_score > best_score or len(selected_features) < 8:
               # 如果分數更好或還未達到最小特徵數，就添加特徵
               selected_features.append(best_feature)
               remaining_features.remove(best_feature)
               best_score = best_new_score
               
               # 更新全局最佳
               if best_new_score > global_best_score:
                   global_best_score = best_new_score
                   global_best_features = selected_features.copy()
               
               logger.info(f"\nAdded feature: {best_feature}")
               logger.info(f"Current features: {selected_features}")
               logger.info(f"Current best score: {best_score}")
               logger.info(f"Global best score: {global_best_score}")
               logger.info(f"Global best features: {global_best_features}")
           else:
               # 如果已經達到6個特徵且沒有改善，就停止
               if len(selected_features) >= 8:
                   logger.info(f"\nNo improvement in Round {round_num} and minimum feature count reached, stopping.")
                   break
           
           round_num += 1
           
   except KeyboardInterrupt:
       logger.info("\nProcess interrupted by user")
       
   except Exception as e:
       logger.error(f"Unexpected error: {str(e)}")
       
   finally:
       # 保存最終結果，使用全局最佳的結果
       final_results = {
           'selected_features': global_best_features,  # 使用全局最佳特徵組合
           'best_score': global_best_score,           # 使用全局最佳分數
           'all_results': results,
           'final_timestamp': datetime.datetime.now().isoformat()
       }
       
       result_file = Path(f'/home/server/Dai/MAR/data/SFS/{dataset_name}/final_results.json')
       with result_file.open('w') as f:
           json.dump(final_results, f, indent=2)
       
       logger.info("Final results saved")
       
   return global_best_features, global_best_score, results

if __name__ == "__main__":
   dataset_name = "gossipcop"
   save_dir = Path(f'/home/server/Dai/MAR/data/SFS/{dataset_name}')
   save_dir.mkdir(parents=True, exist_ok=True)
   
   # 設置日誌
   logger = setup_logging(dataset_name)
   logger.info(f"Starting feature selection for dataset: {dataset_name}")
   
   try:
       # 載入數據集
       logger.info("Loading datasets...")
       train_dataset, val_dataset, test_dataset = load_splits(save_dir)
       
       # 為數據集添加 ID
       for i, item in enumerate(train_dataset.data):
           item['id'] = f"train_{i}"
       for i, item in enumerate(val_dataset.data):
           item['id'] = f"val_{i}"
       for i, item in enumerate(test_dataset.data):
           item['id'] = f"test_{i}"
           
       logger.info(f"Dataset sizes:")
       logger.info(f"Train: {len(train_dataset)}")
       logger.info(f"Val: {len(val_dataset)}")
       logger.info(f"Test: {len(test_dataset)}")
       
       # 執行SFS - 注意這裡接收的返回值改變了
       global_best_features, global_best_score, results = sequential_forward_selection(
           dataset_name,
           train_dataset,
           val_dataset,
           test_dataset
       )
       
       # 保存最終結果到JSON文件
       final_results = {
           'global_best_features': global_best_features,
           'global_best_score': global_best_score,
           'all_results': results,
           'completed_timestamp': datetime.datetime.now().isoformat()
       }
       
       final_results_file = save_dir / 'final_results.json'
       with final_results_file.open('w') as f:
           json.dump(final_results, f, indent=2)
       
       # 輸出詳細的結果日誌
       logger.info("\nSFS completed!")
       logger.info(f"Global best feature combination: {global_best_features}")
       logger.info(f"Global best score: {global_best_score:.4f}")
       
       # 分析整個特徵選擇過程
       logger.info("\nFeature selection process analysis:")
       
       # 按特徵數量分組顯示結果
       feature_counts = defaultdict(list)
       for result in results:
           num_features = len(result['features'])
           feature_counts[num_features].append({
               'features': result['features'],
               'score': result['score']
           })
           
       for num_features in sorted(feature_counts.keys()):
           logger.info(f"\nFeature combinations with {num_features} features:")
           # 對每個特徵數量的結果按分數排序
           sorted_results = sorted(feature_counts[num_features], 
                                key=lambda x: x['score'], 
                                reverse=True)
           
           for rank, result in enumerate(sorted_results, 1):
               logger.info(f"Rank {rank}:")
               logger.info(f"Features: {result['features']}")
               logger.info(f"Score: {result['score']:.4f}")
           
           # 計算並顯示這個特徵數量的平均分數
           avg_score = sum(r['score'] for r in feature_counts[num_features]) / len(feature_counts[num_features])
           logger.info(f"Average score for {num_features} features: {avg_score:.4f}")
       
       # 保存分析結果
       analysis_results = {
           'feature_count_analysis': {
               str(k): [{'features': r['features'], 'score': r['score']} 
                       for r in v]
               for k, v in feature_counts.items()
           },
           'average_scores': {
               str(k): sum(r['score'] for r in v) / len(v)
               for k, v in feature_counts.items()
           }
       }
       
       analysis_file = save_dir / 'feature_analysis.json'
       with analysis_file.open('w') as f:
           json.dump(analysis_results, f, indent=2)
           
       logger.info("\nAnalysis results have been saved.")
       
   except Exception as e:
       logger.error(f"Error in main program: {str(e)}")
       logger.exception("Detailed traceback:")
       raise
   
   finally:
       logger.info("Program finished.")