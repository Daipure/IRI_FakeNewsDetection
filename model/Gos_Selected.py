import sys
sys.path.append('/home/server/Dai/IRI')
from data_preprocessing.load_data import *
from utils.get_llm_respond import *

import numpy as np
import argparse
import time
from tqdm import tqdm
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split, Sampler
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score
import pickle
import os
import json


from transformers import BertModel, BertTokenizer

from torch_geometric.nn.conv import GCNConv, GATConv, SAGEConv, GINConv
from torch_scatter import scatter
from torch_geometric.data import Batch, Data
from einops import repeat
from torch_geometric.nn.norm import BatchNorm
import torch.nn.functional as func
from torch_geometric.nn.pool import global_mean_pool

import random
        
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchIRIk = False
    torch.backends.cudnn.deterministic = True

setup_seed(3759)
# 設定隨機種子


def load_news_ids(dataset_name):
    if dataset_name == 'politifact':
        file_path = '/home/server/Dai/IRI/data/politifact/split.json'
    elif dataset_name == 'gossipcop':
        file_path = '/home/server/Dai/IRI/data/gossipcop/split.json'

    with open(file_path, 'r') as file:
        split_id = json.load(file)
        # 從 split_data 中獲取三個列表
        train_ids = split_id['train']  # 訓練集 ID 列表
        val_ids = split_id['val']      # 驗證集 ID 列表
        test_ids = split_id['test']    # 測試集 ID 列表
    return train_ids, val_ids, test_ids
    
        
class NewsDataset(Dataset):
    def __init__(self, dataset_name, spilt_news_ids,profile):
        
        news_ids = get_news_ids(dataset_name)
        real = news_ids["real"]
        fake = news_ids["fake"]


        
        self.data = []
        graph_info = []
        texts = []
        labels = []
        intentions = []

        # 合併 real 和 fake 的處理邏輯
        for subset, news_list, label in [('fake', fake, 0), ('real', real, 1)]:
            for news_id in news_list:

                
                if news_id not in spilt_news_ids:
                    continue
                    
                # intention
                with open(f'/home/server/Dai/IRI/data/news_intention/{dataset_name}/{subset}/{news_id}', encoding='utf-8') as f:
                    intention = json.load(f)
                intentions.append(intention)  


                # social 
                with open(f'/home/server/Dai/IRI/data/reaction_{profile}/{dataset_name}/{subset}/{news_id}') as f:
                    data = json.load(f)

                # emotion
                with open(f'/home/server/Dai/IRI/data/emotion_{profile}/{dataset_name}/{subset}/{news_id}') as f:
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
                emotion  = emo_data[1:]

                # like, share num (optional if not needed)
                like_num = data['like_num']
                share_num = data['share_num']
                
                labels.append(label)
                graph_info.append({
                    'emotion': emotion,
                    'like_num': like_num,
                    'share_num':share_num,
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
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


    
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

def get_like_share(dataset_name):
    all_like_nums = []
    all_share_nums = []
    news_ids = get_news_ids(dataset_name)
    real = news_ids["real"]
    fake = news_ids["fake"]
    
    for subset, news_list, label in [('fake', fake, 0), ('real', real, 1)]:
        for news_id in news_list:
            # social
            with open(f'/home/server/Dai/IRI/data/reaction_occu/{dataset_name}/{subset}/{news_id}') as f:
                data = json.load(f)
            
            like_num = data['like_num']
            share_num = data['share_num']
            all_like_nums.append(like_num)
            all_share_nums.append(share_num)
    
    like_mean = np.mean(all_like_nums)
    like_std = np.std(all_like_nums)
    share_mean = np.mean(all_share_nums)
    share_std = np.std(all_share_nums)    
    return like_mean ,like_std, share_mean,share_std



    
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


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
    
    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0))
        return attn_output.squeeze(0)
        

    
class MyModel(nn.Module):
    def __init__(self, device,  dataset_name, max_length=512):
        super().__init__()

        self.dataset_name = dataset_name

        # Initialize BERT model for text encoding
        self.lm = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

            
        for name, param in self.lm.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        self.cross_attention_intention = MultiHeadCrossAttention(768, 8)
        self.fusion = MultiHeadCrossAttention(768, 8)
        self.max_length = max_length
        self.graph_encoder = GNNEncoder(768,768,2, gnn='GAT', dropout=0.2)
        self.text_transfer = nn.Linear(768 * 2, 768)
        self.graph_transfer = nn.Linear(768* 2 , 768)
        

        self.cls = nn.Sequential(
            nn.Linear(768 * 3, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)    # 改為輸出單一值
        )


        
        self.device = device

        self.pooling = global_mean_pool
        self.loss_fn = nn.BCEWithLogitsLoss()


    def tokenize(self, data):
        tokens = []
        for item in data:
            # Convert None or non-string items to empty string
            if item is None or not isinstance(item, str):
                item = ""
                
            # Tokenize the text
            token = self.tokenizer.tokenize(item)
            
            # Handle empty tokens
            if len(token) == 0:
                token = [self.tokenizer.pad_token]  # Use pad_token instead of pad_token_id
                
            # Truncate if needed
            token = token[:self.max_length-2]
            
            # Convert tokens to ids
            token_ids = self.tokenizer.convert_tokens_to_ids(token)
            
            # Add special tokens
            token_ids = [self.tokenizer.cls_token_id] + token_ids + [self.tokenizer.sep_token_id]
            tokens.append(token_ids)
        
        # Find max length for padding
        max_length = max(len(token) for token in tokens)
        
        # Prepare tensors
        input_ids = []
        token_type_ids = []
        attention_mask = []
        
        for token in tokens:
            padding_length = max_length - len(token)
            
            input_ids.append(token + [self.tokenizer.pad_token_id] * padding_length)
            token_type_ids.append([0] * max_length)
            attention_mask.append([1] * len(token) + [0] * padding_length)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long).to(self.device),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).to(self.device),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long).to(self.device)
        }
    
    
    def forward(self, data):
        
        
        text = data['text']
        text_input = self.tokenize(text)
        text_reps = self.lm(**text_input).last_hidden_state.mean(dim=1)
        
        
        
        intention = data['intention']
        intention_input = self.tokenize(intention)
        intention_reps = self.lm(**intention_input).last_hidden_state.mean(dim=1)
        intention_reps = self.cross_attention_intention(query=text_reps, key=intention_reps, value=intention_reps)
        
        cross_intention_text_reps = self.text_transfer(torch.cat([text_reps, intention_reps], dim=-1))

        
        graphs = []
        for index, graph_info in enumerate(data['graph_info']):
            comment = graph_info['comment']            
            comment_input = self.tokenize(comment)
            comment_reps = self.lm(**comment_input).last_hidden_state.mean(dim=1)
            
            # 加入emotion
            graph_augmentation = graph_info['emotion']
            graph_augmentation_input = self.tokenize(graph_augmentation)
            graph_augmentation_reps = self.lm(**graph_augmentation_input).last_hidden_state.mean(dim=1)
            
            comment_reps = self.graph_transfer(
                torch.cat([comment_reps, graph_augmentation_reps], dim=-1))
            
            #加入like、share這兩個節點輔助偵測          
            like_num = graph_info["like_num"]
            share_num = graph_info["share_num"]
            like_mean ,like_std, share_mean,share_std = get_like_share(self.dataset_name)
        
            # Standardize like and share numbers
            norm_like = (like_num - like_mean) / like_std
            norm_share = (share_num - share_mean) / share_std
            
            # Create standardized like and share node features
            like_node = torch.full((1, 768), norm_like, dtype=torch.float).to(self.device)
            share_node = torch.full((1, 768), norm_share, dtype=torch.float).to(self.device)
            
            
            # 將所有節點特徵連接在一起
            x = torch.cat([
                text_reps[index].unsqueeze(0),  # 文本節點
                comment_reps,              # 評論節點
                like_node,             # like 節點
                share_node             # share 節點
            ], dim=0)
            

        
            edge_index = graph_info['edge_index']
                
            # 添加 like 和 share 節點的邊
            num_nodes = x.shape[0]
            like_node_index = num_nodes - 2
            share_node_index = num_nodes - 1
            
            # 創建新的邊，將 like 和 share 節點連接到文本節點（假設文本節點的索引為 0）
            new_edges = torch.tensor([[like_node_index, share_node_index],
                                      [0, 0]], dtype=torch.long)
        
            # 將新的邊加入到原有的 edge_index 中
            edge_index = torch.cat([edge_index, new_edges], dim=1)            
            graphs.append(Data(x=x, edge_index=edge_index).to(self.device))
    
        graph = Batch.from_data_list(graphs)
        graph_reps = self.graph_encoder(graph.x, graph.edge_index)
        graph_reps = self.pooling(graph_reps, graph.batch)
        


        fusion_reps = self.fusion(query=graph_reps, key=cross_intention_text_reps, value=cross_intention_text_reps)
        pred = self.cls(torch.cat([cross_intention_text_reps, fusion_reps,graph_reps], dim=-1))
        loss = self.loss_fn(pred, data['label'].float().unsqueeze(1).to(self.device))
        return pred, loss
        

import json

from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score,f1_score
from argparse import ArgumentParser

device = torch.device('cuda')

    


def get_metrics(truth, preds):
    accuracy = accuracy_score(truth, preds)
    f1_macro = f1_score(truth, preds, average='macro')
    f1_real = f1_score(truth, preds, average='binary', pos_label=1)  # 假設 1 代表 real
    f1_fake = f1_score(truth, preds, average='binary', pos_label=0)  # 假設 0 代表 fake
    return accuracy, f1_macro, f1_real, f1_fake

def train_one_epoch(model, optimizer, loader, epoch):
    model.train()
    ave_loss = 0
    cnt = 0
    all_truth = []
    all_preds = []
    pbar = tqdm(loader, desc='train {} epoch'.format(epoch), leave=False)
    for batch in pbar:
        optimizer.zero_grad()
        out, loss = model(batch)

        preds = out.gt(0).float().to('cpu')
        truth = batch['label'].to('cpu')
        loss.backward()
        optimizer.step()
        ave_loss += loss.item() * len(batch)
        cnt += len(batch)
        all_truth.append(truth)
        all_preds.append(preds)
    ave_loss /= cnt
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_truth = torch.cat(all_truth, dim=0).numpy()
    accuracy, f1_macro, f1_real, f1_fake = get_metrics(all_truth, all_preds)
    return ave_loss, accuracy, f1_macro, f1_real, f1_fake

@torch.no_grad()
def validation(model, loader, epoch):
    model.eval()
    all_truth = []
    all_preds = []
    pbar = tqdm(loader, desc='validate {} epoch'.format(epoch), leave=False)
    for batch in pbar:
        out, _ = model(batch)
        preds = out.gt(0).float().to('cpu')
        truth = batch['label'].to('cpu')
        all_truth.append(truth)
        all_preds.append(preds)
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_truth = torch.cat(all_truth, dim=0).numpy()
    accuracy, f1_macro, f1_real, f1_fake = get_metrics(all_truth, all_preds)
    return accuracy, f1_macro, f1_real, f1_fake

def train(train_loader, val_loader, test_loader, name, dataset_name, lr):
    learn_rate = lr
    save_path = 'checkpoints/{}'.format(name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 獲取當前最佳的f1分數
    best_saved_f1 = 0
    for filename in os.listdir(save_path):
        if filename.endswith('.pt'):
            try:
                saved_f1 = float(filename.split('_')[0])
                best_saved_f1 = max(best_saved_f1, saved_f1)
            except:
                continue
    
    print(f"Current best saved F1: {best_saved_f1:.3f}")

    model = MyModel(device=device, dataset_name=dataset_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=3e-5)

    best_f1 = 0
    best_state = model.state_dict()
    for key, value in best_state.items():
        best_state[key] = value.clone()
    no_up = 0
    
    for epoch in range(200):
        # 訓練階段
        train_loss, train_acc, train_f1_macro, train_f1_real, train_f1_fake = train_one_epoch(
            model, optimizer, train_loader, epoch
        )
        
        # 驗證階段
        val_acc, val_f1_macro, val_f1_real, val_f1_fake = validation(model, val_loader, epoch)

        # 根據驗證集F1分數決定早停
        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            no_up = 0
            for key, value in model.state_dict().items():
                best_state[key] = value.clone()
        else:
            if epoch >= 16:
                no_up += 1
        
        # 輸出訓練和驗證集的指標
        print(
            f'Epoch {epoch}:\n'
            f'Train - loss: {train_loss:.3f}, acc: {train_acc:.3f}, '
            f'F1 macro: {train_f1_macro:.3f}, F1 real: {train_f1_real:.3f}, F1 fake: {train_f1_fake:.3f}\n'
            f'Val   - acc: {val_acc:.3f}, F1 macro: {val_f1_macro:.3f}, '
            f'F1 real: {val_f1_real:.3f}, F1 fake: {val_f1_fake:.3f}\n'
            f'Best val F1 so far: {best_f1:.3f}'
        )

        if no_up >= 8:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    # 載入最佳模型狀態並進行最終測試
    model.load_state_dict(best_state)
    final_test_acc, final_test_f1_macro, final_test_f1_real, final_test_f1_fake = validation(
        model, test_loader, 0
    )
    
    print(f'Final Results - LR: {lr}, train {name}:\n'
          f'Best val F1: {best_f1:.3f}\n'
          f'Test metrics - accuracy: {final_test_acc:.3f}, F1 macro: {final_test_f1_macro:.3f}, '
          f'F1 real: {final_test_f1_real:.3f}, F1 fake: {final_test_f1_fake:.3f}')
    
    # 保存模型的邏輯
    if final_test_f1_macro > best_saved_f1 or abs(final_test_f1_macro - best_saved_f1) <= 0.02:
        base_f1_str = f"{final_test_f1_macro:.3f}"
        cnt = 0
        while os.path.exists(f'{save_path}/{base_f1_str}_{cnt}.pt'):
            cnt += 1
        
        save_path_full = f'{save_path}/{base_f1_str}_{cnt}.pt'
        torch.save(best_state, save_path_full)
        print(f"Model saved as: {save_path_full}")
    else:
        print(f"Model not saved. Current F1 ({final_test_f1_macro:.3f}) is not better than best saved F1 ({best_saved_f1:.3f})")

    return {
        'learning_rate': lr,
        'best_val_f1': best_f1,
        'test_acc': final_test_acc,
        'test_f1_macro': final_test_f1_macro,
        'test_f1_real': final_test_f1_real,
        'test_f1_fake': final_test_f1_fake
    }
    
def main():
    batch_size = 64
    
    dataset_name = 'gossipcop'


    # 讀取訓練、測試和驗證的ID
    train_ids,val_ids,test_ids = load_news_ids(dataset_name)


    profile = 'occu'

    # 創建不同的資料集
    train_dataset = NewsDataset(dataset_name=dataset_name, spilt_news_ids=train_ids, profile=profile)
    test_dataset = NewsDataset(dataset_name=dataset_name, spilt_news_ids=test_ids, profile=profile)
    val_dataset = NewsDataset(dataset_name=dataset_name, spilt_news_ids=val_ids, profile=profile)
    

    train_loader = DataLoader(train_dataset, collate_fn=my_collate_fn, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, collate_fn=my_collate_fn, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, collate_fn=my_collate_fn, batch_size=batch_size, shuffle=False)
    
    train_name = 'Gos_Selected'


    

    
    base_learning_rates = [1e-5,2.5e-5,4.0e-5, 5.5e-5, 7.0e-5, 8.5e-5]
    
    # 每個學習率重複2次
    learning_rates = []
    for lr in base_learning_rates:
        learning_rates.extend([lr])

    results = []
    total_runs = len(learning_rates)

    for i, lr in enumerate(learning_rates, 1):
        print(f"Running experiment {i}/{total_runs}: lr = {lr}")
        result = train(train_loader, val_loader, test_loader, train_name, dataset_name, lr)
        result['run_number'] = (i - 1) % 2 + 1  # 記錄是第幾次運行
        results.append(result)
        
        # 每2次實驗後顯示當前學習率的統計信息
        if i % 2 == 0:
            current_lr = lr
            current_lr_results = [r['test_acc'] for r in results[-2:]]
            mean_acc = np.mean(current_lr_results)
            max_acc = max(current_lr_results)
            print(f"\nCurrent LR = {current_lr} statistics:")
            print(f"Mean accuracy: {mean_acc:.4f}")
            print(f"Max accuracy: {max_acc:.4f}\n")

    # 保存結果
    with open('Gos_Selected.json', 'w') as f:
        json.dump(results, f, indent=4)
    

if __name__ == '__main__':
    main()
    
    
    
    
    
