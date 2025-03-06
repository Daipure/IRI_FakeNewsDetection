import sys
sys.path.append('/home/server/Dai/IRI')
sys.path.append('/home/server/Dai/IRI/model')
import os
import json
import nltk
from tqdm import tqdm
from data_preprocessing.load_data import *
from utils.get_llm_respond import *

def get_prompt(News_Content, comment, parent_comment=None):
    if parent_comment is None:
        Prompt = """
        Please analyze the stance and emotional state of the following comments on the news:
        News: {news_content}
        Comment: {comment}
        Brief explanation of your reasoning, considering the relationship to the news content.
        """
        Prompt = Prompt.format(news_content=News_Content, comment=comment)
    else:
        Prompt = """
        Please analyze the stance and emotional state of the following comments to another comment:
        News: {news_content}
        Parent Comment: {parent_comment}
        Response Comment: {comment}
        Brief explanation of your reasoning, considering the relationship to the parent comment and the news content.
        """
        Prompt = Prompt.format(news_content=News_Content, parent_comment=parent_comment, comment=comment)
    return Prompt

def get_one_emotion(data):
    father = data['father']
    res = data['res']
    emotions = []

    news_content = res[0]
    news_content = construct_length(news_content, 6000)
    
    for index, comment in enumerate(res):
        if father[index] == None:
            emotions.append(None)
        elif res[index] == "No comment":
            emotions.append(None)
        elif father[index] == 0:
            prompt = get_prompt(news_content, comment)
            out = get_completion(prompt)
            emotions.append(out)
        else:
            parent_comment = res[father[index]]
            prompt = get_prompt(news_content, comment, parent_comment)
            out = get_completion(prompt)
            emotions.append(out)
    
    assert len(emotions) == len(res)
    return emotions

# 其餘函數保持不變
def process_dataset(dataset_name, subset):
    folder_path = f'/home/server/Dai/IRI/data/reaction_best/{dataset_name}/{subset}'
    save_dir = f'/home/server/Dai/IRI/data/emotion_best/{dataset_name}/{subset}'
    os.makedirs(save_dir, exist_ok=True)
    
    file_list = os.listdir(folder_path)
    
    for file_name in tqdm(file_list):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_path = os.path.join(folder_path, file_name)
            save_path = f'{save_dir}/{file_name}'
            
            if os.path.exists(save_path):
                continue
            
            with open(file_path, 'r') as file:
                data = json.load(file)        
            comment_emotions = get_one_emotion(data)
            json.dump(comment_emotions, open(save_path, 'w'))

# 處理 gossipcop 數據集的 fake 和 real 子集
for subset in ['fake', 'real']:
    process_dataset('gossipcop', subset)

# # 處理 pol 數據集的 fake 和 real 子集
# for subset in ['fake', 'real']:
#     process_dataset('politifact', subset)