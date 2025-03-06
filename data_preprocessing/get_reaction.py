
import os
import random
import json
import torch
from tqdm import tqdm
import re
import numpy as np
import json
import nltk
import sys

sys.path.append('/home/server/Dai/IRI')
sys.path.append('/home/server/Dai/IRI/model')
from utils.get_llm_respond import *

from data_preprocessing.load_data import *

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
                       'Your media literacy level is low.']
}

def generate_a_character():
    profile = 'You are a social media user. '
    for item in profile_text:
        profile += random.choice(profile_text[item]) + ' '
    return profile


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
    

def generate(news):
    torch.cuda.empty_cache()
    res = [news]
    comments = []
    users = []
    father = [None]
    rt = 0
    like_num = 0
    share_num = 0
    try_time = 0

    
    while len(res) < 11:
        try_time += 1
        if try_time > 20:
            break
        
        user = generate_a_character()
        
        while len(comments) < 1:
            out = get_news_step1(user, res[rt])
                
            # 處理評論
            comment_match = re.search(r'Comment:\s*(.*)', out)
            if not comment_match:
                continue
            comment_value = comment_match.group(1)

            # 檢查是否包含 "like the news"
            if "like the news" in out:
                like_num += 1

            # 檢查是否包含 "share the news"
            if "share the news" in out:
                share_num += 1
                
            father.append(rt)
            res.append(comment_value)
            users.append(user)
            comments.append(comment_value)
        
        
        out = get_news_step2(user, res[rt], father, res)

        # 分割輸出結果為多個部分，處理每一個 Response
        responses = re.findall(r'(Response:\s*.*?)(?=Response:|$)', out, re.DOTALL)


        for response in responses:
            response_type = re.search(r'Response:\s*(.*)', response)
            if response_type:
                response_type = response_type.group(1).strip()

                # 分割 response_type 中的多重选项
                response_options = response_type.split(" || ")
                for option in response_options:
                    option = option.strip().lower()


                    # 如果有 "no comment"，確保不會有評論內容
                    if "no comment" in option or "no comment on the news" in option:
                        res.append("No comment")
                        father.append(rt)  # 確保與 res 長度一致
                        users.append(user)
                        continue

                    # 處理 "comment on the news"
                    if "comment on the news" in option:
                        comment_value = re.search(r'Comment:\s*(.*)', response)
                        if comment_value:
                            res.append(comment_value.group(1))
                            father.append(rt)  # 確保更新 father 列表
                            users.append(user)
                        else:
                            print("No valid comment found. Output was:")
                            print(response)
                            continue

                    if "select a comment from the comment network to reply to" in option:
                        match_choose = re.search(r'Choose:\s*(comment_\d+)', response)
                        comment_value = re.search(r'Comment:\s*(.*)', response)
                        if match_choose and comment_value:
                            comment_number = match_choose.group(1)
                            comment_index = int(re.search(r'\d+', comment_number).group())
                            if 0 < comment_index < len(res):  # 確保索引有效
                                res.append(comment_value.group(1))
                                father.append(comment_index)
                                users.append(user)
                            else:
                                print(f"Invalid comment index: {comment_index}. Skipping this response.")
                                continue
                        else:
                            print("No valid comment selection or comment content found. Output was:")
                            print(response)
                            continue
                            
                    
                    # 檢查是否包含 "like the news"
                    if "like the news" in option:
                        like_num += 1

                    # 檢查是否包含 "share the news"
                    if "share the news" in option:
                        share_num += 1
                            



    # 最後檢查父節點列表和評論列表的長度是否一致
    assert len(father) == len(res), f"Length mismatch: father={len(father)}, res={len(res)}"
    # 最後返回按讚和轉發的數量
    return {
        'father': father,  # 每个节点的父节点 ID
        'users': users,
        'res': res,  # 每个节点的评论内容
        'like_num': like_num,  # 按讚的次數
        'share_num': share_num  # 轉發的次數
    }

def process_news_comments(news_ids, dataset, subset):  
    for news_id in tqdm(news_ids, desc=f"Processing {subset} news", unit="news"):
        comments_save_dir = f'/home/server/Dai/IRI/data/reaction_occu/{dataset}/{subset}'
        os.makedirs(comments_save_dir, exist_ok=True)
        save_path = os.path.join(comments_save_dir, f"{news_id}")
        
        # 如果檔案已經存在，則跳過此news_id的處理
        if (os.path.exists(save_path)):
            print(f"File {save_path} already exists. Skipping...")
            continue
            
        news_content = get_content(news_id, dataset_name=dataset, news_type=subset)
        news_content = construct_news_length(news_content)
        
        try:
            out = generate(news_content)
            json.dump(out, open(save_path, 'w', encoding='utf-8'))  # save the network
        except AssertionError as e:
            print(f"AssertionError for news_id {news_id}: {str(e)}. Skipping this news article.")

    

def main():

    

    dataset_name = "gossipcop"
    news_ids = get_news_ids(dataset_name)

    process_news_comments(news_ids["real"], dataset=dataset_name, subset='real')
    process_news_comments(news_ids["fake"], dataset=dataset_name, subset='fake')

    dataset_name = "politifact"
    news_ids = get_news_ids(dataset_name)
   
    process_news_comments(news_ids["fake"], dataset=dataset_name, subset='fake')
    process_news_comments(news_ids["real"], dataset=dataset_name, subset='real')

if __name__ == "__main__":
    main()
