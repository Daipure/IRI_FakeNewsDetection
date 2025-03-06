import sys
sys.path.append('/home/server/Dai/IRI')
sys.path.append('/home/server/Dai/IRI/model')
import os
import json
import nltk
from tqdm import tqdm

from data_preprocessing.load_data import *


from utils.get_llm_respond import *



def get_prompt(News_Content):
    Prompt = """
    Please perform the following actions:
    Step 1: Select the three most likely news creation intentions of the news creator from the options below, and provide them as a comma-separated list, ordered from most to least relevant, using the exact wording from the options:
    Intentions: Deliberate Misinformation, Selective Reporting, Emotional Manipulation, Incorrect Citation, Propaganda and Public Opinion Guidance, Political Attacks, Ideological Promotion, Policy Influence, Advertising Revenue and Economic Interests, IRIket Manipulation, Competitor Attacks, Social Impact Intentions, Social Division and Public Safety, Public Health, Environmental Protection, Knowledge Dissemination, Misleading Education, Neutral Education, Other Intentions if None of the Above Apply.

    Step 2: For each of the three selected intentions, provide a detailed explanation (2-3 sentences) of why you chose it. Include specific examples from the news article that support your choice, quoting relevant sentences or phrases.
    """
    Prompt += "News Content:\n```{}```\n".format(News_Content)
    return Prompt



def create_intention(news_id_dict, dataset, subset='fake'):
    save_dir = f'/home/server/Dai/IRI/data/news_intention/{dataset}/{subset}'
    os.makedirs(save_dir, exist_ok=True)

    for news_id in tqdm(news_id_dict):
        save_path = os.path.join(save_dir, news_id)
        if os.path.exists(save_path):
            print(f"news_id:{news_id} motivation已建構")
            continue

        news_content = get_content(news_id, dataset_name=dataset, news_type=subset)
        if news_content is None:
            continue

        news_content = construct_length(news_content)
        prompt = get_prompt(news_content)
        res = get_completion(prompt)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)
        print(f"news_id:{news_id} motivation建構成功")

def main():
    dataset_name = 'gossipcop'
    news_ids = get_news_ids(dataset_name)
    create_intention(news_ids["real"], dataset=dataset_name, subset='real')
    create_intention(news_ids["fake"], dataset=dataset_name, subset='fake')

if __name__ == "__main__":
    main()
 