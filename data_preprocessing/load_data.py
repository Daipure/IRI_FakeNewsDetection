import sys
import os
import json
import nltk
from tqdm import tqdm

sys.path.append('/home/server/Dai/IRI')
sys.path.append('/home/server/Dai/IRI/model')
from utils.get_llm_respond import *

DATA_PATH = {
    'gossipcop_fake_dir': '/home/server/Dai/IRI/data/gossipcop/gos_fake_news/',
    'gossipcop_real_dir': '/home/server/Dai/IRI/data/gossipcop/gos_real_news/',
    'politifact_fake_dir': '/home/server/Dai/IRI/data/politifact/pol_fake_news/',
    'politifact_real_dir': '/home/server/Dai/IRI/data/politifact/pol_real_news/',
}

def get_directories(dataset_name):
    if dataset_name == 'politifact':
        true_directory = DATA_PATH['politifact_real_dir']
        fake_directory = DATA_PATH['politifact_fake_dir']
    elif dataset_name == 'gossipcop':
        true_directory = DATA_PATH['gossipcop_real_dir']
        fake_directory = DATA_PATH['gossipcop_fake_dir']
    else:
        print("[ERROR] Wrong dataset parameter specified.")
        sys.exit(0)
    return true_directory, fake_directory

def construct_length(text, length=7000):
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

def get_news_ids(dataset_name):
    true_directory, fake_directory = get_directories(dataset_name)
    news_ids = {'real': [], 'fake': []}

    for file_name in os.listdir(true_directory):
        if file_name.endswith('.json'):
            news_ids['real'].append(file_name)
    
    for file_name in os.listdir(fake_directory):
        if file_name.endswith('.json'):
            news_ids['fake'].append(file_name)
    
    return news_ids
    

def get_content(news_id, dataset_name, news_type): 
    true_directory, fake_directory = get_directories(dataset_name)
    directory = true_directory if news_type == 'real' else fake_directory

    file_path = os.path.join(directory, news_id)
    try:
        with open(file_path, 'r') as file:
            content = json.load(file)
            n_content = content.get('text', '[ERROR] Text field not found in the file')
    except FileNotFoundError:
        print(f"[ERROR] File {news_id} not found in directory {directory}")
        return None
    except json.JSONDecodeError:
        print(f"[ERROR] Failed to decode JSON from file {news_id}")
        return None
    return n_content

