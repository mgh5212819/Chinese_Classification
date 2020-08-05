'''
@Author: xiaoyao jiang
@Date: 2020-04-09 16:21:08
@LastEditTime: 2020-07-06 20:25:23
@LastEditors: xiaoyao jiang
@Description: data set
@FilePath: /bookClassification(ToDo)/src/data/dataset.py
'''
import pandas as pd
import torch
import json
from torch.utils.data import Dataset
from __init__ import *
from src.utils import config


class MyDataset(Dataset):
    def __init__(self,
                 path,
                 dictionary=None,
                 max_length=128,
                 tokenizer=None,
                 word=False):
        super(MyDataset, self).__init__()
        self.data = pd.read_csv(path, sep='\t').dropna()
        with open(config.root_path + '/data/label2id.json', 'r') as f:
            self.label2id = json.load(f)
        self.data['category_id'] = self.data['label'].map(self.label2id)
        if not word:
            self.data['text'] = self.data['text'].apply(
                lambda x: " ".join("".join(x.split())))
        if tokenizer is not None:
            self.model_name = 'bert'
            self.tokenizer = tokenizer
#             self.data['text'] = self.data['text'].apply(lambda x: "".join(x.split()))
        else:
            self.model_name = 'normal'
            self.tokenizer = dictionary
        self.max_length = max_length

    def __getitem__(self, i):
        data = self.data.iloc[i]
        text = data['text']
        labels = int(data['category_id'])
        attention_mask, token_type_ids = [0], [0]
        if 'bert' in self.model_name:
            text_dict = self.tokenizer.encode_plus(
                text,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=self.max_length,  # Pad & truncate all sentences.
                ad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                #                                                    return_tensors='pt',     # Return pytorch tensors.
            )
            input_ids, attention_mask, token_type_ids = text_dict[
                'input_ids'], text_dict['attention_mask'], text_dict[
                    'token_type_ids']
        else:
            text = text.split()
            #             print(text)
            text = text + [0] * max(0, self.max_length - len(text)) if len(
                text) < self.max_length else text[:self.max_length]
            input_ids = [self.tokenizer.indexer(x) for x in text]


#             print(input_ids)
        output = {
            "token_ids": input_ids,
            'attention_mask': attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels
        }
        return output

    def __len__(self):
        return self.data.shape[0]


def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """
    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        注意 token type id 右侧pad是添加1而不是0，1表示属于句子B
        """
        ### TODO 
        # 1. 根据max_length 加 padding
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]
    attention_mask = [data["attention_mask"] for data in batch]
    labels = torch.tensor([data["labels"] for data in batch])
    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    attention_mask_padded = padding(attention_mask, max_length)
    #     print(token_ids_padded)
    return token_ids_padded, attention_mask_padded, token_type_ids_padded, labels