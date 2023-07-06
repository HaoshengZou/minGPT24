import torch
from dataset import DatasetOf24Game


def reward_v0(query, response, tokenizer):
    query_txt = tokenizer.decode(query)
    response_txt = tokenizer.decode(response)
    full_txt = query_txt + response_txt

    reward = 0
    all_correct = full_txt in DatasetOf24Game.all_data_set
    reward += 1 if all_correct else -1

    return torch.tensor(reward, dtype=torch.float)


def reward_v1(query, response, tokenizer):
    # 数字使用限制，用mask保证？
    pass
