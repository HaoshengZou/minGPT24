import torch
from dataset import DatasetOf24Game
from collections import Counter


def reward_v0(query, response, tokenizer):
    query_txt = tokenizer.decode(query)
    response_txt = tokenizer.decode(response)
    full_txt = query_txt + response_txt

    reward = 0
    all_correct = full_txt in DatasetOf24Game.all_data_set
    reward += 1 if all_correct else -1

    return torch.tensor(reward, dtype=torch.float)


def reward_v1(query, response, tokenizer, already_txt=False):
    # easy: query中的4个数字是否都用且仅用1次，中间计算过程先不管. -1 到 0.6, 每用1个加0.4
    if not already_txt:
        query_txt = tokenizer.decode(query)
        response_txt = tokenizer.decode(response)
    else:
        query_txt, response_txt = query, response
    full_txt = query_txt + response_txt

    reward = 0
    all_correct = full_txt in DatasetOf24Game.all_data_set
    if all_correct:
        reward += 1
    else:
        reward -= 1
        digits_cnt = Counter(query_txt[1: 11].split(', '))
        digits_cnt0 = Counter(query_txt[1: 11].split(', '))  #

        equations = response_txt.split(', ')
        if len(equations) != 3:
            reward -= 1
        for e in equations:
            if '=' not in e:
                reward -= 0.2
            else:
                expression = e.split('=')[0]
                if not any(cal in expression for cal in '+-*/'):
                    reward -= 0.2
                else:
                    for cal in '+-*/':
                        if cal in expression:
                            break
                    used_digits = expression.split(cal)
                    if len(used_digits) != 2:
                        reward -= 0.2
                    else:
                        for d in used_digits:
                            if d.strip() in digits_cnt:
                                digits_cnt[d.strip()] -= 1
        for d, v in digits_cnt.items():
            if v == 0:
                reward += 0.4 * digits_cnt0[d]

    return torch.tensor(reward, dtype=torch.float)


if __name__ == '__main__':
    query_txt = '[4, 8, 9, 3]: '
    response_txt = '8 + 5 = 12, 9 + 7 = 12, 12 + 12 = 24  '
    print(reward_v1(query_txt, response_txt, None, True))
