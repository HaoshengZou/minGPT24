import os
import sys
import json

import torch
from torch.utils.data.dataloader import DataLoader

from model import GPT
from trainer import Trainer
from utils import set_seed, ConfigNode
from dataset import DatasetOf24Game
from tokenizer import get_TokenizerV0
from trl.core import respond_to_batch
from itertools import permutations


# -----------------------------------------------------------------------------

def get_config():
    C = ConfigNode()

    # system
    C.system = ConfigNode()
    C.system.seed = 515  # 3407

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
    C.trainer.batch_size = 512
    C.trainer.max_iters = int(1e6)

    return C

def eval(model, dataset, tokenizer, device, sample=True):
    results = list()
    for x, solutions in dataset.items():
        # prepare input
        x_digits = x.split(' ')
        perms = permutations(x_digits)
        can_solve = False
        inputs = list()
        for perm in perms:
            inp = '[' + ', '.join(str(digit) for digit in perm) + ']: '
            # print('inp:', inp)
            inputs.append(inp)

        query_tensors = tokenizer(inputs, return_tensors="pt").to(device)['input_ids']  # P4 x len
        if sample:
            response_tensors = respond_to_batch(model, query_tensors,
                                                txt_len=len(DatasetOf24Game.one_result_sample))
        else:
            response_tensors = model.generate(query_tensors, len(DatasetOf24Game.one_result_sample), do_sample=False)
            response_tensors = response_tensors[:, -len(DatasetOf24Game.one_result_sample):]
        respones = [tokenizer.decode(response_tensors[i]) for i in range(len(response_tensors))]
        # print('out:', response)
        if any(r in solutions for r in respones):
            can_solve = True

        results.append(int(can_solve))
        # print(f'{x_digits} solved? {can_solve}, {response if can_solve else None}')
        # print(f'{x_digits} solved? {can_solve}')

    return sum(results) / len(results)


if __name__ == '__main__':
    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    
    # see random seeds for everywhere
    set_seed(config.system.seed)

    # construct train and test datasets
    train_dataset = DatasetOf24Game(split='train')
    test_dataset  = DatasetOf24Game(split='test')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # construct the model
    config.model.vocab_size = DatasetOf24Game.get_vocab_size()
    config.model.block_size = DatasetOf24Game.get_block_size()
    print(config)
    model = GPT(config.model)
    # print(model)
    model.to(device)
    model.eval()
    model_path = '/out/data1_9_v2_vf/model.pt'
    model.load_state_dict(torch.load(os.path.dirname(os.path.realpath(__file__)) + model_path, map_location=torch.device(device)), strict=False)
    print(f'loaded model from {model_path}')

    tokenizer = get_TokenizerV0(for_trl=True)

    dataset = DatasetOf24Game.all_test_mapping

    model.train()
    print('test solve ratio:', eval(model, dataset, tokenizer, device, sample=True))
