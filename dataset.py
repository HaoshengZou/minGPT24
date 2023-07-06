import os
import pickle
import torch
from torch.utils.data import Dataset

class DatasetOf24Game(Dataset):
    # initialize class vars 
    one_data_sample = "[4, 8, 9, 3]: 8 + 4 = 12, 9 + 3 = 12, 12 + 12 = 24  "
    one_problem_sample = "[4, 8, 9, 3]: "
    one_result_sample = "8 + 4 = 12, 9 + 3 = 12, 12 + 12 = 24  "

    all_chars = "0123456789[,]:+-*/= "
    itoc = {i:c for i, c in enumerate(all_chars)}
    ctoi = {c:i for i, c in enumerate(all_chars)}
    all_data_set = set()
    all_test_data_set = set()
    all_train_data = []
    all_test_data = []
    all_data = []

    # train test split
    with open('data/dataset1_9.pkl', 'rb') as f:
        res = pickle.load(f)
    my_train_xys, my_test_xys = res['train'], res['test']
    my_test_digits = set(x for x, _ in my_test_xys)

    with open(os.path.dirname(os.path.realpath(__file__)) + "/data/24_game_all_data.txt", "r") as f:
        for line in f:
            line = line.rstrip('\n')
            all_data_set.add(line)
            all_data.append(line)

            digits = sorted(line[1: 11].split(', '))
            digits_str = ' '.join(str(d) for d in digits)
            if digits_str in my_test_digits:
                all_test_data.append(line)
                all_test_data_set.add(line)
            else:
                all_train_data.append(line)

    def __init__(self, split, return_tokenized=True):
        self.return_tokenized = return_tokenized

        self.split = split # train/test        
        self.ixes = []
        if split == 'train':
            self.ixes = DatasetOf24Game.all_train_data
        elif split == 'test':
            self.ixes = DatasetOf24Game.all_test_data
        elif split == 'all':
            self.ixes = DatasetOf24Game.all_data
        else: 
            raise Exception("'split' must be 'all', 'train' or 'test'!") 

    @staticmethod
    def get_vocab_size():
        return len(DatasetOf24Game.all_chars)

    @staticmethod
    def get_block_size():
        # return len of an example
        return len(DatasetOf24Game.one_data_sample)

    def __len__(self):
        return len(self.ixes)

    def __getitem__(self, idx):
        # a data sample: [4, 8, 9, 3]: 8 + 4 = 12, 9 + 3 = 12, 12 + 12 = 24  
        s = self.ixes[idx]
        if self.return_tokenized:
            dix = [DatasetOf24Game.ctoi[c] for c in s] # convert each character to its token index
            # x will be input to GPT and y will be the associated expected outputs
            x = torch.tensor(dix[:-1], dtype=torch.long)
            y = torch.tensor(dix[1:], dtype=torch.long) # predict the next token in the sequence
            y[:len(DatasetOf24Game.one_problem_sample)] = -1 # we will only train in the output locations. -1 will mask loss to zero
            return x, y
        else:
            return s


if __name__ == '__main__':
    dataset = DatasetOf24Game('all')
    exit()
    # print([dataset.ctoi[c] for c in '[4, 8, 9, 3]: 8 + 4 = 12, 9 + 3 = 12, 12 + 12 = 24  '])
    # print('[5, 5, 5, 5]: 5 + 5 = 10, 5 + 10 = 15, 15 + 9 = 24  ' in dataset.all_data_set)
    dataset = DatasetOf24Game('train', return_tokenized=False)
    print(type(dataset[515]))
    from torch.utils.data.dataloader import DataLoader
    loader = DataLoader(dataset, batch_size=10)
    for mini_batch in loader:
        print(mini_batch)
        print(type(mini_batch))
        break
