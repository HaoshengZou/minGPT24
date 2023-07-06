# imports
import torch
import os
from model import GPT
from dataset import DatasetOf24Game
from utils import ConfigNode
from reward import reward_v0
from trl import PPOTrainer, PPOConfig, create_reference_model
from trl.core import respond_to_batch
from tokenizer import get_TokenizerV0
from torch.utils.data.dataloader import DataLoader
from trainer import Trainer


# HParams
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 256
PATH_TO_LOGS = 'logs/v0'

# 1. get models
# construct the model
config = ConfigNode()
config.model = GPT.get_default_config()
config.model.model_type = 'gpt-mini'
config.model.vocab_size = DatasetOf24Game.get_vocab_size()
config.model.block_size = DatasetOf24Game.get_block_size()
# print(config)
model = GPT(config.model, dummy_v=True)
model.to(device)
# print(model)
model.load_state_dict(torch.load(os.path.dirname(
    os.path.realpath(__file__)) + "/out/get24/4train-6test-6layer-6head-384emb-20000steps.model.pt", map_location=torch.device(device)))

model_ref = create_reference_model(model)
model_ref.to(device)

# 2. tokenizer
tokenizer = get_TokenizerV0(for_transformer=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 3. initialize trainer
train_config = Trainer.get_default_config()
train_config.learning_rate = 1e-5
opt = model.configure_optimizers(train_config)
ppo_config = PPOConfig(
    batch_size=BATCH_SIZE,
    mini_batch_size=BATCH_SIZE // 2,
    ppo_epochs=3,
    learning_rate=1e-5,
    log_with='tensorboard',
    project_kwargs={"logging_dir": PATH_TO_LOGS},
)
ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer, optimizer=opt)

# 4. dataset
train_dataset = DatasetOf24Game(split='all', return_tokenized=False)
train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, batch_size=ppo_config.batch_size)

# 5. training loop
epoch = 0
while True:
    for batch in train_loader:
        # get model response
        query_tensors = tokenizer(batch, return_tensors="pt").to(device)['input_ids'][:, :len(DatasetOf24Game.one_problem_sample)]
        response_tensors = respond_to_batch(model, query_tensors, txt_len=len(DatasetOf24Game.one_result_sample))

        # train model for one step with ppo
        queries = [query_tensors[i] for i in range(len(batch))]
        responses = [response_tensors[i] for i in range(len(batch))]

        # compute reward
        # reward = [torch.tensor(1.0).to(device)] * len(batch)
        rewards = [reward_v0(q, r, tokenizer).to(device) for q, r in zip(queries, responses)]

        train_stats = ppo_trainer.step(queries, responses, rewards)
        batch4log = {'query': query_tensors, 'response': response_tensors}
        ppo_trainer.log_stats(train_stats, batch4log, rewards)

        epoch += 1
        print(f"Epoch {epoch}: ppo/mean_scores {train_stats['ppo/mean_scores']:.2f}, ppo/returns/mean {train_stats['ppo/returns/mean']:.2f}")
