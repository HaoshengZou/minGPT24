# imports
import torch
import os
from model import GPT
from dataset import DatasetOf24Game
from utils import ConfigNode, set_seed
from reward import reward_v1
from trl import PPOTrainer, PPOConfig, create_reference_model
from trl.core import respond_to_batch
from tokenizer import get_TokenizerV0
from torch.utils.data.dataloader import DataLoader
from trainer import Trainer
from eval import eval


# HParams
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 256
PATH_TO_LOGS = 'logs/v0'
sft_model_path = '/out/data1_9_v2_vf/model.pt'
rl_dir = 'out/rl_17.40/'
os.makedirs(rl_dir, exist_ok=True)
set_seed(3407)

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
    os.path.realpath(__file__)) + sft_model_path, map_location=torch.device(device)))
print(f'loaded sft model from {sft_model_path}')

model_ref = create_reference_model(model)
model_ref.to(device)

# 2. tokenizer
tokenizer = get_TokenizerV0(for_trl=True)
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
train_dataset = DatasetOf24Game(split='train', return_tokenized=False)
train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, batch_size=ppo_config.batch_size)

# 5. training loop
epoch = 0
model.eval()
max_eval_acc = eval(model, DatasetOf24Game.all_test_mapping, tokenizer, device)  # 初始分数
model.train()
print(f"Epoch 0: init eval acc {max_eval_acc}")
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
        rewards = [reward_v1(q, r, tokenizer).to(device) for q, r in zip(queries, responses)]

        train_stats = ppo_trainer.step(queries, responses, rewards)
        batch4log = {'query': query_tensors, 'response': response_tensors}
        ppo_trainer.log_stats(train_stats, batch4log, rewards)

        epoch += 1
        print(f"Epoch {epoch}: ppo/mean_scores {train_stats['ppo/mean_scores']:.2f}, ppo/returns/mean {train_stats['ppo/returns/mean']:.2f}")

        # eval on test
        model.eval()
        eval_acc = eval(model, DatasetOf24Game.all_test_mapping, tokenizer, device)
        model.train()
        if eval_acc > max_eval_acc:
            max_eval_acc = eval_acc
            ckpt_path = os.path.join(rl_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f'saved new model to {ckpt_path}')
        print(f"Epoch {epoch}: eval acc {eval_acc}, max_eval_acc {max_eval_acc}")
