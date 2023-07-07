# 训练GPT算24点

## abstract

目前仅考虑了1~9的数字构成的24点问题，采用SL+RL的框架，训练一个随机初始化的10M参数GPT模型算24点。

问题规模比1~13的完整24点问题小4倍，输入的4个数字考虑了所有排列，SL模型在测试集上最高有70%的正确率，接着RL fine-tune有一定的提升，正确率最高到78%。

![res1](img/result1.png)
