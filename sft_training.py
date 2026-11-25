import json
import torch
import math
import random
from attention import BigramLM
from sft_utils import render_conversation, make_xy
from corpus import tokenizer

#Config


def load_jsonl(path:str) -> list[dict]:

    result = []

    with open(path, 'r', encoding="utf-8") as file:
        for line in file:
            if line.strip():
                curr_dict = json.loads(line)
                result.append(curr_dict)

    return result

def get_sft_batch(convs:list[dict], tok:tokenizer, device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Takes a batch of convos and turns them X, Y pairs of B x T stacks    
    """

    xs = []
    ys = []

    for conv in convs:
        tokenized_msg, mask= render_conversation(conv, tok)
        x, y =  make_xy(tokenized_msg, mask)
        xs.append(x)
        ys.append(y)

    #padding since size t isn't guaranteed
    max_t = max(len(x) for x in xs)
    for x,y in zip(xs, ys):
        x.extend([0] * max(0, max_t - len(x)))
        y.extend([-100] * max(0, max_t - len(y)))

    xs = torch.tensor(xs, dtype=torch.long, device=device)
    ys = torch.tensor(ys, dtype=torch.long, device=device)

    return (xs, ys)

def get_train_batch(step, sft_iter, replay_steps,corpus, tok, block_size,batch_size, device) -> tuple[torch.tensor, torch.tensor, str]:

    """
    Retrieve sft batch, but occasionally mix in replay from pretraining dataset to prevent forgetting
    """

    if replay_steps > 0 and step > 0 and step % replay_steps == 0:
        x, y = corpus.get_batch('train', block_size, batch_size)

        x = x.to(device)
        y = y.to(device)
        kind = 'replay'

    else: 
        convs = next(sft_iter)
        x,y = get_sft_batch(convs, tok, device)
        kind = 'sft'

    return x, y, kind

def get_lr(steps, total_steps, peak_lr, min_lr, warmup_steps) -> float:

    """
    lr is especially touchy for sft, so we tune it carefully here
    """

    if steps < warmup_steps:
        lr = peak_lr * (steps + 1) / warmup_steps

    else:
        progress = (steps - warmup_steps) / (total_steps - warmup_steps)

        lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))

    return lr

def sft_batches(convs:list[dict], batch_size:int):

    while True:
        shuffled = random.sample(convs, len(convs))
        for i in range(0, len(shuffled), batch_size):
            yield shuffled[i:i+batch_size]

def train(model, tok, corpus, train_convs, val_convs, device, *, replay_block_size=256, batch_size=32, epochs=3, replay_steps=10,
          peak_lr=2e-4, min_lr=2e-5, warmup_steps=25,
          log_interval=10, eval_interval=50) -> BigramLM:

    sft_iter = sft_batches(train_convs, batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.1)

    replay_loss = []
    sft_loss = []

    steps_per_epoch = math.ceil(len(train_convs)/batch_size)
    total_sft_steps = epochs * steps_per_epoch
    total_steps = total_sft_steps if replay_steps == 0 else math.ceil(total_sft_steps / (1-1/replay_steps))

    model.train()
    for step in range(total_steps):
        lr = get_lr(step, total_steps, peak_lr, min_lr, warmup_steps)

        for g in optimizer.param_groups:
            g['lr'] = lr

        xb, yb, kind = get_train_batch(step, sft_iter, replay_steps, corpus, tok, replay_block_size, batch_size, device)

        _, loss = model(xb,yb)
        model.zero_grad(set_to_none = True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if kind == 'replay':
            replay_loss.append(loss.item())
        else:
            sft_loss.append(loss.item())

        if step % log_interval == 0 or step == total_steps - 1:
            line = f"step {step}/{total_steps} | lr {lr:.2e} | gnorm {norm.item():.2f}"
            if sft_loss:
                line += f" | sft {sum(sft_loss)/len(sft_loss):.4f}"
            if replay_loss:
                line += f" | replay {sum(replay_loss)/len(replay_loss):.4f}"
            print(line)
            sft_loss, replay_loss = [], []

        # TODO: Mix in testing on validation dataset and keep gradients for best validation performance


    return model



