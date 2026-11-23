from corpus import tokenizer
from corpus import SPECIAL_TOKENS
from attention import BigramLM
import numpy as np
import torch


def render_conversation(conv:dict, tokenizer:tokenizer) -> tuple[list[int], list[int]]:

    messages: list[dict] = conv["messages"]
    mask = []
    encoded = []

    for message in messages:
        role = message['role']
        content = message['content']

        if role == "user":
            encoded.append(tokenizer.encode_special(SPECIAL_TOKENS[0]))
            mask.append(0)
        elif role == "assistant":
            encoded.append(tokenizer.encode_special(SPECIAL_TOKENS[2]))
            mask.append(0)

        encoded.extend(tokenizer.encode(content))
        if role == "assistant":
            mask.extend([1] * len(content))
        else:
            mask.extend([0] * len(content))

        if role == "user":
            encoded.append(tokenizer.encode_special(SPECIAL_TOKENS[1]))
            mask.append(0)
        elif role == "assistant":
            encoded.append(tokenizer.encode_special(SPECIAL_TOKENS[3]))
            mask.append(1)

    return (encoded, mask)

def make_xy(ids, mask) -> tuple[list[int], list[int]]:

    x = np.array(ids[:-1])

    mask = np.array(mask[1:])
    y = np.where(mask == 1, ids[1:], -100)

    return x.tolist(), y.tolist()


def show_mask(ids, mask, tokenizer) -> None:

    out = ""
    for char, unit_mask in zip(ids, mask):
        if unit_mask == 1:
            out += f"[{tokenizer.decode([char])}]"
        else:
            out += tokenizer.decode([char])

    print(out)


def load_grow(path) -> tuple[BigramLM, tokenizer]:

    checkpoint = torch.load(path, map_location='cpu')
    tok = tokenizer(chars=checkpoint["chars"])
    config = checkpoint['config']
    config['vocab_size'] = tok.vocab_size
    old = checkpoint['model']
    model = BigramLM(**config)
    new = model.state_dict()

    for k in old:
        if old[k].shape ==  new[k].shape:
            new[k] = old[k]

        else:
            n = old[k].shape[0]
            new[k][:n] = old[k]
            new[k][n:] = old[k].mean(0)

    model.load_state_dict(new)
    return (model,tok)












    


        

        
