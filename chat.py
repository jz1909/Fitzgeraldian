from attention import BigramLM
from corpus import tokenizer
from sft_utils import render_conversation
import torch
import torch.nn.functional as F

def load_sft(path, device) -> tuple[BigramLM, tokenizer]:


    checkpoint = torch.load(path, map_location=device)
    model = BigramLM(**checkpoint["config"])
    tok = tokenizer(chars=checkpoint["chars"])
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    return model, tok

def clean(input:str, tok:tokenizer) -> str:

    replacements = {"“": '"', "”": '"', "‘": "'", "’": "'", "…": "...", "–": "—", "\t": " "}

    output = []

    for char in input:
        if char in replacements:
            output.append(replacements[char])
        elif char in tok.chars:
            output.append(char)
        

    output = "".join(output)
    output = " ".join(output.split())
    return output.strip()

def build_prompt(history: list[dict], user_msg: str, tok:tokenizer, max_new: int, block_size:int) -> list[int]:

    ids, _ = render_conversation({"messages":history+ [{"role":"user", "content": user_msg}]}, tok)

    ids.append(tok.encode_special("<|assistant_start|>"))

    budget =  block_size - max_new
    while(len(ids) > budget and history):
        history = history[2:]
        ids, _ = render_conversation({"messages":history+ [{"role":"user", "content": user_msg}]}, tok)

        ids.append(tok.encode_special("<|assistant_start|>"))

    if len(ids) > budget :
        raise ValueError("Message to long")

    return ids

@torch.no_grad()
def generate_reply(model, tok, ids, max_new, temperature=1.0) -> tuple[str, str] | None:
    model.eval()
    device = next(model.parameters()).device

    #Returns (1, T)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    end_id = tok.encode_special("<|assistant_end|>")

    reply = []

    for _ in range(max_new):
        logits, _ = model.forward(idx[:, -model.block_size:, ])
        #(1,B,C) -> (1,1,C)
        logits = logits[:,-1,:]
        logits = logits/temperature

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)

        if next_id.item() == end_id:
            return (tok.decode(reply), "end")
        if next_id.item() >= len(tok.chars):
            return (tok.decode(reply), "stray")

        
        reply.append(next_id.item())
        idx = torch.cat([idx, next_id], dim=1)

    return (tok.decode(reply), "max_new")

def chat(model, tok, max_new, temperature) -> None:








