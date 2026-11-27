from attention import BigramLM
from corpus import tokenizer
from sft_utils import render_conversation
import torch
import torch.nn.functional as F

def load_sft(path, device) -> tuple[BigramLM, tokenizer, int]:


    checkpoint = torch.load(path, map_location=device)
    model = BigramLM(**checkpoint["config"])
    tok = tokenizer(chars=checkpoint["chars"])
    block_size = checkpoint["config"]["block_size"]
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    return model, tok, block_size

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

def build_prompt(user_msg: str, tok:tokenizer, max_new: int, block_size:int) -> list[int]:

    ids, _ = render_conversation({"messages":[{"role":"user", "content": user_msg}]}, tok)

    ids.append(tok.encode_special("<|assistant_start|>"))

    budget =  block_size - max_new

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

def chat(model, tok, max_new, temperature, block_size) -> None:

    while True:
        user_message = input("you: ")
        msg = clean(user_message, tok)
        if not msg:
            print("Input is invalid")
            continue

        try:
            ids = build_prompt(msg, tok, max_new, block_size)
        except ValueError:
            print("message was too long")
            continue

        reply, reason = generate_reply(model, tok, ids, max_new, temperature)

        print(f"fitz: {reply if reply else 'no response'}")

        if reason == "max_new":
            print("[cut off]")
        elif reason == "stray":
            print("EOF stray")

if __name__ == "__main__":

    #Configs
    max_new = 128   
    temperature = 0.8 

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tok, block_size = load_sft("fitzgerald_sft.pth", device)

    chat(model=model, tok=tok, block_size=block_size, max_new=max_new, temperature=temperature)






