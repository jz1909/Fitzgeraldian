from corpus import tokenizer
from corpus import SPECIAL_TOKENS

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


        

        
