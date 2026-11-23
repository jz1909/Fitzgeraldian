import json
import random
import re
from pathlib import Path

PLAY_RE =  re.compile(r"""
    ([A-Z][A-Z .']{1,29})      # speaker - a capital letter, then 1-29 more caps/space/./'
    (?:\ (\[[^\]]*\]))?        # optional " [stage direction]"
    [.:]\                      # a period or colon and a space 
    (?=.*[a-z])                # the speech contains at least one lowercase letter
    (.+)                       # the speech itself
""", re.VERBOSE | re.DOTALL)



def load_paragraphs(path:str) -> list[tuple[int, str]]:
    """
    Dissect file into list of paragraphs matched with its offset, where offset is the place in the file where the paragraph starts 
    """

    with open(path, encoding='utf-8') as f:
        text = f.read()

    result = []

    offset_count = 0
    paragraph_list = text.split("\n\n")

    for paragraph in paragraph_list:
        paragraph_length = len(paragraph)

        paragraph = paragraph.strip()

        if len(paragraph) > 0:
            result.append((offset_count, paragraph))
        
        #+2 to account for \n
        offset_count += paragraph_length + 2

    return result


def classify(text:str)->str:
    """
    Looks at a paragraph and label it with play, quote, or other
    """

    message = PLAY_RE.match(text)

    if message:
        return "play"

    if text.startswith('"'):
        return "quote"

    return "other"

def find_runs(paragraphs: list[tuple[int, str]])-> list[list[tuple[int, str, str]]]:
    """
    group consecutive diaglogue paragraphs into runs, which is one stretch of conversation
    """

    total_runs = []
    curr_convo = []

    for block in paragraphs:
        label = block[0]
        paragraph = block[1]

        category = classify(paragraph)

        if category == "other":
            if len(curr_convo) > 0:
                total_runs.append(curr_convo)
            curr_convo = []
        else:
            if curr_convo and category != curr_convo[-1][2]:
                total_runs.append(curr_convo)
                curr_convo = []

            curr_convo.append((label, paragraph, category))

    if curr_convo:
        total_runs.append(curr_convo)

    return total_runs

def clean_play_line(text:str) -> tuple[str, str] | None:
    """
    Turn a line of play dialogue into a tuple of (speaker, speech)
    """

    DIRECTION_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")

    dialogue = PLAY_RE.match(text)   
    if dialogue is None:
        return None

    speaker, direction, speech = dialogue.groups()
    speech = DIRECTION_RE.sub(" ", speech)

    speech = re.sub(r"\s+", " ", speech).strip()

    if not speech:
        return None

    return (speaker.strip(), speech)

def extract_quote(text: str) -> tuple[str, bool] | None:
    """
    Keep only the spoken words of a quote paragraph. continues is True when the last quote is left open
    """

    pieces = [p.strip() for p in re.findall(r'"([^"]*)(?:"|$)', text)]
    pieces = [p for p in pieces if p]
    if not pieces:
        return None

    speech = pieces[0]
    for piece in pieces[1:]:
        if speech.endswith(",") and piece[0].isupper():
            speech = speech[:-1] + "."
        speech += " " + piece

    if speech.endswith(","):
        speech = speech[:-1] + "."

    return re.sub(r"\s+", " ", speech), text.count('"') % 2 == 1


def extract_turns(run: list[tuple[int, str, str]]) -> list[dict]:
    """
    Turn a run into speaker turns, merging consecutive paragraphs from the same speaker
    """

    turns = []
    kind = run[0][2]
    speaker, continues = "B", False

    for offset, text, _ in run:
        if kind == "play":
            result = clean_play_line(text)
            if result is None:
                continue
            who, speech = result
        else:
            result = extract_quote(text)
            if result is None:
                continue
            speech, open_quote = result
            # novels don't name speakers, so assume they alternate unless the quote carried over
            who = speaker if continues else ("A" if speaker == "B" else "B")
            speaker, continues = who, open_quote

        if turns and turns[-1]["speaker"] == who:
            turns[-1]["content"] += " " + speech
        else:
            turns.append({"speaker": who, "content": speech, "offset": offset})

    return turns


def rendered_len(turns: list[dict]) -> int:
    """
    Length in tokens once special tokens are added: <|bos|> plus start/end tokens per turn
    """

    return 1 + sum(len(t["content"]) + 2 for t in turns)


def chunk(turns: list[dict], max_len: int = 256) -> list[list[dict]]:
    """
    Cut turns into pieces that fit max_len, each with an even number of turns (user first, assistant last)
    """

    chunks = []
    curr = []

    for turn in turns:
        if rendered_len(curr + [turn]) <= max_len:
            curr.append(turn)
            continue

        # an odd leftover turn would be a dangling user turn, so it starts the next chunk instead
        carry = [curr.pop()] if len(curr) % 2 else []
        if len(curr) >= 2:
            chunks.append(curr)

        curr = carry + [turn]
        if rendered_len(curr) > max_len:
            curr = [turn] if rendered_len([turn]) <= max_len else []

    if len(curr) % 2:
        curr.pop()
    if len(curr) >= 2:
        chunks.append(curr)

    return chunks


def assign_roles(turns: list[dict]) -> list[dict]:
    """
    First turn is the user, then alternate
    """

    return [{"role": "user" if i % 2 == 0 else "assistant", "content": t["content"]} for i, t in enumerate(turns)]


def split_train_val(convs: list[dict], cutoff: int) -> tuple[list[dict], list[dict]]:
    """
    Same split point as pretraining; conversations straddling the cutoff are dropped
    """

    train = [c for c in convs if c["end"] < cutoff]
    val = [c for c in convs if c["start"] >= cutoff]
    return train, val


def write_jsonl(convs: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for c in convs:
            f.write(json.dumps({"messages": c["messages"], "source": c["source"]}, ensure_ascii=False) + "\n")


def print_stats(train: list[dict], val: list[dict]):
    for name, convs in [("train", train), ("val", val)]:
        lengths = sorted(rendered_len(c["messages"]) for c in convs)
        turns = sum(len(c["messages"]) for c in convs) / len(convs)
        sources = {s: sum(c["source"] == s for c in convs) for s in ("play", "quote")}
        print(f"{name}: {len(convs)} conversations {sources}, {turns:.1f} turns avg, "
              f"length median {lengths[len(lengths) // 2]} / p90 {lengths[int(0.9 * len(lengths))]} / max {lengths[-1]}")

    for c in random.sample(train, 3):
        print(f"\n--- {c['source']} ---")
        for m in c["messages"]:
            print(f"{m['role']:>9}: {m['content']}")


def main():
    data_dir = Path(__file__).parent / "fitzgerald_data"
    path = data_dir / "input.txt"

    paragraphs = load_paragraphs(path)
    cutoff = int(0.9 * len(path.read_text(encoding="utf-8")))

    convs = []
    for run in find_runs(paragraphs):
        for c in chunk(extract_turns(run), ):
            convs.append({"source": run[0][2], "start": c[0]["offset"], "end": c[-1]["offset"],
                          "messages": assign_roles(c)})

    train, val = split_train_val(convs, cutoff)
    write_jsonl(train, data_dir / "sft_train.jsonl")
    write_jsonl(val, data_dir / "sft_val.jsonl")
    print_stats(train, val)


if __name__ == "__main__":
    main()
