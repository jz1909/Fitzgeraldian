
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



    
