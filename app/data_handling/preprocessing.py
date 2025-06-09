import spacy
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])


def lemmatize_line(line: str):
    text = ""
    for token in nlp(line):
        text += token.lemma_.lower() + " "
    return text.strip()
