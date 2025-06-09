import spacy
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
vocab = set()


def process_ingredient(ingredient_doc: spacy.tokens.doc.Doc,
                       include_variety: bool = False):
    food_lemmas = []
    prep_lemmas = [[]]
    optional = False
    if include_variety:
        var_lemmas = [[]]
    for t in ingredient_doc.ents:
        if t.label_ == 'Food':
            lemmas = [w.lemma_.lower() for w in nlp(str(t))]
            food_lemmas.append(lemmas)
            if len(prep_lemmas[-1]) > 0:
                # Add a new list of preparation that corresponds to a
                # new food item on the same line
                prep_lemmas.append([])
            if include_variety and len(var_lemmas[-1]) > 0:
                var_lemmas.append([])
        elif t.label_ == 'Preparation':
            lemmas = [w.lemma_.lower() for w in nlp(str(t))]
            prep_lemmas[-1] = prep_lemmas[-1] + lemmas
        elif include_variety and t.label_ == 'Variety':
            lemmas = [w.lemma_.lower() for w in nlp(str(t))]
            var_lemmas[-1] = var_lemmas[-1] + lemmas
        elif t.label_ == "Optional":
            optional = True
    ret_obj = {'food': food_lemmas,
               'prep': prep_lemmas,
               'optional': optional}
    if include_variety:
        ret_obj['var': var_lemmas]
    return ret_obj


def transform_ingredients_to_tokens(ingredients: list, ner_model,
                                    create_vocab=False):
    ner_lines = [ner_model(ingredient_line) for ingredient_line in ingredients]
    foods = []
    preparations = {}
    optionals = []
    for line in ner_lines:
        processed_line_obj = process_ingredient(line)
        if len(processed_line_obj['food']) == 0:  # No food found, continue
            continue
        # Only take first food item in a line
        food = "_".join(processed_line_obj['food'][0])
        foods.append(food)
        # If there's only one food, all prep would belong to that food
        if len(processed_line_obj['food']) == 1:
            preparation_items = [item for sublist in processed_line_obj['prep']
                                 for item in sublist]
        # Otherwise, we assume only the first list of preparations
        # corresponds to the first food
        else:
            preparation_items = processed_line_obj['prep'][0]
        if len(preparation_items) > 0:
            preparations[food] = preparation_items
        optionals.append(processed_line_obj['optional'])
        if create_vocab:
            vocab.add(food)
    datapoint_obj = {
        'foods': foods,
        'preps': preparations,
        'optionals': optionals
        }
    return datapoint_obj


def lemmatize_line(line: str):
    text = ""
    for token in nlp(line):
        text += token.lemma_.lower() + " "
    return text.strip()
