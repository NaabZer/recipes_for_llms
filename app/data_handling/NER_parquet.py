import spacy
import duckdb
import polars as pl
from pydantic import BaseModel
from app.data_handling.preprocessing import nlp


class ParquetDefinition(BaseModel):
    name: str
    data: list
    is_map: bool = False


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


def transform_ingredients_to_tokens(ingredients: list,
                                    ner_model: spacy.lang.en.English):
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
    datapoint_obj = {
        'foods': foods,
        'preps': preparations,
        'optionals': optionals
        }
    return datapoint_obj


# Assume recipeNLG dataset, where each ingredient line is a list item
def transform_data_to_tokens(data: list[str],
                             ner_model: spacy.lang.en.English):
    tokens = []
    preps = []
    optionals = []
    for ingredients in data:
        datapoint_obj = transform_ingredients_to_tokens(ingredients, ner_model)
        tokens.append(datapoint_obj['foods'])
        prep_dict = datapoint_obj['preps']
        prep = []
        for k, v in prep_dict.items():
            prep.append({'key': k, 'value': v})
        preps.append(prep)
        optionals.append(datapoint_obj['optionals'])
    return tokens, preps, optionals,


def create_parquet_file(parquet_path: str, df: pl.DataFrame,
                        fields: list[ParquetDefinition]):
    cols_to_add = [pl.Series(name=pq.name, values=pq.data) for pq in fields]
    new_df = df.with_columns(*cols_to_add)
    new_df.write_parquet(parquet_path)

    # Turn the map structs into actual maps
    duckdb.sql(f"""COPY (
                    SELECT
                        * EXCLUDE (preps), map_from_entries(preps) AS preps
                    FROM '{parquet_path}'
               ) TO '{parquet_path}' (FORMAT PARQUET, OVERWRITE TRUE)""")
    return parquet_path


def construct_ingredient_query(pq_path: str, ingredients: list, preps: dict):
    base_sql = f""" SELECT * FROM '{pq_path}' AS tbl
                    WHERE list_has_all(tbl.tokens, {ingredients})
                """
    prep_filter = ""
    for i, (key, value) in enumerate(preps.items()):
        prep_filter += f"""AND map_contains(preps, '{key}')
                           AND list_has_all(preps['{key}'], {value})
                        """
    sql = base_sql + prep_filter
    return sql
