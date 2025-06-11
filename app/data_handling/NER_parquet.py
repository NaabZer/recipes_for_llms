import spacy
import duckdb
import polars as pl
import os
import os.path
from pydantic import BaseModel
from app.data_handling.preprocessing import nlp


exclude_list = [
        'salt',
        'pepper',
        'black_pepper',
        'water',
        'ice'
        ]


class ParquetDefinition(BaseModel):
    name: str
    data: list
    is_map: bool = False


def process_ingredient(ingredient_doc: spacy.tokens.doc.Doc,
                       include_variety: bool = False,
                       include_brand: bool = False
                       ):
    food_lemmas = []
    prep_lemmas = []
    optional = False
    if include_variety:
        var_lemmas = []
    if include_brand:
        brand_lemmas = []
    for t in ingredient_doc.ents:
        if t.label_ == 'Food':
            # lemmas = [w.lemma_.lower() for w in t if not w.is_punct]
            lemmas = [w.lemma_.lower() for w in nlp(str(t)) if not w.is_punct]
            food_lemmas.append(lemmas)
            # Add a new list of preparation that corresponds to a
            # new food item on the same line
            prep_lemmas.append([])
            if include_variety:
                var_lemmas.append([])
            if include_brand:
                brand_lemmas.append([])
        elif t.label_ == 'Preparation':
            # lemmas = [w.lemma_.lower() for w in t if not w.is_punct]
            lemmas = [w.lemma_.lower() for w in nlp(str(t)) if not w.is_punct]
            if len(prep_lemmas) == 0:
                prep_lemmas.append([])
            prep_lemmas[-1] = prep_lemmas[-1] + lemmas
        elif include_variety and t.label_ == 'Variety':
            # lemmas = [w.lemma_.lower() for w in t if not w.is_punct]
            lemmas = [w.lemma_.lower() for w in nlp(str(t)) if not w.is_punct]
            if len(var_lemmas) == 0:
                var_lemmas.append([])
            var_lemmas[-1] = var_lemmas[-1] + lemmas
        elif include_brand and t.label_ == 'Brand':
            # lemmas = [w.lemma_.lower() for w in t if not w.is_punct]
            lemmas = [w.lemma_.lower() for w in nlp(str(t)) if not w.is_punct]
            if len(brand_lemmas) == 0:
                brand_lemmas.append([])
            brand_lemmas[-1] = brand_lemmas[-1] + lemmas
        elif t.label_ == "Optional":
            optional = True
    ret_obj = {'food': food_lemmas,
               'prep': prep_lemmas,
               'optional': optional}
    if include_variety:
        ret_obj['var'] = var_lemmas
    if include_brand:
        ret_obj['brand'] = brand_lemmas
    return ret_obj


def split_recipe_obj_to_sentences(ingredient_doc: spacy.tokens.doc.Doc):
    last_whitespace = False
    for o in ingredient_doc[1:]:  # Skip first since it is sent_start
        if last_whitespace:
            o.is_sent_start = True
            last_whitespace = False
        else:
            o.is_sent_start = False

        if o.is_space:
            last_whitespace = True
    return [sent.as_doc() for sent in ingredient_doc.sents]


def transform_ingredients_to_tokens(
        ingredients: list[str] | str,
        ner_model: spacy.lang.en.English,
        include_variety: bool = False,
        include_brand: bool = False,
        use_alternate_food: bool = False,
        remove_optional: bool = False,
        exclude_list: list[str] | None = exclude_list
        ):
    """
    if type(ingredients) is str:
        ingr_str = ingredients
    else:
        ingr_str = "\n".join(ingredients)
    ner_ingr = ner_model(ingr_str)
    ner_lines = split_recipe_obj_to_sentences(ner_ingr)
    """
    ner_lines = [ner_model(ingredient_line) for ingredient_line in ingredients]

    foods = []
    preparations = {}
    optionals = []
    if include_variety:
        varieties = {}
    if include_brand:
        brands = {}
    if use_alternate_food:
        alt_foods = []
    for line in ner_lines:
        processed_line_obj = process_ingredient(
                line,
                include_variety=include_variety,
                include_brand=include_brand
                )
        if len(processed_line_obj['food']) == 0:  # No food found, continue
            continue
        if remove_optional and processed_line_obj['optional']:
            continue

        # If there's only one food, all preps etc would belong to that food
        food = "_".join(processed_line_obj['food'][0])
        if exclude_list and food in exclude_list:
            continue
        foods.append(food)
        if len(processed_line_obj['food']) == 1:
            preparation_items = [item for sublist in processed_line_obj['prep']
                                 for item in sublist]
            if include_variety:
                variety_items = [item for sublist in processed_line_obj['var']
                                 for item in sublist]
            if include_brand:
                brand_items = [item for sublist in processed_line_obj['brand']
                               for item in sublist]
        # Otherwise, we assume the prep list correspond to the ingredient list
        else:
            preparation_items = processed_line_obj['prep'][0]
            if include_variety:
                variety_items = processed_line_obj['var'][0]
            if include_brand:
                brand_items = processed_line_obj['brand'][0]
        if len(preparation_items) > 0:
            preparations[food] = preparations.get(food, []) + preparation_items
        if include_variety and len(variety_items) > 0:
            varieties[food] = varieties.get(food, []) + variety_items
        if include_brand and len(brand_items) > 0:
            brands[food] = brands.get(food, []) + brand_items

        # Add the rest of the food in the same line as a separate datapoint
        if use_alternate_food:
            try:
                for i, alt_food in enumerate(processed_line_obj['food'][1:]):
                    food = "_".join(alt_food)
                    alt_foods.append(food)
                    preparation_items = processed_line_obj['prep'][i+1]
                    if len(preparation_items) > 0:
                        preparations[food] = \
                                preparations.get(food, []) + preparation_items

                    if include_variety:
                        variety_items = processed_line_obj['var'][i+1]
                        if len(variety_items) > 0:
                            varieties[food] = \
                                    varieties.get(food, []) + variety_items
                    if include_brand:
                        brand_items = processed_line_obj['brand'][i+1]
                        if len(brand_items) > 0:
                            brands[food] = brands.get(food, []) + brand_items
            except Exception as e:
                print(e)
                print(f"alt_food: {alt_food}")
                print(f"food: {processed_line_obj['food']}")
                print(f"prep: {processed_line_obj['prep']}")
                print(f"var: {processed_line_obj['var']}")
                print(f"brand: {processed_line_obj['brand']}")
        optionals.append(processed_line_obj['optional'])
    datapoint_obj = {
        'foods': foods,
        'preps': preparations,
        'optionals': optionals
        }
    if use_alternate_food:
        datapoint_obj['alt_foods'] = alt_foods
    if include_variety:
        datapoint_obj['vars'] = varieties
    if include_brand:
        datapoint_obj['brands'] = brands
    return datapoint_obj


# Assume recipeNLG dataset, where each ingredient line is a list item
def transform_data_to_tokens(data: list[str],
                             ner_model: spacy.lang.en.English,
                             include_variety: bool = False,
                             include_brand: bool = False,
                             use_alternate_food: bool = False,
                             remove_optional: bool = False,
                             exclude_list: list[str] | None = exclude_list):
    tokens = []
    preps = []
    optionals = []
    if include_variety:
        varieties = []
    if include_brand:
        brands = []
    if use_alternate_food:
        alt_foods = []
    for ingredients in data:
        datapoint_obj = transform_ingredients_to_tokens(
                ingredients, ner_model,
                include_variety=include_variety,
                include_brand=include_brand,
                use_alternate_food=use_alternate_food,
                remove_optional=remove_optional,
                exclude_list=exclude_list
                )
        tokens.append(datapoint_obj['foods'])
        prep_dict = datapoint_obj['preps']
        prep = []
        for k, v in prep_dict.items():
            prep.append({'key': k, 'value': v})
        preps.append(prep)
        optionals.append(datapoint_obj['optionals'])
        if include_variety:
            var_dict = datapoint_obj['vars']
            var = []
            for k, v in var_dict.items():
                var.append({'key': k, 'value': v})
            varieties.append(var)
        if include_brand:
            brand_dict = datapoint_obj['brands']
            brand = []
            for k, v in brand_dict.items():
                brand.append({'key': k, 'value': v})
            brands.append(brand)
        if use_alternate_food:
            alt_foods.append(datapoint_obj['alt_foods'])
    ret_obj = [tokens, preps, optionals]
    if include_variety:
        ret_obj.append(varieties)
    if include_brand:
        ret_obj.append(brands)
    if use_alternate_food:
        ret_obj.append(alt_foods)
    return ret_obj


def create_parquet_file(parquet_path: str, df: pl.DataFrame,
                        fields: list[ParquetDefinition],
                        force_overwrite=False
                        ):
    if os.path.isfile(parquet_path) and not force_overwrite:
        return parquet_path
    elif os.path.isfile(parquet_path) and force_overwrite:
        os.remove(parquet_path)
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
