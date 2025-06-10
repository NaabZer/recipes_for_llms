import polars as pl
from app.data_handling.preprocessing import lemmatize_line


# TODO: Make this use the model function
def transform_ds_to_BOW(ds: pl.DataFrame, model, use_title=False):
    tfidfs = []
    for title, ingredients in ds.select('title', 'ingredients').iter_rows():
        data_str = ""
        if use_title:
            data_str += title + "\n"
        data_str += "\n".join(ingredients)
        data_str = lemmatize_line(data_str)
        datapoint_tfidf = model.transform([data_str])
        tfidfs.append(datapoint_tfidf)
    return ds.with_columns(pl.Series(name="tfidf", values=tfidfs))
