import polars as pl


# TODO: Make this use the model function
def transform_ds_to_BOW(ds: pl.DataFrame, model):
    tfidfs = []
    for ingredients in ds.select('ingredients').iter_rows():
        datapoint_tfidf = model.transform(["\n".join(ingredients[0])])
        tfidfs.append(datapoint_tfidf)
    return ds.with_columns(pl.Series(name="tfidf", values=tfidfs))
