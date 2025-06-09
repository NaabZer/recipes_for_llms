import polars as pl
import scipy
from sklearn.metrics.pairwise import cosine_similarity


def reciprocal_rank(ranked_ds: pl.dataframe.frame.DataFrame, correct_idx: int):
    filtered_ds = ranked_ds.with_row_index("id", offset=1)\
            .filter(pl.col("index") == correct_idx)
    if filtered_ds.shape[0] == 0:  # Index not found, reciprocal rank is 0
        return 0
    return 1 / filtered_ds.select('id')[0, 0]


def rank_results(ds: pl.DataFrame, embedding: scipy.sparse._csr.csr_matrix,
                 embedding_col='tfidf'):
    similarities = cosine_similarity(embedding,
                                     scipy.sparse.vstack(ds[embedding_col]))
    ranked_ds = ds.with_columns(
            pl.Series(name='rank', values=similarities[0])
            ).filter(pl.col('rank') > 0).sort('rank', descending=True)
    return ranked_ds
