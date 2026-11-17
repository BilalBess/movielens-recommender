import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import recsys


def test_temporal_split_has_no_future_leak():
    rng = np.random.RandomState(0)
    r = pd.DataFrame({"user": np.repeat(np.arange(5), 20), "item": rng.randint(0, 50, 100),
                      "rating": rng.randint(1, 6, 100), "ts": rng.permutation(100)})
    train, test = recsys.temporal_split(r, 0.2)
    assert len(train) + len(test) == len(r)
    for u in range(5):
        # every training rating of a user happened before every test rating of that user
        assert train[train["user"] == u]["ts"].max() < test[test["user"] == u]["ts"].min()
    assert len(test) == 5 * 4                 # 20% of 20 ratings per user


def synthetic_ratings(n_users=60, n_items=40, seed=0):
    rng = np.random.RandomState(seed)
    bu, bi = rng.normal(0, 0.5, n_users), rng.normal(0, 0.5, n_items)
    u, i = np.meshgrid(np.arange(n_users), np.arange(n_items), indexing="ij")
    u, i = u.ravel(), i.ravel()
    keep = rng.rand(len(u)) < 0.5
    u, i = u[keep], i[keep]
    y = 3.5 + bu[u] + bi[i] + rng.normal(0, 0.05, len(u))
    return u, i, y, bu, bi, n_users, n_items


def test_bias_model_recovers_item_ordering():
    u, i, y, bu, bi, nu, ni = synthetic_ratings()
    m = recsys.BiasModel(reg_u=1, reg_i=1).fit(u, i, y, nu, ni)
    assert np.corrcoef(m.bi, bi)[0, 1] > 0.95
    assert np.corrcoef(m.bu, bu)[0, 1] > 0.95
    assert recsys.rmse(y, m.predict(u, i)) < 0.2


def test_als_improves_on_bias_model_when_there_is_interaction():
    rng = np.random.RandomState(1)
    nu, ni, k = 80, 50, 2
    P, Q = rng.normal(0, 1, (nu, k)), rng.normal(0, 1, (ni, k))
    u, i = np.meshgrid(np.arange(nu), np.arange(ni), indexing="ij")
    u, i = u.ravel(), i.ravel()
    keep = rng.rand(len(u)) < 0.4
    u, i = u[keep], i[keep]
    y = 3 + np.sum(P[u] * Q[i], axis=1) * 0.6
    bias = recsys.BiasModel(reg_u=1, reg_i=1).fit(u, i, y, nu, ni)
    als = recsys.ALS(k=2, reg=0.05, n_iter=15).fit(u, i, y, nu, ni, bias)
    assert recsys.rmse(y, als.predict(u, i)) < recsys.rmse(y, bias.predict(u, i)) * 0.6


def test_predictions_stay_in_rating_range():
    u, i, y, *_, nu, ni = synthetic_ratings()
    bias = recsys.BiasModel().fit(u, i, y, nu, ni)
    als = recsys.ALS(k=4, reg=0.1, n_iter=3).fit(u, i, y, nu, ni, bias)
    p = als.predict(u, i)
    assert p.min() >= 1 and p.max() <= 5


def test_ranking_metrics_perfect_and_worst_case():
    n_users, n_items, k = 3, 20, 3
    liked = {0: [5, 6, 7], 1: [1, 2, 3], 2: [10, 11, 12]}
    rel = sparse.lil_matrix((n_users, n_items))
    for u, items in liked.items():
        for it in items:
            rel[u, it] = 1
    train = sparse.csr_matrix((n_users, n_items))

    perfect = np.zeros((n_users, n_items))
    for u, items in liked.items():
        perfect[u, items] = 1.0
    m, top = recsys.evaluate_ranking(perfect, train, rel.tocsr(), k=k)
    assert m[f"precision@{k}"] == pytest.approx(1.0)
    assert m[f"recall@{k}"] == pytest.approx(1.0)
    assert m[f"ndcg@{k}"] == pytest.approx(1.0)

    worst = 1.0 - perfect                     # liked items get the lowest scores
    m, _ = recsys.evaluate_ranking(worst, train, rel.tocsr(), k=k)
    assert m[f"precision@{k}"] == 0


def test_items_already_seen_are_never_recommended():
    n_items, k = 15, 3
    train = sparse.csr_matrix(([1.0] * 3, ([0, 0, 0], [0, 1, 2])), shape=(1, n_items))
    rel = sparse.csr_matrix(([1.0], ([0], [5])), shape=(1, n_items))
    scores = np.zeros((1, n_items))
    scores[0, [0, 1, 2]] = 100.0              # best scores, but already seen
    _, top = recsys.evaluate_ranking(scores, train, rel, k=k)
    assert set(top[0]).isdisjoint({0, 1, 2})
