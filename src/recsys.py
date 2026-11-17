"""Movie recommender on MovieLens 1M, written with numpy/scipy (no recommender library).

Two tasks:
  1) rating prediction  -> RMSE
  2) top-10 recommendation -> precision, recall, NDCG, coverage, popularity bias
Split: for each user, the last 20% of the ratings (by time) are the test set.

Run:  python src/recsys.py
"""
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REP = ROOT / "reports"
FIG = REP / "figures"
FIG.mkdir(parents=True, exist_ok=True)
URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
SEED = 0
K = 10


def load():
    folder = DATA / "ml-1m"
    if not folder.exists():
        DATA.mkdir(exist_ok=True)
        z = DATA / "ml-1m.zip"
        z.write_bytes(requests.get(URL, timeout=120).content)
        zipfile.ZipFile(z).extractall(DATA)
    r = pd.read_csv(folder / "ratings.dat", sep="::", engine="python", names=["user", "item", "rating", "ts"])
    m = pd.read_csv(folder / "movies.dat", sep="::", engine="python", names=["item", "title", "genres"],
                    encoding="latin-1")
    return r, m


def temporal_split(r, test_frac=0.2):
    r = r.sort_values(["user", "ts"]).copy()
    r["rank"] = r.groupby("user").cumcount()
    r["n"] = r.groupby("user")["item"].transform("size")
    is_test = r["rank"] >= (r["n"] * (1 - test_frac)).round()
    return r[~is_test].drop(columns=["rank", "n"]), r[is_test].drop(columns=["rank", "n"])


# ---------------------------------------------------------------- rating models
class BiasModel:
    """mu + b_u + b_i, with regularisation (a few alternating passes)."""

    def __init__(self, reg_u=10, reg_i=10, n_iter=10):
        self.reg_u, self.reg_i, self.n_iter = reg_u, reg_i, n_iter

    def fit(self, u, i, y, n_users, n_items):
        self.mu = y.mean()
        bu, bi = np.zeros(n_users), np.zeros(n_items)
        for _ in range(self.n_iter):
            bi = np.bincount(i, weights=y - self.mu - bu[u], minlength=n_items) / (
                self.reg_i + np.bincount(i, minlength=n_items))
            bu = np.bincount(u, weights=y - self.mu - bi[i], minlength=n_users) / (
                self.reg_u + np.bincount(u, minlength=n_users))
        self.bu, self.bi = bu, bi
        return self

    def predict(self, u, i):
        return self.mu + self.bu[u] + self.bi[i]


class ALS:
    """Matrix factorisation on the residuals of the bias model, solved with Alternating Least Squares."""

    def __init__(self, k=32, reg=10.0, n_iter=12, seed=SEED):
        self.k, self.reg, self.n_iter, self.seed = k, reg, n_iter, seed

    def fit(self, u, i, y, n_users, n_items, bias, val=None):
        self.bias = bias
        res = y - bias.predict(u, i)
        R = sparse.csr_matrix((res, (u, i)), shape=(n_users, n_items))
        Rt = R.T.tocsr()
        rng = np.random.RandomState(self.seed)
        self.P = rng.normal(0, 0.1, (n_users, self.k))
        self.Q = rng.normal(0, 0.1, (n_items, self.k))
        self.history = []
        for it in range(self.n_iter):
            self.P = self._solve(R, self.Q)
            self.Q = self._solve(Rt, self.P)
            if val is not None:
                self.history.append(rmse(val[2], self.predict(val[0], val[1])))
        return self

    def _solve(self, M, F):
        out = np.zeros((M.shape[0], self.k))
        eye = np.eye(self.k)
        for row in range(M.shape[0]):
            s, e = M.indptr[row], M.indptr[row + 1]
            if s == e:
                continue
            idx, vals = M.indices[s:e], M.data[s:e]
            Fi = F[idx]
            out[row] = np.linalg.solve(Fi.T @ Fi + self.reg * (e - s) * eye, Fi.T @ vals)
        return out

    def predict(self, u, i):
        return np.clip(self.bias.predict(u, i) + np.sum(self.P[u] * self.Q[i], axis=1), 1, 5)

    def predict_all(self):
        return self.bias.mu + self.bias.bu[:, None] + self.bias.bi[None, :] + self.P @ self.Q.T


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


# ---------------------------------------------------------------- ranking metrics
def evaluate_ranking(scores, train_mat, test_rel, k=K, pop=None):
    """scores: dense (users x items). train_mat: items to exclude. test_rel: sparse 0/1 of liked test items."""
    scores = scores.copy()
    scores[train_mat.nonzero()] = -np.inf
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
    top = np.take_along_axis(top, order, axis=1)

    rel = test_rel.tocsr()
    n_rel = np.asarray(rel.sum(axis=1)).ravel()
    users = np.where(n_rel > 0)[0]
    hits = np.array([rel[u, top[u]].toarray().ravel() for u in users])
    disc = 1 / np.log2(np.arange(2, k + 2))
    dcg = (hits * disc).sum(axis=1)
    idcg = np.array([disc[:min(int(n_rel[u]), k)].sum() for u in users])
    out = {
        f"precision@{k}": float(hits.mean()),
        f"recall@{k}": float(np.mean(hits.sum(axis=1) / n_rel[users])),
        f"ndcg@{k}": float(np.mean(dcg / idcg)),
        "coverage": float(len(np.unique(top[users])) / scores.shape[1]),
    }
    if pop is not None:
        out["avg_log_popularity_of_recs"] = float(np.log1p(pop[top[users]]).mean())
    return out, top


def main():
    r, movies = load()
    # re-index users and items from 0
    r["u"] = r["user"].astype("category").cat.codes
    r["i"] = r["item"].astype("category").cat.codes
    item_ids = r["item"].astype("category").cat.categories
    n_users, n_items = r["u"].nunique(), r["i"].nunique()
    train, test = temporal_split(r)
    print(f"users {n_users} | items {n_items} | train {len(train)} | test {len(test)} | "
          f"density {len(r) / (n_users * n_items):.3%}")

    ut, it_, yt = train["u"].values, train["i"].values, train["rating"].values.astype(float)
    us, is_, ys = test["u"].values, test["i"].values, test["rating"].values.astype(float)

    # ------------------------------------------------ 1) rating prediction
    res = {"Global mean": rmse(ys, np.full(len(ys), yt.mean()))}
    item_mean = pd.Series(yt).groupby(it_).mean()
    res["Item mean"] = rmse(ys, pd.Series(is_).map(item_mean).fillna(yt.mean()).values)
    bias = BiasModel().fit(ut, it_, yt, n_users, n_items)
    res["Bias model (mu + user + item)"] = rmse(ys, bias.predict(us, is_))

    # choose k / reg on a validation slice taken from the END of the training data (time-based again)
    tr2, va2 = temporal_split(train, 0.15)
    b2 = BiasModel().fit(tr2["u"].values, tr2["i"].values, tr2["rating"].values.astype(float), n_users, n_items)
    best, best_v = None, 9
    for k in [16, 32]:
        for reg in [0.05, 0.1, 0.2]:
            m = ALS(k=k, reg=reg, n_iter=8).fit(tr2["u"].values, tr2["i"].values, tr2["rating"].values.astype(float),
                                                  n_users, n_items, b2)
            v = rmse(va2["rating"].values.astype(float), m.predict(va2["u"].values, va2["i"].values))
            print(f"ALS k={k} reg={reg}: val RMSE {v:.4f}")
            if v < best_v:
                best, best_v = (k, reg), v
    print("best ALS:", best)
    als = ALS(k=best[0], reg=best[1], n_iter=12).fit(ut, it_, yt, n_users, n_items, bias, val=(us, is_, ys))
    res[f"ALS matrix factorisation (k={best[0]})"] = rmse(ys, als.predict(us, is_))
    rmse_table = pd.Series(res).round(4)
    print(rmse_table)
    rmse_table.to_csv(REP / "rmse_results.csv", header=["test_rmse"])

    # ------------------------------------------------ 2) top-10 recommendation
    train_mat = sparse.csr_matrix((np.ones(len(train)), (ut, it_)), shape=(n_users, n_items))
    liked = test[test["rating"] >= 4]
    rel = sparse.csr_matrix((np.ones(len(liked)), (liked["u"].values, liked["i"].values)), shape=(n_users, n_items))
    pop = np.asarray(train_mat.sum(axis=0)).ravel()

    rank_res, tops = {}, {}
    # random
    rng = np.random.RandomState(SEED)
    rank_res["Random"], tops["Random"] = evaluate_ranking(rng.rand(n_users, n_items), train_mat, rel, pop=pop)
    # popularity
    rank_res["Most popular"], tops["Most popular"] = evaluate_ranking(
        np.tile(pop, (n_users, 1)).astype(float), train_mat, rel, pop=pop)
    # item-item collaborative filtering (cosine on who-watched-what)
    norms = np.sqrt(np.asarray(train_mat.multiply(train_mat).sum(axis=0)).ravel()) + 1e-9
    Xn = train_mat @ sparse.diags(1 / norms)
    S = (Xn.T @ Xn).toarray()
    np.fill_diagonal(S, 0)
    knn_scores = train_mat @ S
    rank_res["Item-item CF (cosine)"], tops["Item-item CF (cosine)"] = evaluate_ranking(
        np.asarray(knn_scores), train_mat, rel, pop=pop)
    # matrix factorisation scores (predicted rating)
    rank_res["ALS predicted rating"], tops["ALS predicted rating"] = evaluate_ranking(
        als.predict_all(), train_mat, rel, pop=pop)
    # ALS only among items with at least 50 ratings (avoids recommending obscure movies with a lucky bias)
    scores = als.predict_all()
    scores[:, pop < 50] = -np.inf
    rank_res["ALS (items with >= 50 ratings)"], tops["ALS (items with >= 50 ratings)"] = evaluate_ranking(
        scores, train_mat, rel, pop=pop)
    rank_table = pd.DataFrame(rank_res).T.round(4)
    print(rank_table)
    rank_table.to_csv(REP / "ranking_results.csv")

    # ------------------------------------------------ example for one user
    titles = movies.set_index("item")["title"]
    u0 = 0
    seen = train[(train["u"] == u0) & (train["rating"] >= 5)].sort_values("ts").tail(5)["item"].map(titles).tolist()
    rec = [titles[item_ids[j]] for j in tops["Item-item CF (cosine)"][u0]]
    ex = {"user_id": 1, "recently_loved_5_stars": seen, "item_item_recommendations": rec}
    (REP / "example_user.json").write_text(json.dumps(ex, indent=2))
    print(json.dumps(ex, indent=2))

    # ------------------------------------------------ figures
    fig, ax = plt.subplots(figsize=(8, 4))
    rmse_table.sort_values(ascending=False).plot.barh(ax=ax, color="#2b6cb0")
    ax.set_xlim(0.8, 1.3)
    ax.set_xlabel("Test RMSE (lower is better)")
    ax.set_title("Rating prediction")
    fig.tight_layout()
    fig.savefig(FIG / "rmse.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    rank_table[["precision@10", "ndcg@10"]].plot.bar(ax=axes[0], rot=30)
    axes[0].set_title("Top-10 recommendation quality (test)")
    rank_table["coverage"].plot.bar(ax=axes[1], rot=30, color="#2f855a")
    axes[1].set_title("Catalog coverage (share of movies ever recommended)")
    fig.tight_layout()
    fig.savefig(FIG / "ranking.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(als.history) + 1), als.history, marker="o")
    ax.set_xlabel("ALS iteration")
    ax.set_ylabel("Test RMSE")
    ax.set_title("ALS convergence")
    fig.tight_layout()
    fig.savefig(FIG / "als_convergence.png", dpi=150)
    plt.close(fig)

    (REP / "summary.json").write_text(json.dumps(
        {"n_users": int(n_users), "n_items": int(n_items), "n_train": len(train), "n_test": len(test),
         "best_als": {"k": best[0], "reg": best[1]}}, indent=2))


if __name__ == "__main__":
    main()
