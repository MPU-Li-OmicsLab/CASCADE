import numpy as np
import pytest

from cascade.analysis.stats import (cluster_boot_ci, paired_wilcoxon, icc_1, spearman,
                                    kmeans, silhouette, choose_k)


def test_cluster_boot_ci_brackets_point_estimate():
    by = {"p1": [1.0, 0.0, 1.0], "p2": [0.0, 0.0, 1.0], "p3": [1.0, 1.0, 1.0], "p4": [0.0, 1.0]}
    lo, hi = cluster_boot_ci(by, seed=0, n_boot=300)
    point = sum(sum(v) for v in by.values()) / sum(len(v) for v in by.values())
    assert lo <= point <= hi
    assert 0.0 <= lo < hi <= 1.0


def test_cluster_boot_ci_single_cluster_degenerate():
    lo, hi = cluster_boot_ci({"p1": [0.2, 0.4]}, seed=0, n_boot=50)
    assert lo == hi == pytest.approx(0.3)


def test_paired_wilcoxon_known():
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    b = [x + 1.0 for x in a]
    out = paired_wilcoxon(a, b)
    assert out["n"] == 8
    assert out["mean_diff"] == pytest.approx(-1.0)
    assert out["p"] is not None and out["p"] < 0.05
    assert out["ci"][0] <= -1.0 <= out["ci"][1]


def test_paired_wilcoxon_too_small_or_all_equal():
    assert paired_wilcoxon([1.0, 2.0], [1.0, 2.0])["p"] is None
    out = paired_wilcoxon([1.0] * 8, [1.0] * 8)
    assert out["p"] is None and out["mean_diff"] == 0.0


def test_icc_identical_within_cell_is_one_and_noise_near_zero():
    cells = [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    out = icc_1(cells)
    assert out["icc11"] == pytest.approx(1.0) and out["icc1k"] == pytest.approx(1.0)
    assert out["k"] == 3
    rng = np.random.default_rng(0)
    noise = rng.normal(size=(60, 3)).tolist()
    assert abs(icc_1(noise)["icc11"]) < 0.2


def test_spearman():
    rho, p = spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5])
    assert rho == pytest.approx(0.8)


def test_kmeans_separates_two_blobs_and_silhouette_high():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.1, size=(20, 4)); b = rng.normal(5, 0.1, size=(20, 4))
    X = np.vstack([a, b])
    labels, cents = kmeans(X, 2, seed=0)
    assert len(set(labels[:20])) == 1 and len(set(labels[20:])) == 1
    assert labels[0] != labels[20]
    assert cents.shape == (2, 4)
    s = silhouette(X, labels)
    assert 0.9 < s <= 1.0
    k, scores = choose_k(X, ks=(2, 3, 4), seed=0)
    assert k == 2 and set(scores) == {2, 3, 4}
