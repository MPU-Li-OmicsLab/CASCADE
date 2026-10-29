from cascade.goldeval.agreement import (cohen_kappa, weighted_kappa, accuracy,
                                        prf1, confusion, pairwise_iaa)


def test_cohen_perfect_and_chance_and_negative():
    assert cohen_kappa([(1, 1), (1, 1), (0, 0), (0, 0)]) == 1.0
    assert cohen_kappa([(1, 1), (1, 0), (0, 1), (0, 0)]) == 0.0          # pe==po
    assert cohen_kappa([(1, 1), (1, 1), (1, 0), (0, 1)]) == -0.3333      # worse-than-chance


def test_cohen_none_and_empty():
    assert cohen_kappa([]) is None
    assert cohen_kappa([(None, 1), (1, None)]) is None                  # all excluded
    assert cohen_kappa([(1, 1), (None, 0), (0, 0)]) == 1.0              # None pair dropped, rest perfect


def test_weighted_kappa_known_value_quadratic():
    # hand-computed: po=0.2, pe=0.52 -> kappa = 1 - 0.2/0.52 = 0.6154
    pairs = [(0, 0), (2, 2), (0, 0), (2, 2), (0, 2)]
    assert weighted_kappa(pairs, (0, 1, 2)) == 0.6154
    assert weighted_kappa([(0, 0), (1, 1), (2, 2)], (0, 1, 2)) == 1.0   # perfect


def test_weighted_penalizes_far_more_than_near():
    near = weighted_kappa([(0, 0), (1, 1), (2, 2), (1, 2), (2, 1)], (0, 1, 2))
    far = weighted_kappa([(0, 0), (1, 1), (2, 2), (0, 2), (2, 0)], (0, 1, 2))
    assert near > far     # a 1-step disagreement hurts less than a 2-step one


def test_accuracy_exact_and_within1():
    a = accuracy([("a", "a"), ("a", "b"), ("a", "c")], ("a", "b", "c"))
    assert a == {"exact": 0.3333, "within1": 0.6667, "n": 3}   # a==a exact; a~b within1; a~c not


def test_prf1_known():
    r = prf1([(1, 1), (1, 0), (0, 1), (1, 1)], positive=1)     # tp=2 fp=1 fn=1
    assert r == {"precision": 0.6667, "recall": 0.6667, "f1": 0.6667, "n": 4}


def test_confusion_matrix():
    m = confusion([("a", "a"), ("a", "b"), ("b", "b")], ("a", "b"))
    assert m == [[1, 1], [0, 1]]   # row a: 1->a,1->b ; row b: 0->a,1->b


def test_pairwise_iaa_two_annotators():
    ann = {"ann1": {"i1": 1, "i2": 0, "i3": 1}, "ann2": {"i1": 1, "i2": 0, "i3": 1}}
    assert pairwise_iaa(ann, "binary", (0, 1)) == 1.0
    assert pairwise_iaa({"ann1": {"i1": 1}}, "binary", (0, 1)) is None   # <2 annotators
