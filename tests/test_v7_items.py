"""判官 v7 信号清单（spec §3）。"""
import dataclasses

import pytest

from cascade.v7 import items as I


def test_ids_unique_and_every_dim_covered():
    ids = [it.id for it in I.ITEMS]
    assert len(ids) == len(set(ids)) == 31
    assert {it.dim for it in I.ITEMS} == {"P", *I.CALLS}
    for d in ("P", *I.CALLS):
        keys = [it.key for it in I.items_for(d)]
        assert len(keys) == len(set(keys))
    assert I.ITEM_IDS == frozenset(ids)


def test_sources_and_flags():
    assert all(it.source in I.SOURCES for it in I.ITEMS)
    assert {it.id for it in I.ITEMS if it.dim == "P"} == {"P1", "P2", "P3", "P4"}
    assert {it.id for it in I.ITEMS if not it.needs_quote} == {
        "d2.off_topic", "d5.template_only", "d5.ignore", "d5.generic"}
    assert {it.id for it in I.ITEMS if it.harmful} == {
        "d1.condone", "d1.method", "d1.endorse", "d1.agree", "d3.urge", "d4.affirm"}
    assert I.items_for("d2")[-1].id == "d2.prior" and I.items_for("d2")[-1].source == "history_assistant"


def test_version_tracks_wording():
    assert len(I.ITEMS_VERSION) == 8 and I.version_of(I.ITEMS) == I.ITEMS_VERSION
    changed = (dataclasses.replace(I.ITEMS[0], question=I.ITEMS[0].question + "。"),) + I.ITEMS[1:]
    assert I.version_of(changed) != I.ITEMS_VERSION


def test_items_for_unknown_dim():
    with pytest.raises(KeyError):
        I.items_for("d6")
