import json
from pathlib import Path

from filaments.data import CACHE
from filaments.folds import groups_by_time, split_folds


def test_simultaneous_pairs_stay_together_and_folds_balanced():
    index_path = CACHE / "index.json"
    if not index_path.exists():
        # The public inference package includes the original IDs without labels.
        index_path = Path(__file__).resolve().parents[1] / "metadata/original_split.json"
    stems = json.loads(index_path.read_text())["train"]
    folds = split_folds(stems, 5, 20260926)
    assert sorted(sum(folds, [])) == sorted(stems)
    sizes = [len(f) for f in folds]
    assert max(sizes) - min(sizes) <= 2
    where = {s: k for k, f in enumerate(folds) for s in f}
    for g in groups_by_time(stems):
        assert len({where[s] for s in g}) == 1
    assert split_folds(stems, 5, 20260926) == folds  # determinista
