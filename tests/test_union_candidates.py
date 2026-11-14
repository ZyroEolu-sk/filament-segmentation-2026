import numpy as np

from filaments.postprocess import PostCfg
from filaments.rescore import MIN_CANDIDATE, append_label_candidates, candidates, union_candidates, with_direct_confidence

CFG = PostCfg(t_fg=0.5, t_b=0.5, split=False, merge_dist=0, min_area=400, min_score=0.9)


def _prob(boxes, value=0.9):
    p = np.zeros((2, 128, 128), np.float32)
    for y0, y1, x0, x1 in boxes:
        p[0, y0:y1, x0:x1] = value
    return p


def test_union_adds_only_novel_disjoint_candidates():
    ens = _prob([(10, 30, 10, 60)])                      # un candidato del ensemble
    m1 = _prob([(10, 30, 10, 60), (80, 100, 10, 60)])    # repite el del ensemble y añade uno nuevo
    m2 = _prob([(12, 32, 12, 62)])                       # casi igual al del ensemble (IoU alto) -> no se añade
    lab, origin = union_candidates(ens, [m1, m2], CFG)
    assert int(lab.max()) == 2 and origin.tolist() == [0, 1]
    assert np.array_equal(lab == 1, candidates(ens, CFG) == 1)
    assert (lab == 2).sum() == 20 * 50


def test_union_drops_candidates_that_lose_most_pixels():
    ens = _prob([(10, 30, 10, 60)])
    m1 = _prob([(10, 30, 10, 90)])  # solapa con IoU 0.625 >= 0.5 -> descartado por IoU
    m3 = _prob([(20, 40, 10, 60)])  # IoU 1/3 < 0.5, pero pierde la mitad de sus píxeles -> keep_frac 0.5 lo deja justo; con 0.6 se descarta
    lab, origin = union_candidates(ens, [m1], CFG)
    assert int(lab.max()) == 1
    lab, origin = union_candidates(ens, [m3], CFG, keep_frac=0.6)
    assert int(lab.max()) == 1
    lab, origin = union_candidates(ens, [m3], CFG, keep_frac=0.5)
    assert int(lab.max()) == 2 and (lab == 2).sum() == 10 * 50 >= MIN_CANDIDATE


def test_direct_labels_preserve_touching_detector_instances():
    ens = _prob([(10, 30, 10, 60)])
    direct = np.zeros((128, 128), np.int16)
    direct[60:80, 10:40] = 1
    direct[60:80, 40:70] = 2  # toca a la primera; un mapa binario las fusionaría
    base = candidates(ens, CFG)
    lab, added = append_label_candidates(base, direct)
    assert added == 2
    assert (lab == 2).sum() == 600
    assert (lab == 3).sum() == 600
    assert (base == 1).sum() == 1000  # entrada intacta
    union, origin = union_candidates(ens, [], CFG, direct_labs=[direct])
    assert np.array_equal(lab, union)
    assert origin.tolist() == [0, 1, 1]


def test_direct_confidence_uses_detector_scores_on_retained_masks():
    import pandas as pd

    lab = np.zeros((4, 5), np.int32)
    lab[:2, :2] = 1
    lab[2:, 3:] = 2
    prob = np.zeros((2, 4, 5), np.float32)
    prob[0, :2, :2] = 0.3
    prob[0, 2:, 3:] = 0.8
    df = with_direct_confidence(pd.DataFrame({"id": [1, 2]}), [prob], lab)
    assert np.allclose(df.direct_conf, [0.3, 0.8])
