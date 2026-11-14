import numpy as np
import pandas as pd

from filaments.rescore import MEMBER_FEATURES, member_features, with_member_features


def _probs(fill_a, fill_b):
    p = np.zeros((2, 32, 32), np.float32)
    p[0, 4:12, 4:20] = fill_a
    p[0, 20:28, 4:20] = fill_b
    return p


def test_member_features_agreement_and_disagreement():
    lab = np.zeros((32, 32), np.int32)
    lab[4:12, 4:20] = 1
    lab[20:28, 4:20] = 2
    m1, m2 = _probs(0.9, 0.9), _probs(0.9, 0.1)
    df = member_features([m1, m2], lab, t_fg=0.5)
    assert list(df.columns) == ["id"] + MEMBER_FEATURES
    a, b = df.set_index("id").loc[1], df.set_index("id").loc[2]
    assert a.mem_p_std == 0 and a.mem_cov_min == 1 and a.mem_detect_frac == 1
    assert b.mem_p_std > 0.3 and b.mem_cov_min == 0 and b.mem_cov_mean == 0.5 and b.mem_detect_frac == 0.5
    assert np.isclose(b.mem_p_range, 0.8)


def test_member_features_single_member_is_degenerate_and_merge_keeps_rows():
    lab = np.zeros((32, 32), np.int32)
    lab[4:12, 4:20] = 1
    m1 = _probs(0.7, 0.0)
    df = pd.DataFrame({"id": [1], "area": [128]})
    out = with_member_features(df, [m1], lab, 0.5)
    assert len(out) == 1 and out.mem_p_std[0] == 0 and np.isclose(out.mem_p_min[0], 0.7)
    assert with_member_features(pd.DataFrame(), [m1], lab, 0.5).empty
