import numpy as np
import pandas as pd
import pytest

from mmm.transforms import geometric_adstock, hill_saturation, make_lag_tensor


def test_lag_tensor_never_crosses_group_boundary():
    frame = pd.DataFrame({"Show": ["A", "A", "B"], "Season": [1, 1, 1]})
    media = np.array([[1.0], [2.0], [9.0]])
    lagged, valid = make_lag_tensor(frame, media, max_lag=2)
    assert lagged[1, 0].tolist() == [2.0, 1.0, 0.0]
    assert lagged[2, 0].tolist() == [9.0, 0.0, 0.0]
    assert valid[2].tolist() == [1.0, 0.0, 0.0]


def test_normalized_adstock_matches_hand_calculation():
    lagged = np.array([[[2.0, 1.0]]])
    valid = np.array([[1.0, 1.0]])
    actual = geometric_adstock(lagged, valid, np.array([[0.5]]))[0, 0, 0]
    assert actual == pytest.approx((2 + .5) / 1.5)


def test_hill_limits_and_invalid_values():
    x = np.array([[[0.0, 1.0, 1000.0]]])
    result = hill_saturation(x, np.array([[1.0]]))
    assert result[0, 0, 0] == 0
    assert result[0, 0, 1] == pytest.approx(.5)
    assert result[0, 0, 2] < 1
    with pytest.raises(ValueError):
        hill_saturation(np.array([[[-1.0]]]), np.array([[1.0]]))
