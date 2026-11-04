"""The UAI file must be read with each factor's declared axis order."""
from __future__ import annotations

import json

import numpy as np
import pytest


def test_asia_header_and_scopes(asia_model):
    model = asia_model
    assert model.variables == 8
    assert model.dims == (2,) * 8
    assert [factor.scope for factor in model.factors] == [
        (0,), (1, 0), (2,), (3, 2), (4, 2), (5, 3, 1), (6, 5), (7, 5, 4)]


def test_visit_and_tuberculosis_tables(asia_model):
    """The sheet's first two tables, read with the last scope axis varying fastest."""
    p_a = asia_model.factors[0].entries
    assert p_a == (0.01, 0.99)
    p_t_given_a = asia_model.factors[1].table()          # rows T, columns A
    assert p_t_given_a[0, 0] == pytest.approx(0.05)
    assert p_t_given_a[0, 1] == pytest.approx(0.01)
    assert p_t_given_a[1, 0] == pytest.approx(0.95)
    assert p_t_given_a[1, 1] == pytest.approx(0.99)
    assert p_t_given_a.sum(axis=0) == pytest.approx([1.0, 1.0])


def test_smoking_lung_bronchitis_columns_normalized(asia_model):
    for index in (3, 4):
        table = asia_model.factors[index].table()
        assert table.sum(axis=0) == pytest.approx([1.0, 1.0]), index


def test_tool_reports_the_parsed_model(asia, asia_model):
    reported = asia['prior']['model']
    assert reported['variables'] == asia_model.variables
    assert reported['dims'] == list(asia_model.dims)
    assert len(reported['factors']) == len(asia_model.factors)
    for got, want in zip(reported['factors'], asia_model.factors):
        assert list(got['scope']) == list(want.scope)
        assert list(got['dims']) == list(want.dims)
        np.testing.assert_allclose(got['entries'], want.entries, rtol=0, atol=1e-15)


def test_observations_are_recorded(asia):
    assert asia['prior']['evidence']['observations'] == {}
    assert asia['AD']['evidence']['observations'] == {'0': 0, '7': 0}
    assert asia['ADX']['evidence']['observations'] == {'0': 0, '6': 0, '7': 0}
