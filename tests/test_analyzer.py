import json

import numpy as np
import pandas as pd
import pytest

from agent import analyze
from analyzer import (correlation_sensitivity, iqr_outlier_report, missing_value_summary,
                      numeric_correlations, row_has_iqr_outlier, scan)


def test_tiny_values_do_not_become_false_outliers():
    df = pd.DataFrame({'x': [0, .000001, .000002, .000003, .000004]})
    report = iqr_outlier_report(df)
    assert report.iloc[0]['iqr'] == pytest.approx(.000002)
    assert report['outlier_count'].sum() == 0
    assert row_has_iqr_outlier(df).sum() == 0


def test_report_and_mask_agree_near_rounded_boundary():
    df = pd.DataFrame({'x': [0, 1, 2, 3, 4, 8.00001]})
    assert iqr_outlier_report(df)['outlier_count'].sum() == row_has_iqr_outlier(df).sum()


def test_zero_iqr_flags_only_values_outside_fences():
    df = pd.DataFrame({'x': [1, 1, 1, 1, 10]})
    assert row_has_iqr_outlier(df).tolist() == [False, False, False, False, True]


def test_pair_scope_ignores_unrelated_column():
    df = pd.DataFrame({'x': range(10), 'y': range(10), 'other': [0] * 9 + [1000]})
    assert row_has_iqr_outlier(df).sum() == 1
    result = correlation_sensitivity(df, 'x', 'y')
    assert result['flagged_rows'] == 0
    assert result['after']['n'] == 10
    assert result['after']['pearson_r'] == pytest.approx(1)


def test_pair_counts_and_nonfinite_values():
    df = pd.DataFrame({'x': [1, 2, 3, np.inf, 5], 'y': [2, 4, 6, 8, np.nan]})
    assert correlation_sensitivity(df, 'x', 'y')['before']['n'] == 3
    report = analyze(df, 'x', 'y')
    assert report['nonfinite_count'] == 1
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('df', [pd.DataFrame({'x': [1, 1, 1], 'y': [2, 3, 4]}),
                                pd.DataFrame({'x': [1, 2], 'y': [2, 4]})])
def test_undefined_correlations_explain_why(df):
    before = correlation_sensitivity(df, 'x', 'y')['before']
    assert before['pearson_r'] is None
    assert before['unavailable_reason']


def test_no_complete_pairs_after_filtering():
    df = pd.DataFrame({'x': [0, 0, 0, 0, 100], 'y': [np.nan] * 4 + [1]})
    after = correlation_sensitivity(df, 'x', 'y')['after']
    assert after['n'] == 0
    assert after['pearson_r'] is None


@pytest.mark.parametrize('df', [pd.DataFrame({'text': ['a', 'b']}),
                                pd.DataFrame({'x': [np.nan, np.nan]})])
def test_text_and_all_missing_are_supported(df):
    assert iqr_outlier_report(df).empty
    assert not row_has_iqr_outlier(df).any()
    assert numeric_correlations(df) is None
    json.dumps(scan(df), allow_nan=False)


def test_model_report_omits_raw_values_and_index():
    df = pd.DataFrame({'x': [1, 1, 1, 1, 987654321]}, index=['private'] * 5)
    report = scan(df)
    encoded = json.dumps(report)
    assert 'outlier_values' not in encoded
    assert '987654321' not in encoded
    assert 'private' not in encoded


def test_missing_percent_and_empty_input():
    assert missing_value_summary(pd.DataFrame({'x': [1, None]})).iloc[0]['missing_pct'] == 50
    with pytest.raises(ValueError, match='no rows'):
        scan(pd.DataFrame(columns=['x']))


def test_sample_shows_sensitivity():
    from pathlib import Path
    df = pd.read_csv(Path(__file__).parents[1] / 'sample_data/demo.csv')
    result = analyze(df, 'hours_studied', 'exam_score')['correlation_sensitivity']
    assert result['flagged_rows'] == 1
    assert result['after']['pearson_r'] == pytest.approx(1)
    assert result['before']['pearson_r'] < 0


def test_student_scores_expected_statistics():
    from pathlib import Path
    df = pd.read_csv(Path(__file__).parents[1] / 'sample_data/student_scores.csv')
    assert df.shape == (13, 3)
    missing = missing_value_summary(df).set_index('column')
    assert missing['missing_count'].to_dict() == {'exam_score': 1, 'hours_studied': 1, 'attendance_pct': 0}
    assert missing.loc['exam_score', 'missing_pct'] == 7.69
    outliers = iqr_outlier_report(df).set_index('column')
    assert outliers['outlier_count'].to_dict() == {'hours_studied': 1, 'exam_score': 1, 'attendance_pct': 0}
    assert outliers.loc['hours_studied', 'outlier_values'] == [35]
    assert outliers.loc['exam_score', 'outlier_values'] == [10]
    assert row_has_iqr_outlier(df).sum() == 1
    result = correlation_sensitivity(df, 'hours_studied', 'exam_score')
    assert result['before']['n'] == 11
    assert result['after']['n'] == 10
    assert result['before']['pearson_r'] == pytest.approx(-0.6442771031252582)
    assert result['after']['pearson_r'] == pytest.approx(1)
