"""Tests for Week 3's data preparation: the split and the feature preprocessor.

Two properties are checked, and they are the two the week is graded on:

1. **Scaling works and is fitted on the training set only.** After
   ``fit_transform`` on the training features, every column has mean ~0 and
   standard deviation ~1. The test features, transformed with the *training*
   statistics, are close to but deliberately not exactly 0/1 — if they were
   exactly 0/1 the scaler would have been fitted on them, which is leakage.
2. **The split is stratified.** Every class holds the same share of the training
   rows as of the test rows, to within one row's worth of rounding.

Most tests run on a synthetic frame so they pass even without the CSV present;
the few that need the real 2,200 rows are marked with ``requires_raw_dataset``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.data import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    EXPECTED_LABELS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    class_proportions,
    load_data,
    stratified_split,
)
from src.preprocessing import build_preprocessing_pipeline, build_preprocessor
from tests.conftest import requires_raw_dataset

SYNTHETIC_FEATURES = ["small", "large"]


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    """Build a deterministic frame with two very differently scaled features.

    ``small`` sits around 5 with a spread of about 1; ``large`` sits around
    5,000 with a spread of about 1,000. Four classes hold 50 rows each, so the
    frame is balanced in the same way the real dataset is.
    """
    rng = np.random.default_rng(seed=0)
    rows_per_class = 50
    frames = []
    for offset, name in enumerate(["one", "two", "three", "four"]):
        frames.append(
            pd.DataFrame(
                {
                    "small": rng.normal(5.0 + offset, 1.0, rows_per_class),
                    "large": rng.normal(5_000.0 + 500.0 * offset, 1_000.0, rows_per_class),
                    TARGET_COLUMN: name,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# stratified_split
# --------------------------------------------------------------------------


def test_split_sizes_follow_test_size(synthetic_frame):
    train, test = stratified_split(synthetic_frame, test_size=0.2)

    assert len(train) + len(test) == len(synthetic_frame)
    assert len(test) == pytest.approx(0.2 * len(synthetic_frame), abs=1)


def test_split_keeps_every_column(synthetic_frame):
    train, test = stratified_split(synthetic_frame)

    assert list(train.columns) == list(synthetic_frame.columns)
    assert list(test.columns) == list(synthetic_frame.columns)


def test_split_is_stratified(synthetic_frame):
    train, test = stratified_split(synthetic_frame)

    train_shares = class_proportions(train)
    test_shares = class_proportions(test)

    assert list(train_shares.index) == list(test_shares.index)
    # One row of a 40-row test set is 2.5%; anything stratified is far inside 1%.
    assert np.allclose(train_shares.to_numpy(), test_shares.to_numpy(), atol=0.01)


def test_split_rows_are_disjoint_and_complete(synthetic_frame):
    train, test = stratified_split(synthetic_frame)

    rejoined = pd.concat([train, test]).sort_values(SYNTHETIC_FEATURES)
    original = synthetic_frame.sort_values(SYNTHETIC_FEATURES)

    assert len(rejoined) == len(original)
    assert np.allclose(
        rejoined[SYNTHETIC_FEATURES].to_numpy(), original[SYNTHETIC_FEATURES].to_numpy()
    )


def test_split_is_reproducible_with_the_default_seed(synthetic_frame):
    first_train, first_test = stratified_split(synthetic_frame)
    second_train, second_test = stratified_split(synthetic_frame)

    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_test, second_test)


def test_split_differs_with_a_different_seed(synthetic_frame):
    default_train, _ = stratified_split(synthetic_frame)
    other_train, _ = stratified_split(synthetic_frame, random_state=DEFAULT_RANDOM_STATE + 1)

    assert not default_train[SYNTHETIC_FEATURES].equals(other_train[SYNTHETIC_FEATURES])


def test_split_resets_the_index(synthetic_frame):
    train, test = stratified_split(synthetic_frame)

    assert list(train.index) == list(range(len(train)))
    assert list(test.index) == list(range(len(test)))


def test_split_rejects_a_missing_target(synthetic_frame):
    with pytest.raises(KeyError, match="not found"):
        stratified_split(synthetic_frame, target="does_not_exist")


@pytest.mark.parametrize("bad_size", [0.0, 1.0, -0.1, 1.5])
def test_split_rejects_an_out_of_range_test_size(synthetic_frame, bad_size):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        stratified_split(synthetic_frame, test_size=bad_size)


def test_split_rejects_a_class_with_a_single_row(synthetic_frame):
    solo_row = pd.DataFrame([{"small": 1.0, "large": 1.0, TARGET_COLUMN: "solo"}])
    lonely = pd.concat([synthetic_frame, solo_row], ignore_index=True)

    with pytest.raises(ValueError, match="at least 2 rows per class"):
        stratified_split(lonely)


def test_class_proportions_sum_to_one(synthetic_frame):
    shares = class_proportions(synthetic_frame)

    assert shares.sum() == pytest.approx(1.0)
    assert list(shares.index) == sorted(shares.index)


def test_class_proportions_rejects_a_missing_target(synthetic_frame):
    with pytest.raises(KeyError, match="not found"):
        class_proportions(synthetic_frame, target="does_not_exist")


# --------------------------------------------------------------------------
# build_preprocessor
# --------------------------------------------------------------------------


def test_build_preprocessor_returns_an_unfitted_column_transformer():
    preprocessor = build_preprocessor(SYNTHETIC_FEATURES)

    assert isinstance(preprocessor, ColumnTransformer)
    name, transformer, columns = preprocessor.transformers[0]
    assert name == "numeric"
    assert isinstance(transformer, StandardScaler)
    assert columns == SYNTHETIC_FEATURES
    assert not hasattr(preprocessor, "transformers_")


def test_build_preprocessor_defaults_to_the_project_features():
    preprocessor = build_preprocessor()

    assert preprocessor.transformers[0][2] == list(FEATURE_COLUMNS)


def test_build_preprocessor_rejects_empty_or_duplicated_columns():
    with pytest.raises(ValueError, match="at least one column"):
        build_preprocessor([])
    with pytest.raises(ValueError, match="duplicate column"):
        build_preprocessor(["small", "small"])


def test_scaled_training_features_have_zero_mean_and_unit_std(synthetic_frame):
    train, _ = stratified_split(synthetic_frame)
    preprocessor = build_preprocessor(SYNTHETIC_FEATURES)

    scaled = preprocessor.fit_transform(train)

    assert scaled.shape == (len(train), len(SYNTHETIC_FEATURES))
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-9)
    # StandardScaler divides by the population std (ddof=0), so measure it the
    # same way rather than with numpy's default sample std.
    assert np.allclose(scaled.std(axis=0, ddof=0), 1.0, atol=1e-9)


def test_scaling_is_fitted_on_train_only(synthetic_frame):
    train, test = stratified_split(synthetic_frame)
    preprocessor = build_preprocessor(SYNTHETIC_FEATURES)

    preprocessor.fit(train)
    scaled_test = preprocessor.transform(test)

    # Close to standardised, because train and test come from one distribution.
    # The tolerance is loose: a 40-row test set drifts noticeably by chance.
    assert np.allclose(scaled_test.mean(axis=0), 0.0, atol=0.5)
    # ...but not *exactly* standardised: exact zeros would mean the scaler had
    # seen the test rows, which is the leak this project forbids.
    assert not np.allclose(scaled_test.mean(axis=0), 0.0, atol=1e-12)


def test_learned_statistics_come_from_the_training_rows(synthetic_frame):
    train, _ = stratified_split(synthetic_frame)
    preprocessor = build_preprocessor(SYNTHETIC_FEATURES)
    preprocessor.fit(train)

    scaler = preprocessor.named_transformers_["numeric"]

    assert np.allclose(scaler.mean_, train[SYNTHETIC_FEATURES].mean().to_numpy())
    assert np.allclose(scaler.scale_, train[SYNTHETIC_FEATURES].std(ddof=0).to_numpy())


def test_transform_is_reversible(synthetic_frame):
    train, test = stratified_split(synthetic_frame)
    preprocessor = build_preprocessor(SYNTHETIC_FEATURES)
    preprocessor.fit(train)

    restored = preprocessor.named_transformers_["numeric"].inverse_transform(
        preprocessor.transform(test)
    )

    assert np.allclose(restored, test[SYNTHETIC_FEATURES].to_numpy())


def test_preprocessor_drops_columns_it_was_not_given(synthetic_frame):
    preprocessor = build_preprocessor(["small"])

    scaled = preprocessor.fit_transform(synthetic_frame)

    assert scaled.shape == (len(synthetic_frame), 1)


def test_preprocessor_output_column_order_matches_the_request(synthetic_frame):
    preprocessor = build_preprocessor(["large", "small"])
    preprocessor.fit(synthetic_frame)

    assert list(preprocessor.get_feature_names_out()) == ["large", "small"]


def test_fit_transform_equals_fit_then_transform(synthetic_frame):
    train, _ = stratified_split(synthetic_frame)

    combined = build_preprocessor(SYNTHETIC_FEATURES).fit_transform(train)
    separate = build_preprocessor(SYNTHETIC_FEATURES).fit(train).transform(train)

    assert np.allclose(combined, separate)


def test_transform_before_fit_raises(synthetic_frame):
    with pytest.raises(Exception, match="not fitted"):
        build_preprocessor(SYNTHETIC_FEATURES).transform(synthetic_frame)


# --------------------------------------------------------------------------
# build_preprocessing_pipeline
# --------------------------------------------------------------------------


def test_pipeline_wraps_the_preprocessor(synthetic_frame):
    pipeline = build_preprocessing_pipeline(SYNTHETIC_FEATURES)

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["preprocess"]
    assert isinstance(pipeline.named_steps["preprocess"], ColumnTransformer)


def test_pipeline_produces_the_same_numbers_as_the_bare_preprocessor(synthetic_frame):
    train, test = stratified_split(synthetic_frame)

    pipeline = build_preprocessing_pipeline(SYNTHETIC_FEATURES).fit(train)
    preprocessor = build_preprocessor(SYNTHETIC_FEATURES).fit(train)

    assert np.allclose(pipeline.transform(test), preprocessor.transform(test))


# --------------------------------------------------------------------------
# Label encoding
# --------------------------------------------------------------------------


def test_label_encoder_maps_classes_to_contiguous_integers(synthetic_frame):
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(synthetic_frame[TARGET_COLUMN])

    assert sorted(set(encoded)) == list(range(synthetic_frame[TARGET_COLUMN].nunique()))
    assert list(encoder.classes_) == sorted(synthetic_frame[TARGET_COLUMN].unique())
    assert list(encoder.inverse_transform(encoded)) == list(synthetic_frame[TARGET_COLUMN])


# --------------------------------------------------------------------------
# The real dataset
# --------------------------------------------------------------------------


@requires_raw_dataset
def test_real_split_is_stratified_across_all_22_crops():
    train, test = stratified_split(load_data())

    train_shares = class_proportions(train)
    test_shares = class_proportions(test)

    assert set(train_shares.index) == EXPECTED_LABELS
    assert set(test_shares.index) == EXPECTED_LABELS
    # 2,200 rows, 22 crops, 20% held back -> exactly 80 train / 20 test per crop.
    assert np.allclose(train_shares.to_numpy(), test_shares.to_numpy(), atol=0.001)


@requires_raw_dataset
def test_real_split_holds_back_the_expected_number_of_rows():
    frame = load_data()
    train, test = stratified_split(frame)

    assert len(train) == 1_760
    assert len(test) == 440
    assert len(test) / len(frame) == pytest.approx(DEFAULT_TEST_SIZE)


@requires_raw_dataset
def test_real_scaled_training_features_are_standardised():
    train, test = stratified_split(load_data())
    preprocessor = build_preprocessor()

    scaled_train = preprocessor.fit_transform(train)
    scaled_test = preprocessor.transform(test)

    assert scaled_train.shape == (1_760, len(FEATURE_COLUMNS))
    assert scaled_test.shape == (440, len(FEATURE_COLUMNS))
    assert np.allclose(scaled_train.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(scaled_train.std(axis=0, ddof=0), 1.0, atol=1e-9)
    # The test set is merely *near* standardised — see the module docstring.
    assert np.allclose(scaled_test.mean(axis=0), 0.0, atol=0.15)
    assert np.allclose(scaled_test.std(axis=0, ddof=0), 1.0, atol=0.15)
