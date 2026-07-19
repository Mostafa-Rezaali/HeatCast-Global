"""Fold-leakage, harmonic target, global input, and vector-parser tests."""

from datetime import date, timedelta

import numpy as np
import torch

from build_driver_tables import parse_rmm_components_file, parse_rmm_file
from data_pipeline.build_cache import CACHE_CHANNELS
from global_dataset import (
    GLOBAL_INPUT_CHANNELS,
    VECTOR_INPUT_CHANNELS,
    assemble_global_tensors,
    build_model_condition_vectors,
    identity_preprocessor,
    normalize_condition_vectors,
)
from global_targets import fit_fold_preprocessor_from_arrays, harmonic_features


def harmonic_fixture():
    dates = []
    for year in (2000, 2001, 2002):
        current = date(year, 1, 1)
        while current.year == year:
            dates.append(current)
            current += timedelta(days=1)
    doy = np.array([value.timetuple().tm_yday for value in dates])
    design = harmonic_features(doy, 4)
    coefficients = np.array([10.0, 2.0, -1.0, 0.5, 0.25, -0.2, 0.1, 0.05, -0.03])
    signal = design @ coefficients
    spatial = np.array([[0.0, 1.0], [2.0, 3.0]])
    tmax = signal[:, None, None] + spatial[None]
    predictor = 0.5 * signal[:, None, None] - spatial[None]
    data = np.stack((tmax, predictor), axis=-1).astype(np.float32)
    return dates, data


def test_four_harmonic_climatology_anomaly_round_trip_and_sidecar(tmp_path):
    dates, data = harmonic_fixture()
    processor = fit_fold_preprocessor_from_arrays(
        dates,
        data,
        ("tmax", "predictor"),
        train_years=(2000, 2001),
        anomaly_channels=("tmax", "predictor"),
    )
    assert processor.coefficients["tmax"].shape[0] == 9
    index = next(index for index, value in enumerate(dates) if value == date(2002, 7, 15))
    normalized = processor.transform("tmax", dates[index], data[index, ..., 0])
    restored = processor.inverse("tmax", dates[index], normalized)
    assert np.allclose(restored, data[index, ..., 0], atol=1e-5)
    path = tmp_path / "fold0_climatology.npz"
    processor.save(path)
    loaded = type(processor).load(path)
    assert loaded.train_years == (2000, 2001)
    assert np.allclose(loaded.coefficients["tmax"], processor.coefficients["tmax"])


def test_poisoning_test_year_cannot_change_climatology_or_normalization():
    dates, clean = harmonic_fixture()
    poisoned = clean.copy()
    test_index = next(index for index, value in enumerate(dates) if value == date(2002, 8, 1))
    poisoned[test_index, ..., :] = 1.0e9
    kwargs = dict(
        dates=dates,
        channels=("tmax", "predictor"),
        train_years=(2000, 2001),
        anomaly_channels=("tmax", "predictor"),
    )
    baseline = fit_fold_preprocessor_from_arrays(data=clean, **kwargs)
    attacked = fit_fold_preprocessor_from_arrays(data=poisoned, **kwargs)
    for channel in kwargs["channels"]:
        assert np.array_equal(baseline.coefficients[channel], attacked.coefficients[channel])
        assert np.array_equal(baseline.means[channel], attacked.means[channel])
        assert np.array_equal(baseline.stds[channel], attacked.stds[channel])


def test_global_input_assembler_has_authoritative_26_channels():
    height, width = 4, 8
    context = np.zeros((3, height, width, len(CACHE_CHANNELS)), dtype=np.float32)
    index = {name: position for position, name in enumerate(CACHE_CHANNELS)}
    context[..., index["sst_valid"]] = 1.0
    context[..., index["land_mask"]] = 1.0
    context[..., index["orography"]] = np.arange(height, dtype=np.float32)[None, :, None]
    target = np.zeros((14, height, width), dtype=np.float32)
    assembled = assemble_global_tensors(
        context,
        target,
        (20000501, 20000430, 20000429),
        tuple(20000516 + value for value in range(14)),
        np.linspace(90.0, -90.0, height),
        np.arange(width) * 45.0,
        np.arange(8, dtype=np.float32),
        identity_preprocessor((height, width)),
    )
    model_input = torch.cat((
        assembled["x_t"], assembled["x_tm1"], assembled["x_tm2"], assembled["spatial_c"]
    ))
    assert len(GLOBAL_INPUT_CHANNELS) == 26
    assert model_input.shape == (26, height, width)
    assert assembled["spatial_c"].shape == (23, height, width)
    assert assembled["target"].shape == (14, height, width)
    assert assembled["vector"].shape == (len(VECTOR_INPUT_CHANNELS),)
    assert assembled["global_fields"].shape == (0, height, width)


def test_existing_rmm_parser_is_reused_for_model_components(tmp_path):
    rmm = tmp_path / "rmm.txt"
    rmm.write_text(
        "year month day RMM1 RMM2 phase amplitude method\n"
        "2000 5 1 1.25 -0.50 3 1.35 final\n"
        "2000 5 2 0.75 0.25 4 0.79 final\n",
        encoding="utf-8",
    )
    assert parse_rmm_file(rmm)[20000501] == (3, 1.35)
    assert parse_rmm_components_file(rmm)[20000501] == (1.25, -0.5, 1.35)
    base = np.arange(10, dtype=np.float32).reshape(2, 5)
    vectors = build_model_condition_vectors((20000501, 20000502), base, rmm)
    assert vectors.shape == (2, 8)
    normalized, mean, std = normalize_condition_vectors(vectors, (0,))
    assert normalized.shape == vectors.shape
    assert mean.shape == std.shape == (8,)
