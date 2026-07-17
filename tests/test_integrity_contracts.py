from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import torch

import cfm_mesh_train as cfm
import exceedance_eval as ee
from mesh_backbone import MeshFlowNet
from mode_dispatch import gaussian_crps_numerical_check, split_distributional_prediction
from repo_integrity import audit_repository


ROOT = Path(__file__).resolve().parents[1]


def test_repository_contract_audit_passes():
    failures = [result for result in audit_repository(ROOT) if not result.passed]
    assert not failures, "\n".join(f"{result.name}: {result.detail}" for result in failures)


def test_five_cv_folds_are_disjoint_and_cover_all_years_once():
    base = datetime(1981, 5, 1)
    dates = [datetime(year, 7, 1) for year in range(1981, 2024)]
    time_values = np.array([(value - base).days for value in dates], dtype=np.float64)
    valid_indices = list(range(len(dates)))
    all_test_years = []
    for fold in range(5):
        _, _, _, train_years, val_years, test_years = cfm.build_crossval_split(
            valid_indices,
            time_values,
            val_stride=5,
            test_stride=5,
            val_offsets=((fold + 1) % 5,),
            test_offsets=(fold,),
        )
        assert not (train_years & val_years)
        assert not (train_years & test_years)
        assert not (val_years & test_years)
        all_test_years.extend(test_years)
    assert sorted(all_test_years) == list(range(1981, 2024))


def test_month_q95_uses_only_train_year_days_and_is_month_specific():
    original_size = cfm.Config.IMAGE_SIZE
    cfm.Config.IMAGE_SIZE = (2, 2)
    try:
        base = datetime(1981, 5, 1)
        dates = []
        fields = []
        for month in ee.MJJAS_MONTHS:
            for day in range(1, 21):
                dates.append(datetime(1981, month, day))
                field = np.full((2, 2), month * 100 + day, dtype=np.float32)
                field[1, 1] = 0.0
                fields.append(field)
            dates.append(datetime(1982, month, 1))
            held_out = np.full((2, 2), 10000 + month, dtype=np.float32)
            held_out[1, 1] = 0.0
            fields.append(held_out)
        shared = {
            "heat_index": np.stack(fields, axis=2),
            "time_values": np.array([(value - base).days for value in dates], dtype=np.float64),
        }
        train_mask = np.array([value.year == 1981 for value in dates], dtype=bool)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            q95 = ee.build_month_q95(
                shared,
                train_mask,
                {"hi_mean": torch.tensor(0.0), "hi_std": torch.tensor(1.0)},
            )
        base_rate = ee.build_month_base_rate(
            shared,
            train_mask,
            {"hi_mean": torch.tensor(0.0), "hi_std": torch.tensor(1.0)},
            q95,
        )
        for month in ee.MJJAS_MONTHS:
            assert month * 100 + 19 < q95[month, 0, 0] < month * 100 + 20
            assert np.isclose(base_rate[month, 0, 0], 0.05)
            assert np.isnan(q95[month, 1, 1])
        assert q95[5, 0, 0] != q95[9, 0, 0]
    finally:
        cfm.Config.IMAGE_SIZE = original_size


def test_distributional_mean_is_persistence_residual_and_sigma_is_positive():
    model = SimpleNamespace(
        module=SimpleNamespace(
            sigma_floor=0.1,
            distributional_head=True,
            predict_persistence_residual=True,
        )
    )
    raw = torch.zeros(1, 3, 2, 2, 2)
    raw[:, :, 0] = 0.25
    raw[:, :, 1] = -100.0
    persistence = torch.full((1, 1, 2, 2), 2.0)
    mean, sigma = split_distributional_prediction(model, raw, persistence)
    assert torch.allclose(mean, torch.full_like(mean, 2.25))
    assert torch.all(sigma >= 0.1)


def test_closed_form_crps_matches_numerical_quadrature():
    assert gaussian_crps_numerical_check(num_points=8, num_samples=20001) < 1e-4


def test_shared_meshflow_factory_preserves_constructor_and_runtime_config(monkeypatch):
    class CapturingModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def to(self, device):
            self.device = device
            return self

    config = SimpleNamespace(
        IMAGE_CHANNELS=2,
        NUM_SPATIAL_CONDITIONS=25,
        CONDITION_DIM=5,
        MESH_LATENT_DIM=128,
        MESH_PROCESSOR_ROUNDS=8,
        IMAGE_SIZE=(621, 1405),
        NUM_GLOBAL_CHANNELS=118,
        GLOBAL_ENCODER_DIM=64,
        DETERMINISTIC=True,
        DROPOUT_RATE=0.15,
        PREDICT_PERSISTENCE_RESIDUAL=True,
        MULTI_LEAD_TUBE=True,
        PREDICTION_LEADS=tuple(range(15, 29)),
        LEAD_TIME=21,
        TUBE_TEMPORAL_HEADS=4,
        TUBE_DECODE_CHUNK_SIZE=2,
        TUBE_LOSS_DAILY_WEIGHT=0.9,
        TUBE_LOSS_CENTER_WEIGHT=0.1,
        TUBE_LOSS_WEEKLY_WEIGHT=0.0,
        GRADIENT_LOSS_WEIGHT=0.0,
        ENABLE_EXCEEDANCE_HEAD=False,
        EXCEEDANCE_INITIAL_PROB=0.05,
        DISTRIBUTIONAL_HEAD=True,
        SIGMA_FLOOR=0.1,
        CRPS_LOSS=True,
        MSE_ANCHOR_WEIGHT=0.0,
        EXCEEDANCE_BCE_WEIGHT=0.2,
        EXCEEDANCE_COUNT_WEIGHT=0.05,
        EXCEEDANCE_POS_WEIGHT=1.0,
        EXCEEDANCE_FOCAL_GAMMA=0.0,
    )
    mesh = object()
    device = torch.device("cpu")
    monkeypatch.setattr(cfm, "MeshFlowNet", CapturingModel)

    model = cfm.build_meshflow_model(config, mesh, device)

    assert model.device == device
    assert model.kwargs == {
        "img_channels": 2,
        "spatial_cond_channels": 25,
        "condition_dim": 5,
        "latent_dim": 128,
        "hidden_dim": 256,
        "num_processor_rounds": 8,
        "mesh": mesh,
        "image_size": (621, 1405),
        "num_global_channels": 118,
        "global_encoder_dim": 64,
        "deterministic": True,
        "dropout": 0.15,
        "predict_persistence_residual": True,
        "multi_lead_tube": True,
        "prediction_leads": tuple(range(15, 29)),
        "tube_temporal_heads": 4,
        "tube_decode_chunk_size": 2,
        "tube_loss_weights": (0.9, 0.1, 0.0),
        "gradient_loss_weight": 0.0,
        "enable_exceedance_head": False,
        "exceedance_initial_logit": np.log(0.05 / 0.95),
        "distributional_head": True,
        "sigma_floor": 0.1,
    }
    assert model.crps_loss is True
    assert model.mse_anchor_weight == 0.0
    assert model.exceedance_bce_weight == 0.2
    assert model.exceedance_count_weight == 0.05
    assert model.exceedance_pos_weight == 1.0
    assert model.exceedance_focal_gamma == 0.0


def test_w34_model_parameter_budget_and_decoder_chunking_do_not_change_weights():
    common = dict(
        mesh=None,
        image_size=(8, 8),
        img_channels=2,
        spatial_cond_channels=25,
        condition_dim=5,
        latent_dim=128,
        hidden_dim=256,
        num_processor_rounds=8,
        num_global_channels=118,
        global_encoder_dim=64,
        deterministic=True,
        distributional_head=True,
        multi_lead_tube=True,
        prediction_leads=tuple(range(15, 29)),
    )
    full = MeshFlowNet(**common, tube_decode_chunk_size=0)
    chunked = MeshFlowNet(**common, tube_decode_chunk_size=2)
    full_count = sum(parameter.numel() for parameter in full.parameters())
    chunked_count = sum(parameter.numel() for parameter in chunked.parameters())
    assert full_count == chunked_count == 4_637_891
