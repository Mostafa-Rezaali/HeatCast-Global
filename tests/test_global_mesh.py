"""Global spherical mesh and Phase B memory-control contracts."""

import numpy as np
import torch

import cfm_mesh_train as cfm
from icosahedral_mesh import IcosahedralMesh
from mesh_backbone import MeshFlowNet, MeshProcessor, PeriodicLonConv2d


def global_level4_mesh():
    return IcosahedralMesh(
        refinement_level=4,
        lat_range=(-90.0, 90.0),
        lon_range=(0.0, 360.0),
        grid_lat=np.array([-90.0, 0.0, 90.0]),
        grid_lon=np.array([0.0, 1.0, 180.0, 359.0]),
        land_mask=None,
        k_grid2mesh=3,
        k_mesh2grid=4,
        global_domain=True,
    )


def test_global_level4_mesh_has_analytic_counts_xyz_features_and_no_orphans():
    mesh = global_level4_mesh()
    level = 4
    assert mesh.num_mesh_nodes == 10 * (4 ** level) + 2
    assert mesh.num_mesh_edges_undirected == 30 * (4 ** level)
    assert mesh.mesh_node_features.shape == (mesh.num_mesh_nodes, 3)
    assert np.allclose(np.linalg.norm(mesh.mesh_node_features, axis=1), 1.0, atol=1e-6)
    assert mesh.g2m_edge_attr.shape[1] == 4
    assert mesh.m2g_edge_attr.shape[1] == 4
    assert np.all(np.bincount(mesh.g2m_edge_index[0], minlength=12) == 3)
    assert np.all(np.bincount(mesh.m2g_edge_index[1], minlength=12) == 4)


def test_global_connectivity_wraps_longitude_and_handles_duplicate_poles():
    mesh = global_level4_mesh()

    def neighbors(local_grid_index):
        return set(mesh.g2m_edge_index[1, mesh.g2m_edge_index[0] == local_grid_index].tolist())

    # Equatorial 0E and 359E cells share spherical neighbors across the seam.
    assert neighbors(4) & neighbors(7)
    # All longitude labels at a pole represent one unit-vector position.
    assert len({tuple(sorted(neighbors(index))) for index in range(4)}) == 1
    assert len({tuple(sorted(neighbors(index))) for index in range(8, 12)}) == 1


def test_global_backbone_derives_xyz_edge_dimensions_and_periodic_refiner():
    mesh = global_level4_mesh()
    model = MeshFlowNet(
        img_channels=2,
        spatial_cond_channels=6,
        condition_dim=2,
        latent_dim=16,
        hidden_dim=32,
        num_processor_rounds=1,
        mesh=mesh,
        image_size=(3, 4),
        num_global_channels=0,
        deterministic=True,
        distributional_head=True,
        multi_lead_tube=False,
    )
    assert model.encoder.mesh_encoder.net[0].in_features == 3
    assert model.encoder.edge_encoder.net[0].in_features == 4
    assert model.decoder.edge_encoder.net[0].in_features == 4
    assert isinstance(model.grid_refiner[0], PeriodicLonConv2d)


def test_processor_gradient_checkpoint_matches_plain_forward_and_gradients():
    torch.manual_seed(7)
    plain = MeshProcessor(8, 4, 16, num_rounds=2, time_dim=16, gradient_checkpointing=False)
    checked = MeshProcessor(8, 4, 16, num_rounds=2, time_dim=16, gradient_checkpointing=True)
    checked.load_state_dict(plain.state_dict())
    plain.train()
    checked.train()
    edge_index = torch.tensor([[0, 1, 2, 1], [1, 2, 1, 0]], dtype=torch.long)
    edge_attr = torch.randn(4, 4)
    time_embedding = torch.randn(1, 16)
    input_plain = torch.randn(1, 3, 8, requires_grad=True)
    input_checked = input_plain.detach().clone().requires_grad_(True)
    out_plain = plain(input_plain, edge_index, edge_attr, time_embedding)
    out_checked = checked(input_checked, edge_index, edge_attr, time_embedding)
    assert torch.allclose(out_plain, out_checked, atol=1e-6, rtol=1e-6)
    out_plain.sum().backward()
    out_checked.sum().backward()
    assert torch.allclose(input_plain.grad, input_checked.grad, atol=1e-6, rtol=1e-6)


def test_gradient_accumulation_closes_full_and_partial_groups():
    boundaries = [
        index for index in range(10)
        if cfm.optimizer_step_boundary(index, num_batches=10, accumulation_steps=4)
    ]
    assert boundaries == [3, 7, 9]
    capped = [
        index for index in range(6)
        if cfm.optimizer_step_boundary(index, num_batches=10, accumulation_steps=4, max_batches=6)
    ]
    assert capped == [3, 5]
    assert cfm.Config.PRECISION == "fp32"
    assert cfm.Config.GRAD_CHECKPOINT is False
    assert cfm.Config.GRAD_ACCUM == 1
