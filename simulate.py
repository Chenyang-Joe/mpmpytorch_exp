from typing import *

import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import trimesh
from typing import List
from skimage import measure

from mpm_pytorch import MPMSolver, set_boundary_conditions, get_constitutive

# def get_cube(
#         center: List[float], 
#         size: List[float], 
#         num: int, 
#         add_noise: bool=False, 
#         device: torch.device=torch.device("cuda")
#     ) -> Tensor:
#     start = torch.tensor(center) - torch.tensor(size) / 2
#     end = torch.tensor(center) + torch.tensor(size) / 2
#     # Generate a cube
#     x = torch.linspace(start[0], end[0], num)
#     y = torch.linspace(start[1], end[1], num)
#     z = torch.linspace(start[2], end[2], num)
#     cube = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1).view(-1, 3)
#     if add_noise:
#         # Add noise to the cube
#         noisy_cube = start + torch.rand_like(cube) * (end - start)
#         cube = torch.cat([cube, noisy_cube], dim=0)
#     return cube.to(device)

def get_obj_model(
        file_path: str, 
        num: int = 1000, 
        add_noise: bool = False, 
        device: torch.device = torch.device("cuda")
    ) -> Tensor:
    """
    Load an OBJ file, sample point cloud from the mesh surface,
    and normalize it into the [0, 1]^3 cube without distorting proportions.
    
    Args:
        file_path (str): Path to the OBJ file.
        num (int): Number of points to sample.
        add_noise (bool): Whether to add noise.
        device (torch.device): Device to load tensor on.
    
    Returns:
        Tensor: [N, 3] normalized point cloud.
    """
    mesh = trimesh.load(file_path, force='mesh')

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("The loaded file does not contain a valid mesh.")

    # Sample points from the surface
    points, _ = trimesh.sample.sample_surface(mesh, num)
    points = torch.tensor(points, dtype=torch.float32)

    if add_noise:
        # Add small uniform noise
        scale = (points.max(0).values - points.min(0).values).max() * 0.01
        noise = (torch.rand_like(points) - 0.5) * 2 * scale
        points = points + noise

    # Normalize into [0, 1]^3 without changing proportions
    min_coord = points.min(0).values
    points = points - min_coord  # shift to origin

    max_extent = points.max(0).values.max()  # longest axis length
    points = points / max_extent  # uniform scale to [0, 1] box

    return points.to(device)


def export_final_mesh_from_points(
    x: torch.Tensor,
    export_path: str,
    resolution: int = 128
):
    """
    Convert final MPM particle positions to a mesh using Marching Cubes, and export as .obj.

    Args:
        x (torch.Tensor): Final particle positions, shape [N, 3].
        export_path (str): Path to export the mesh (.obj).
        resolution (int): Voxel grid resolution.
    """
    points = x.detach().cpu().numpy()
    
    # Normalize to [0, 1]^3 without changing proportions
    min_pt = points.min(axis=0)
    points -= min_pt
    max_extent = points.max()
    points /= max_extent

    # Create voxel grid
    voxels = np.zeros((resolution, resolution, resolution), dtype=bool)
    indices = (points * (resolution - 1)).astype(int)
    voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    # Surface reconstruction via Marching Cubes
    verts, faces, normals, _ = measure.marching_cubes(voxels, level=0.5)

    # Unnormalize verts back to original scale
    verts = verts / resolution * max_extent + min_pt

    # Export mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(export_path)
    print(f"Mesh exported to {export_path}")



def visualize_frames(
    frames: List[np.ndarray], 
    export_path: str, 
    center: List[float] = [0.5, 0.5, 0.5],
    size: List[float] = [2.0, 2.0, 2.0],
    c: str = 'blue',
    s: float = 20,
    fps: int = 30,
): 
    xlim = [center[0] - size[0] / 2, center[0] + size[0] / 2]
    ylim = [center[1] - size[1] / 2, center[1] + size[1] / 2]
    zlim = [center[2] - size[2] / 2, center[2] + size[2] / 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter([], [], [], s=s)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    def update(frame):
        ax.cla()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        scat = ax.scatter(frames[frame][:, 0], frames[frame][:, 1], frames[frame][:, 2], s=s, c=c)
        ax.set_title(f'Frame {frame}')
        return scat
    ani = FuncAnimation(fig, update, frames=len(frames), blit=False)
    ani.save(export_path, writer='pillow', fps=fps)
    plt.close()

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    print(f'Start simulation with config: {args.config}')

    # Load config
    cfg = OmegaConf.load(args.config)
    material_params = cfg.material
    sim_params = cfg.sim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    export_path = os.path.join(cfg.output_dir, cfg.tag + ".gif")

    # Create a cube for simulation
    # particles = get_cube(
    #     center=[0.5, 0.5, 0.5], 
    #     size=[0.5, 0.5, 0.5], 
    #     num=10, 
    #     add_noise=True,
    #     device=device
    # )

    particles = get_obj_model("models/bone.obj", num = 10000 ,add_noise=False,         device=device)
    print(particles.shape)  # should be [2000, 3] or [4000, 3] if noise is added like duplication

    n_particles = particles.shape[0]

    # Initialize MPM solver
    mpm_solver = MPMSolver(
        particles, 
        enable_train=False,
        device=device
    )
    set_boundary_conditions(mpm_solver, sim_params.boundary_conditions)
    # Initialize Constitutive models
    elasticity = get_constitutive(material_params.elasticity, device=device)
    plasticity = get_constitutive(material_params.plasticity, device=device)

    # Initialize particle states
    x = particles
    v = torch.stack([torch.tensor(sim_params.initial_velocity, device=device) for _ in range(n_particles)])
    C = torch.zeros((n_particles, 3, 3), device=device)
    F = torch.eye(3, device=device).unsqueeze(0).repeat(n_particles, 1, 1)

    # Run simulation
    frames = []
    for frame in tqdm(range(sim_params.num_frames), desc='Simulating', leave=False):
        frames.append(x.cpu().numpy())
        for step in tqdm(range(sim_params.steps_per_frame), desc='Step', leave=False):
            # Update stress
            stress = elasticity(F)
            # Particle to grid, grid update, grid to particle
            x, v, C, F = mpm_solver(x, v, C, F, stress)
            print()
            # Plasticity correction
            F = plasticity(F)
    
    # Visualize
    print(f'Rendering to {export_path}...')
    visualize_frames(
        frames, 
        export_path=export_path, 
        size=[1, 1, 1], 
        c=material_params.color
    )

    for i in range(10,100,10):
        i = int(i)
        final_obj_path = os.path.join(cfg.output_dir, cfg.tag + f"_{i}.obj")
        export_final_mesh_from_points(x, export_path=final_obj_path, resolution=i)
