import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rot

from im2scene.camera import get_camera_mat, get_random_pose
from im2scene.common import (
    arange_pixels, image_points_to_world, origin_to_world
)


class VolumeSampler(nn.Module):

    def __init__(self, volume, transformation_mat, transfer_function):
        self.volume = volume
        self.tf = transfer_function
        self.transformation_mat = transformation_mat

    def forward(self, sample_positions):
        # TODO
        pass


# noinspection DuplicatedCode
class Generator(nn.Module):
    ''' GIRAFFE Generator Class.

    Args:
        device (pytorch device): pytorch device
        z_dim (int): dimension of latent code z
        z_dim_bg (int): dimension of background latent code z_bg
        decoder (nn.Module): decoder network
        range_u (tuple): rotation range (0 - 1)
        range_v (tuple): elevation range (0 - 1)
        n_ray_samples (int): number of samples per ray
        range_radius(tuple): radius range
        depth_range (tuple): near and far depth plane
        background_generator (nn.Module): background generator
        bounding_box_generator (nn.Module): bounding box generator
        resolution_vol (int): resolution of volume-rendered image
        neural_renderer (nn.Module): neural renderer
        fov (float): field of view
        background_rotation_range (tuple): background rotation range
         (0 - 1)
    '''

    def __init__(self, device, z_dim=256, z_dim_bg=128, decoder=None,
                 range_u=(0, 0), range_v=(0.25, 0.25), n_ray_samples=64,
                 range_radius=(2.732, 2.732), depth_range=[0.5, 6.],
                 background_generator=None,
                 bounding_box_generator=None, resolution_vol=16,
                 neural_renderer=None,
                 fov=49.13,
                 background_rotation_range=[0., 0.]):
        super().__init__()
        self.device = device
        self.n_ray_samples = n_ray_samples
        self.range_u = range_u
        self.range_v = range_v
        self.resolution_vol = resolution_vol
        self.range_radius = range_radius
        self.depth_range = depth_range
        self.background_rotation_range = background_rotation_range
        self.z_dim = z_dim
        self.z_dim_bg = z_dim_bg
        self.camera_matrix = get_camera_mat(fov=fov).to(device)
        self.decoder = decoder.to(device)
        self.background_generator = background_generator.to(device)
        self.bounding_box_generator = bounding_box_generator.to(device)
        self.neural_renderer = None if neural_renderer is None else neural_renderer.to(device)

    def forward(self, batch_size=32, mode="training",
                not_render_background=False,
                only_render_background=False):
        latent_codes = self.sample_latent_codes(batch_size)
        camera_matrices = self.sample_random_camera(batch_size)
        logging.debug(f"\ncamera mat shape = {camera_matrices[0].shape}"
                      f"\nworld mat shape = {camera_matrices[1].shape}")
        transformations = self.sample_random_transformations(batch_size)
        sizes, translations, rotations = transformations
        logging.debug(f"\nsizes shape = {sizes.shape}"
                      f"\ntranslations shape = {translations.shape}"
                      f"\nrotations shape = {rotations.shape}")
        bg_rotation = self.sample_random_bg_rotation(batch_size)
        rgb_v = self.volume_render_image(
            latent_codes, camera_matrices, transformations, bg_rotation,
            mode=mode, not_render_background=not_render_background,
            only_render_background=only_render_background)

        if self.neural_renderer is not None:
            rgb = self.neural_renderer(rgb_v)
        else:
            rgb = rgb_v
        return rgb

    def get_n_boxes(self):
        n_boxes = self.bounding_box_generator.n_boxes
        return n_boxes

    def sample_latent_codes(self, batch_size=32, temperature=1.):
        z_dim, z_dim_bg = self.z_dim, self.z_dim_bg

        n_boxes = self.get_n_boxes()

        def sample_z(x):
            return self.sample_z(x, temperature=temperature)

        z_shape_obj = sample_z((batch_size, n_boxes, z_dim))
        z_app_obj = sample_z((batch_size, n_boxes, z_dim))
        z_shape_bg = sample_z((batch_size, z_dim_bg))
        z_app_bg = sample_z((batch_size, z_dim_bg))

        return z_shape_obj, z_app_obj, z_shape_bg, z_app_bg

    def sample_z(self, size, to_device=True, temperature=1.):
        z = torch.randn(*size) * temperature
        if to_device:
            z = z.to(self.device)
        return z

    def sample_random_camera(self, batch_size=32, to_device=True):
        camera_mat = self.camera_matrix.repeat(batch_size, 1, 1)
        world_mat = get_random_pose(
            self.range_u, self.range_v, self.range_radius, batch_size)
        if to_device:
            world_mat = world_mat.to(self.device)
        return camera_mat, world_mat

    def sample_random_bg_rotation(self, batch_size, to_device=True):
        if self.background_rotation_range != [0., 0.]:
            bg_r = self.background_rotation_range
            r_random = bg_r[0] + np.random.rand() * (bg_r[1] - bg_r[0])
            bg_rotations = [torch.from_numpy(Rot.from_euler('z', r_random * 2 * np.pi).as_dcm())
                            for _ in range(batch_size)]
            bg_rotations = torch.stack(bg_rotations, dim=0).reshape(
                batch_size, 3, 3).float()
        else:
            bg_rotations = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).float()
        if to_device:
            bg_rotations = bg_rotations.to(self.device)
        return bg_rotations

    def sample_random_transformations(self, batch_size=32, to_device=True):
        device = self.device
        sizes, translations, rotations = self.bounding_box_generator(batch_size)
        if to_device:
            sizes, translations, rotations = sizes.to(device), translations.to(device), rotations.to(device)
        return sizes, translations, rotations

    def add_noise_to_interval(self, di):
        di_mid = .5 * (di[..., 1:] + di[..., :-1])
        di_high = torch.cat([di_mid, di[..., -1:]], dim=-1)
        di_low = torch.cat([di[..., :1], di_mid], dim=-1)
        noise = torch.rand_like(di_low)
        ti = di_low + (di_high - di_low) * noise
        return ti

    def transform_points_to_box(self, p, transformations, box_idx,
                                scale_factor=1.):
        bb_s, bb_t, bb_R = transformations
        p_box = (bb_R[:, box_idx] @ (p - bb_t[:, box_idx].unsqueeze(1)).permute(0, 2, 1)).permute(0, 2, 1) / \
                bb_s[:, box_idx].unsqueeze(1) * scale_factor
        return p_box

    def get_evaluation_points_bg(self, pixels_world, camera_world, di,
                                 rotation_matrix):
        batch_size = pixels_world.shape[0]
        n_steps = di.shape[-1]

        camera_world = (rotation_matrix @
                        camera_world.permute(0, 2, 1)).permute(0, 2, 1)
        pixels_world = (rotation_matrix @
                        pixels_world.permute(0, 2, 1)).permute(0, 2, 1)
        ray_world = pixels_world - camera_world

        p = camera_world.unsqueeze(-2).contiguous() + \
            di.unsqueeze(-1).contiguous() * \
            ray_world.unsqueeze(-2).contiguous()
        r = ray_world.unsqueeze(-2).repeat(1, 1, n_steps, 1)
        assert (p.shape == r.shape)
        p = p.reshape(batch_size, -1, 3)
        r = r.reshape(batch_size, -1, 3)
        return p, r

    def get_evaluation_points(self, pixels_world, camera_world, di,
                              transformations, box_idx):
        batch_size = pixels_world.shape[0]
        n_steps = di.shape[-1]

        pixels_world_i = self.transform_points_to_box(
            pixels_world, transformations, box_idx)
        camera_world_i = self.transform_points_to_box(
            camera_world, transformations, box_idx)
        ray_i = pixels_world_i - camera_world_i

        p_i = camera_world_i.unsqueeze(-2).contiguous() + \
              di.unsqueeze(-1).contiguous() * ray_i.unsqueeze(-2).contiguous()
        ray_i = ray_i.unsqueeze(-2).repeat(1, 1, n_steps, 1)
        assert (p_i.shape == ray_i.shape)
        logging.debug(f"p_i and ray_i shape = {p_i.shape}")
        p_i = p_i.reshape(batch_size, -1, 3)
        ray_i = ray_i.reshape(batch_size, -1, 3)

        return p_i, ray_i

    def composite_function(self, sigma, feat):
        n_boxes = sigma.shape[0]
        if n_boxes > 1:
            denom_sigma = torch.sum(sigma, dim=0, keepdim=True)
            denom_sigma[denom_sigma == 0] = 1e-4
            w_sigma = sigma / denom_sigma
            sigma_sum = torch.sum(sigma, dim=0)
            feat_weighted = (feat * w_sigma.unsqueeze(-1)).sum(0)
        else:
            sigma_sum = sigma.squeeze(0)
            feat_weighted = feat.squeeze(0)
        return sigma_sum, feat_weighted

    def calc_volume_weights(self, z_vals, ray_vector, sigma, last_dist=1e10):
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.ones_like(
            z_vals[..., :1]) * last_dist], dim=-1)
        dists = dists * torch.norm(ray_vector, dim=-1, keepdim=True)
        alpha = 1. - torch.exp(-F.relu(sigma) * dists)
        weights = alpha * \
                  torch.cumprod(torch.cat([
                      torch.ones_like(alpha[:, :, :1]),
                      (1. - alpha + 1e-10), ], dim=-1), dim=-1)[..., :-1]
        return weights

    def volume_render_image(self, latent_codes, camera_matrices,
                            transformations, bg_rotation, mode='training',
                            not_render_background=False,
                            only_render_background=False):
        res = self.resolution_vol
        device = self.device
        n_steps = self.n_ray_samples
        depth_range = self.depth_range
        n_points = res * res
        obj_shape_latent, obj_appearance_latent, bg_shape_latent, bg_appearance_latent = latent_codes
        camera_mat, world_mat = camera_matrices
        batch_size = obj_shape_latent.shape[0]
        n_boxes = obj_shape_latent.shape[1]
        assert (not (not_render_background and only_render_background))

        # Arrange Pixels
        pixels = arange_pixels((res, res), batch_size,
                               invert_y_axis=False)[1].to(device)
        pixels[..., -1] *= -1.
        # Project to 3D world
        pixel_pos_wc = image_points_to_world(
            pixels, camera_mat,
            world_mat)
        camera_pos_wc = origin_to_world(
            n_points, camera_mat,
            world_mat)

        logging.debug(f"pixel_pos_wc shape = {pixel_pos_wc.shape}")
        logging.debug(f"camera_pos_wc shape = {camera_pos_wc.shape}")
        logging.debug(f"camera_pos_wc = {camera_pos_wc}")

        ray_vector = pixel_pos_wc - camera_pos_wc
        # batch_size x n_points x n_steps
        di = depth_range[0] + \
             torch.linspace(0., 1., steps=n_steps).reshape(1, 1, -1) * (
                     depth_range[1] - depth_range[0])
        di = di.repeat(batch_size, n_points, 1).to(device)
        logging.debug(f"di shape = {di.shape}")
        if mode == 'training':
            di = self.add_noise_to_interval(di)

        features, density = [], []
        n_objects = n_boxes if not_render_background else n_boxes + 1
        if only_render_background:
            n_objects = 1
            n_boxes = 0
        for obj_i in range(n_objects):
            if obj_i < n_boxes:  # Object
                # transform points w.r.t each object
                point_pos_wc_i, ray_direction_wc_i = self.get_evaluation_points(pixel_pos_wc, camera_pos_wc, di,
                                                                                transformations, obj_i)
                logging.debug(f"point pos wc i shape = {point_pos_wc_i.shape},"
                              f"ray dir wc i shape = {ray_direction_wc_i.shape}")
                feature_i, density_i = self.decoder(point_pos_wc_i,
                                                    ray_direction_wc_i,
                                                    obj_shape_latent[:, obj_i],
                                                    obj_appearance_latent[:,
                                                    obj_i])  # TODO: replace this with a volume sampler
                logging.debug(f"feature shape = {feature_i.shape}, density shape = {density_i.shape}")
                if mode == 'training':
                    # As done in NeRF, add noise during training
                    density_i += torch.randn_like(density_i)

                # Mask out values outside
                padd = 0.1
                mask_box = torch.all(point_pos_wc_i <= 1. + padd, dim=-1) & \
                           torch.all(point_pos_wc_i >= -1. - padd, dim=-1)
                density_i[mask_box == 0] = 0.

                # Reshape
                density_i = density_i.reshape(batch_size, n_points, n_steps)
                feature_i = feature_i.reshape(batch_size, n_points, n_steps, -1)
            else:  # Background
                p_bg, r_bg = self.get_evaluation_points_bg(pixel_pos_wc, camera_pos_wc, di, bg_rotation)
                feature_i, density_i = self.background_generator(p_bg,
                                                                 r_bg,
                                                                 bg_shape_latent,
                                                                 bg_appearance_latent)
                density_i = density_i.reshape(batch_size, n_points, n_steps)
                feature_i = feature_i.reshape(batch_size, n_points, n_steps, -1)

                if mode == 'training':
                    # As done in NeRF, add noise during training
                    density_i += torch.randn_like(density_i)

            features.append(feature_i)
            density.append(density_i)

        density = F.relu(torch.stack(density, dim=0))
        features = torch.stack(features, dim=0)
        # Composite
        sigma_sum, feat_weighted = self.composite_function(density, features)
        # Get Volume Weights
        weights = self.calc_volume_weights(di, ray_vector, sigma_sum)
        feat_map = torch.sum(weights.unsqueeze(-1) * feat_weighted, dim=-2)
        # Reformat output
        feat_map = feat_map.permute(0, 2, 1).reshape(batch_size, -1, res, res)  # B x feat x h x w
        feat_map = feat_map.permute(0, 1, 3, 2)  # new to flip x/y
        return feat_map
