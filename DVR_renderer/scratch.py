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
        bounding_box_generaor (nn.Module): bounding box generator
        resolution_vol (int): resolution of volume-rendered image
        neural_renderer (nn.Module): neural renderer
        fov (float): field of view
        background_rotation_range (tuple): background rotation range
         (0 - 1)
        sample_object-existance (bool): whether to sample the existance
            of objects; only used for clevr2345
        use_max_composition (bool): whether to use the max
            composition operator instead
    '''

    def __init__(self, device, z_dim=256, z_dim_bg=128, decoder=None,
                 range_u=(0, 0), range_v=(0.25, 0.25), n_ray_samples=64,
                 range_radius=(2.732, 2.732), depth_range=[0.5, 6.],
                 background_generator=None,
                 bounding_box_generator=None, resolution_vol=16,
                 neural_renderer=None,
                 fov=49.13,
                 backround_rotation_range=[0., 0.],
                 use_max_composition=False):
        super().__init__()
        self.device = device
        self.n_ray_samples = n_ray_samples
        self.range_u = range_u
        self.range_v = range_v
        self.resolution_vol = resolution_vol
        self.range_radius = range_radius
        self.depth_range = depth_range
        self.backround_rotation_range = backround_rotation_range
        self.z_dim = z_dim
        self.z_dim_bg = z_dim_bg
        # self.use_max_composition = use_max_composition
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

        def sample_z(x): return self.sample_z(x, temperature=temperature)

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

    # reserved
    def get_vis_dict(self, batch_size=32):
        vis_dict = {
            'batch_size': batch_size,
            'latent_codes': self.sample_latent_codes(batch_size),
            'camera_matrices': self.sample_random_camera(batch_size),
            'transformations': self.sample_random_transformations(batch_size),
            'bg_rotation': self.sample_random_bg_rotation(batch_size)
        }
        return vis_dict

    def sample_random_camera(self, batch_size=32, to_device=True):
        camera_mat = self.camera_matrix.repeat(batch_size, 1, 1)
        world_mat = get_random_pose(
            self.range_u, self.range_v, self.range_radius, batch_size)
        if to_device:
            world_mat = world_mat.to(self.device)
        return camera_mat, world_mat

    def sample_random_bg_rotation(self, batch_size, to_device=True):
        if self.backround_rotation_range != [0., 0.]:
            bg_r = self.backround_rotation_range
            r_random = bg_r[0] + np.random.rand() * (bg_r[1] - bg_r[0])
            R_bg = [
                torch.from_numpy(Rot.from_euler(
                    'z', r_random * 2 * np.pi).as_dcm()
                                 ) for i in range(batch_size)]
            R_bg = torch.stack(R_bg, dim=0).reshape(
                batch_size, 3, 3).float()
        else:
            R_bg = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).float()
        if to_device:
            R_bg = R_bg.to(self.device)
        return R_bg

    def sample_random_transformations(self, batch_size=32, to_device=True):
        device = self.device
        s, t, R = self.bounding_box_generator(batch_size)
        if to_device:
            s, t, R = s.to(device), t.to(device), R.to(device)
        return s, t, R

    def add_noise_to_interval(self, di):
        di_mid = .5 * (di[..., 1:] + di[..., :-1])
        di_high = torch.cat([di_mid, di[..., -1:]], dim=-1)
        di_low = torch.cat([di[..., :1], di_mid], dim=-1)
        noise = torch.rand_like(di_low)
        ti = di_low + (di_high - di_low) * noise
        return ti

    def transform_points_to_box(self, p, transformations, box_idx=0,
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
                              transformations, i):
        batch_size = pixels_world.shape[0]
        n_steps = di.shape[-1]

        pixels_world_i = self.transform_points_to_box(
            pixels_world, transformations, i)
        camera_world_i = self.transform_points_to_box(
            camera_world, transformations, i)
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
            # if self.use_max_composition:
            #     bs, rs, ns = sigma.shape[1:]
            #     sigma_sum, ind = torch.max(sigma, dim=0)
            #     feat_weighted = feat[ind, torch.arange(bs).reshape(-1, 1, 1),
            #                          torch.arange(rs).reshape(
            #                              1, -1, 1), torch.arange(ns).reshape(
            #         1, 1, -1)]
            # else:
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
        z_shape_obj, z_app_obj, z_shape_bg, z_app_bg = latent_codes
        camera_mat, world_mat = camera_matrices
        batch_size = z_shape_obj.shape[0]
        n_boxes = z_shape_obj.shape[1]
        assert (not (not_render_background and only_render_background))

        # Arrange Pixels
        pixels = arange_pixels((res, res), batch_size,
                               invert_y_axis=False)[1].to(device)
        pixels[..., -1] *= -1.
        # Project to 3D world
        pixels_world = image_points_to_world(
            pixels, camera_mat,
            world_mat)
        camera_world = origin_to_world(
            n_points, camera_mat,
            world_mat)

        ray_vector = pixels_world - camera_world
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
                point_pos_wc_i, ray_direction_wc_i = self.get_evaluation_points(pixels_world, camera_world, di,
                                                                                transformations, obj_i)
                logging.debug(f"point pos wc i shape = {point_pos_wc_i.shape},"
                              f"ray dir wc i shape = {ray_direction_wc_i.shape}")
                feature_i, density_i = self.decoder(point_pos_wc_i,
                                                    ray_direction_wc_i,
                                                    z_shape_obj[:, obj_i],
                                                    z_app_obj[:, obj_i])  # TODO: check what this outputs
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
                p_bg, r_bg = self.get_evaluation_points_bg(pixels_world, camera_world, di, bg_rotation)
                feature_i, density_i = self.background_generator(p_bg,
                                                                 r_bg,
                                                                 z_shape_bg,
                                                                 z_app_bg)
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

    # # reserved for deterministic
    # def get_camera(self, val_u=0.5, val_v=0.5, val_r=0.5, batch_size=32,
    #                to_device=True):
    #     camera_mat = self.camera_matrix.repeat(batch_size, 1, 1)
    #     world_mat = get_camera_pose(
    #         self.range_u, self.range_v, self.range_radius, val_u, val_v,
    #         val_r, batch_size=batch_size)
    #     if to_device:
    #         world_mat = world_mat.to(self.device)
    #     return camera_mat, world_mat
    #
    # # reserved for deterministic
    # def get_bg_rotation(self, val, batch_size=32, to_device=True):
    #     if self.backround_rotation_range != [0., 0.]:
    #         bg_r = self.backround_rotation_range
    #         r_val = bg_r[0] + val * (bg_r[1] - bg_r[0])
    #         r = torch.from_numpy(
    #             Rot.from_euler('z', r_val * 2 * np.pi).as_dcm()
    #         ).reshape(1, 3, 3).repeat(batch_size, 1, 1).float()
    #     else:
    #         r = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).float()
    #     if to_device:
    #         r = r.to(self.device)
    #     return r
    #
    # # reserved for deterministic
    # def get_transformations(self, val_s=[[0.5, 0.5, 0.5]],
    #                         val_t=[[0.5, 0.5, 0.5]], val_r=[0.5],
    #                         batch_size=32, to_device=True):
    #     device = self.device
    #     s = self.bounding_box_generator.get_scale(
    #         batch_size=batch_size, val=val_s)
    #     t = self.bounding_box_generator.get_translation(
    #         batch_size=batch_size, val=val_t)
    #     R = self.bounding_box_generator.get_rotation(
    #         batch_size=batch_size, val=val_r)
    #
    #     if to_device:
    #         s, t, R = s.to(device), t.to(device), R.to(device)
    #     return s, t, R
    #
    # # reserved for deterministic
    # def get_rotation(self, val_r, batch_size=32, to_device=True):
    #     device = self.device
    #     R = self.bounding_box_generator.get_rotation(batch_size=batch_size, val=val_r)
    #     if to_device:
    #         R = R.to(device)
    #     return R
