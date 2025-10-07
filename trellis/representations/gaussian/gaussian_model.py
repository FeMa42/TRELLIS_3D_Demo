import torch
import numpy as np
from plyfile import PlyData, PlyElement
from .general_utils import inverse_sigmoid, strip_symmetric, build_scaling_rotation
import utils3d


class Gaussian:
    def __init__(
            self, 
            aabb : list,
            sh_degree : int = 0,
            mininum_kernel_size : float = 0.0,
            scaling_bias : float = 0.01,
            opacity_bias : float = 0.1,
            scaling_activation : str = "exp",
            device='cuda'
        ):
        self.init_params = {
            'aabb': aabb,
            'sh_degree': sh_degree,
            'mininum_kernel_size': mininum_kernel_size,
            'scaling_bias': scaling_bias,
            'opacity_bias': opacity_bias,
            'scaling_activation': scaling_activation,
        }
        
        self.sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self.mininum_kernel_size = mininum_kernel_size 
        self.scaling_bias = scaling_bias
        self.opacity_bias = opacity_bias
        self.scaling_activation_type = scaling_activation
        self.device = device
        self.aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
        self.setup_functions()

        self._xyz = None
        self._features_dc = None
        self._features_rest = None
        self._scaling = None
        self._rotation = None
        self._opacity = None

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        if self.scaling_activation_type == "exp":
            self.scaling_activation = torch.exp
            self.inverse_scaling_activation = torch.log
        elif self.scaling_activation_type == "softplus":
            self.scaling_activation = torch.nn.functional.softplus
            self.inverse_scaling_activation = lambda x: x + torch.log(-torch.expm1(-x))

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        
        self.scale_bias = self.inverse_scaling_activation(torch.tensor(self.scaling_bias)).cuda()
        self.rots_bias = torch.zeros((4)).cuda()
        self.rots_bias[0] = 1
        self.opacity_bias = self.inverse_opacity_activation(torch.tensor(self.opacity_bias)).cuda()

    @property
    def get_scaling(self):
        scales = self.scaling_activation(self._scaling + self.scale_bias)
        scales = torch.square(scales) + self.mininum_kernel_size ** 2
        scales = torch.sqrt(scales)
        return scales
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation + self.rots_bias[None, :])
    
    @property
    def get_xyz(self):
        return self._xyz * self.aabb[None, 3:] + self.aabb[None, :3]
    
    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=2) if self._features_rest is not None else self._features_dc
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity + self.opacity_bias)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation + self.rots_bias[None, :])
    
    def from_scaling(self, scales):
        scales = torch.sqrt(torch.square(scales) - self.mininum_kernel_size ** 2)
        self._scaling = self.inverse_scaling_activation(scales) - self.scale_bias
        
    def from_rotation(self, rots):
        self._rotation = rots - self.rots_bias[None, :]
    
    def from_xyz(self, xyz):
        self._xyz = (xyz - self.aabb[None, :3]) / self.aabb[None, 3:]
        
    def from_features(self, features):
        self._features_dc = features
        
    def from_opacity(self, opacities):
        self._opacity = self.inverse_opacity_activation(opacities) - self.opacity_bias

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
        
    def save_ply(self, path, transform=[[1, 0, 0], [0, 0, -1], [0, 1, 0]]):
        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = inverse_sigmoid(self.get_opacity).detach().cpu().numpy()
        scale = torch.log(self.get_scaling).detach().cpu().numpy()
        rotation = (self._rotation + self.rots_bias[None, :]).detach().cpu().numpy()
        
        if transform is not None:
            transform = np.array(transform)
            xyz = np.matmul(xyz, transform.T)
            rotation = utils3d.numpy.quaternion_to_matrix(rotation)
            rotation = np.matmul(transform, rotation)
            rotation = utils3d.numpy.matrix_to_quaternion(rotation)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        # Save in ASCII format instead of binary for better compatibility
        PlyData([el], text=True).write(path)

    def load_ply(self, path, transform=[[1, 0, 0], [0, 0, -1], [0, 1, 0]]):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        if self.sh_degree > 0:
            extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
            extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
            assert len(extra_f_names)==3*(self.sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        if transform is not None:
            transform = np.array(transform)
            xyz = np.matmul(xyz, transform)
            rots = utils3d.numpy.quaternion_to_matrix(rots)
            rots = np.matmul(rots, transform)
            rots = utils3d.numpy.matrix_to_quaternion(rots)
            
        # convert to actual gaussian attributes
        xyz = torch.tensor(xyz, dtype=torch.float, device=self.device)
        features_dc = torch.tensor(features_dc, dtype=torch.float, device=self.device).transpose(1, 2).contiguous()
        if self.sh_degree > 0:
            features_extra = torch.tensor(features_extra, dtype=torch.float, device=self.device).transpose(1, 2).contiguous()
        opacities = torch.sigmoid(torch.tensor(opacities, dtype=torch.float, device=self.device))
        scales = torch.exp(torch.tensor(scales, dtype=torch.float, device=self.device))
        rots = torch.tensor(rots, dtype=torch.float, device=self.device)
        
        # convert to _hidden attributes
        self._xyz = (xyz - self.aabb[None, :3]) / self.aabb[None, 3:]
        self._features_dc = features_dc
        if self.sh_degree > 0:
            self._features_rest = features_extra
        else:
            self._features_rest = None
        self._opacity = self.inverse_opacity_activation(opacities) - self.opacity_bias
        self._scaling = self.inverse_scaling_activation(torch.sqrt(torch.square(scales) - self.mininum_kernel_size ** 2)) - self.scale_bias
        self._rotation = rots - self.rots_bias[None, :]

    def save_splat(self, path, transform=[[1, 0, 0], [0, 0, -1], [0, 1, 0]], max_splats=50000):
        """
        Save Gaussian splat in standard .splat format.
        
        The .splat format stores each Gaussian as 14 consecutive float32 values:
        [x, y, z, scale_x, scale_y, scale_z, r, g, b, rot_w, rot_x, rot_y, rot_z, opacity]
        
        Key insights from TRELLIS:
        - Positions are in normalized [-0.5, 0.5] space, need to be scaled
        - Features are raw SH DC coefficients, need proper SH->RGB conversion
        - Scales are already processed (get_scaling applies activations)
        - Rotations are normalized quaternions
        - Opacities are already processed (get_opacity applies sigmoid)
        """
        
        # Get processed Gaussian parameters
        xyz = self.get_xyz.detach().cpu().numpy()           # Already denormalized positions
        scales = self.get_scaling.detach().cpu().numpy()    # Already processed scales  
        rotations = self.get_rotation.detach().cpu().numpy() # Already normalized quaternions
        opacities = self.get_opacity.detach().cpu().numpy() # Already processed opacities
        
        # Get raw SH features for color conversion
        features = self.get_features.detach().cpu().numpy() # Raw SH coefficients
        
        # Filter out very small or transparent splats
        if max_splats and len(xyz) > max_splats:
            # Keep the most important splats based on size and opacity
            importance = opacities.squeeze() * np.max(scales, axis=1)
            keep_indices = np.argsort(importance)[-max_splats:]
            xyz = xyz[keep_indices]
            scales = scales[keep_indices]
            rotations = rotations[keep_indices]
            opacities = opacities[keep_indices]
            features = features[keep_indices]
        
        # Convert SH DC coefficients to RGB colors
        # TRELLIS stores features as [N, 1, 3] or [N, 3, 1] format
        if features.shape[1] == 1 and features.shape[2] == 3:
            # Shape: [N, 1, 3] - 1 SH coefficient per RGB channel
            sh_dc = features[:, 0, :]  # [N, 3]
        elif features.shape[1] == 3 and features.shape[2] == 1:
            # Shape: [N, 3, 1] - 3 RGB channels, 1 SH coefficient each
            sh_dc = features[:, :, 0]  # [N, 3]
        else:
            # Fallback for unexpected shapes
            sh_dc = features.reshape(features.shape[0], 3)
        
        # Convert SH DC to RGB using the standard spherical harmonics formula
        # SH DC coefficient corresponds to constant term: DC / (2 * sqrt(π)) + 0.5
        # But TRELLIS seems to use a different encoding, so we need to experiment
        
        # Method 1: Direct sigmoid (common for neural networks)
        colors = 1.0 / (1.0 + np.exp(-sh_dc))
        
        # Ensure colors are in valid [0,1] range
        colors = np.clip(colors, 0.0, 1.0)
        
        # Apply coordinate transform if specified
        if transform is not None:
            transform = np.array(transform, dtype=np.float32)
            xyz = np.matmul(xyz, transform.T)
            
            # Transform rotations properly
            rotation_matrices = utils3d.numpy.quaternion_to_matrix(rotations)
            rotation_matrices = np.matmul(transform, rotation_matrices)
            rotations = utils3d.numpy.matrix_to_quaternion(rotation_matrices)
        
        # Ensure quaternions are normalized
        quat_norms = np.linalg.norm(rotations, axis=1, keepdims=True)
        rotations = rotations / (quat_norms + 1e-8)
        
        # Clamp all values to reasonable ranges
        xyz = np.clip(xyz, -100.0, 100.0)
        scales = np.clip(scales, 1e-6, 10.0)  # Prevent zero or huge scales
        colors = np.clip(colors, 0.0, 1.0)
        opacities = np.clip(opacities.squeeze(), 0.0, 1.0)
        
        # Pack data into .splat format
        # Standard format: [x, y, z, scale_x, scale_y, scale_z, r, g, b, quat_w, quat_x, quat_y, quat_z, opacity]
        num_splats = len(xyz)
        splat_data = np.zeros((num_splats, 14), dtype=np.float32)
        
        splat_data[:, 0:3] = xyz.astype(np.float32)        # position (x, y, z)
        splat_data[:, 3:6] = scales.astype(np.float32)     # scale (sx, sy, sz)
        splat_data[:, 6:9] = colors.astype(np.float32)     # color (r, g, b)
        splat_data[:, 9:13] = rotations.astype(np.float32) # quaternion (w, x, y, z)
        splat_data[:, 13] = opacities.astype(np.float32)   # opacity
        
        # Write binary file
        with open(path, 'wb') as f:
            f.write(splat_data.tobytes())

    def save_ply_gaussian_splatting(self, path, transform=[[1, 0, 0], [0, 0, -1], [0, 1, 0]], max_splats=50000):
        """Save in standard Gaussian Splatting PLY format with proper filtering"""
        
        # Get and filter data (same as save_splat)
        xyz = self.get_xyz.detach().cpu().numpy()
        scales = self.get_scaling.detach().cpu().numpy()
        rotations = self.get_rotation.detach().cpu().numpy()
        opacities = self.get_opacity.detach().cpu().numpy().squeeze()
        features = self.get_features.detach().cpu().numpy()
        
        # Filter tiny/invisible splats
        min_scale = np.max(scales, axis=1)
        valid_mask = (min_scale > 0.001) & (opacities > 0.01)
        
        xyz = xyz[valid_mask]
        scales = scales[valid_mask]
        rotations = rotations[valid_mask]
        opacities = opacities[valid_mask]
        features = features[valid_mask]
        
        # Limit count
        if len(xyz) > max_splats:
            importance = opacities * np.max(scales, axis=1)
            indices = np.argsort(importance)[-max_splats:]
            xyz = xyz[indices]
            scales = scales[indices]
            rotations = rotations[indices]
            opacities = opacities[indices]
            features = features[indices]
        
        # Apply transform
        if transform is not None:
            transform = np.array(transform)
            xyz = np.matmul(xyz, transform.T)
            rotation_matrices = utils3d.numpy.quaternion_to_matrix(rotations)
            rotation_matrices = np.matmul(transform, rotation_matrices)
            rotations = utils3d.numpy.matrix_to_quaternion(rotation_matrices)
        
        # Store raw features for PLY (no color conversion)
        if features.shape[1] == 1 and features.shape[2] == 3:
            features_rgb = features[:, 0, :]  # [N, 3]
        elif features.shape[1] == 3 and features.shape[2] == 1:
            features_rgb = features[:, :, 0]  # [N, 3]
        else:
            features_rgb = np.zeros((len(xyz), 3))
        
        # Create PLY data
        vertex_data = []
        for i in range(len(xyz)):
            vertex = [
                xyz[i, 0], xyz[i, 1], xyz[i, 2],  # position
                0.0, 0.0, 0.0,                     # normals (unused)
                features_rgb[i, 0], features_rgb[i, 1], features_rgb[i, 2],  # SH DC
                opacities[i],                      # opacity
                scales[i, 0], scales[i, 1], scales[i, 2],  # scales
                rotations[i, 0], rotations[i, 1], rotations[i, 2], rotations[i, 3]  # quaternion
            ]
            vertex_data.append(tuple(vertex))
        
        # PLY properties
        properties = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
            ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
        ]
        
        vertex_element = PlyElement.describe(np.array(vertex_data, dtype=properties), 'vertex')
        PlyData([vertex_element], text=True).write(path)