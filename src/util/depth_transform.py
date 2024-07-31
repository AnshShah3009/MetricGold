# Author: Bingxin Ke
# Last modified: 2024-04-18

import torch
import logging

def get_depth_normalizer(cfg_normalizer):
    if cfg_normalizer is None:
        def identical(x):
            return x
        depth_transform = identical

    elif "scale_shift_depth" == cfg_normalizer.type:
        depth_transform = ScaleShiftDepthNormalizer(
            norm_min=cfg_normalizer.norm_min,
            norm_max=cfg_normalizer.norm_max,
            min_max_quantile=cfg_normalizer.min_max_quantile,
            clip=cfg_normalizer.clip,
        )

    elif "log_depth" == cfg_normalizer.type:
        depth_transform = LogDepthNormalizer(
            norm_min=cfg_normalizer.norm_min,
            norm_max=cfg_normalizer.norm_max,
            dmin=cfg_normalizer.dmin,
            dmax=cfg_normalizer.dmax,
            clip=cfg_normalizer.clip,
        )
    else:
        raise NotImplementedError

    return depth_transform

class DepthNormalizerBase:
    is_absolute = None
    far_plane_at_max = None

    def __init__(
        self,
        norm_min=-1.0,
        norm_max=1.0,
    ) -> None:
        self.norm_min = norm_min
        self.norm_max = norm_max
        raise NotImplementedError

    def __call__(self, depth, valid_mask=None, clip=None):
        raise NotImplementedError

    def denormalize(self, depth_norm, **kwargs):
        # For metric depth: convert prediction back to metric depth
        # For relative depth: convert prediction to [0, 1]
        raise NotImplementedError

class ScaleShiftDepthNormalizer(DepthNormalizerBase):
    """
    Use near and far plane to linearly normalize depth,
        i.e. d' = d * s + t,
        where near plane is mapped to `norm_min`, and far plane is mapped to `norm_max`
    Near and far planes are determined by taking quantile values.
    """

    is_absolute = False
    far_plane_at_max = True

    def __init__(
        self, norm_min=-1.0, norm_max=1.0, min_max_quantile=0.02, clip=True
    ) -> None:
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.norm_range = self.norm_max - self.norm_min
        self.min_quantile = min_max_quantile
        self.max_quantile = 1.0 - self.min_quantile
        self.clip = clip

    def __call__(self, depth_linear, valid_mask=None, clip=None):
        clip = clip if clip is not None else self.clip

        if valid_mask is None:
            valid_mask = torch.ones_like(depth_linear).bool()
        valid_mask = valid_mask & (depth_linear > 0)

        # Take quantiles as min and max
        _min, _max = torch.quantile(
            depth_linear[valid_mask],
            torch.tensor([self.min_quantile, self.max_quantile]),
        )

        # scale and shift
        depth_norm_linear = (depth_linear - _min) / (
            _max - _min
        ) * self.norm_range + self.norm_min

        if clip:
            depth_norm_linear = torch.clip(
                depth_norm_linear, self.norm_min, self.norm_max
            )

        return depth_norm_linear

    def scale_back(self, depth_norm):
        # scale to [0, 1]
        depth_linear = (depth_norm - self.norm_min) / self.norm_range
        return depth_linear

    def denormalize(self, depth_norm, **kwargs):
        logging.warning(f"{self.__class__} is not revertible without GT")
        return self.scale_back(depth_norm=depth_norm)

class LogDepthNormalizer(DepthNormalizerBase):
    """
    Use fixed near and far planes to logarithmically normalize depth,
        i.e. d' = log(d / d_min) / log(d_max / d_min),
        where near plane is mapped to `norm_min`, and far plane is mapped to `norm_max`.
    """

    is_absolute = False
    far_plane_at_max = True

    def __init__(
        self, norm_min=-1.0, norm_max=1.0, dmin=0.1, dmax=100.0, clip=True
    ) -> None:
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.norm_range = self.norm_max - self.norm_min
        self.dmin = dmin
        self.dmax = dmax
        self.clip = clip

    def __call__(self, depth_linear, valid_mask=None, clip=None):
        clip = clip if clip is not None else self.clip

        if valid_mask is None:
            valid_mask = torch.ones_like(depth_linear).bool()
        valid_mask = valid_mask & (depth_linear > 0)

        # Logarithmic normalization
        depth_norm_linear = torch.log(depth_linear / self.dmin) / torch.log(self.dmax / self.dmin)

        if clip:
            depth_norm_linear = torch.clip(
                depth_norm_linear, self.norm_min, self.norm_max
            )

        return depth_norm_linear

    def scale_back(self, depth_norm):
        # Reverse the logarithmic normalization
        depth_linear = self.dmin * torch.exp(depth_norm * torch.log(self.dmax / self.dmin))
        return depth_linear

    def denormalize(self, depth_norm, **kwargs):
        logging.warning(f"{self.__class__} is not revertible without GT")
        return self.scale_back(depth_norm=depth_norm)

# Example usage
# if __name__ == "__main__":
#     depth_linear = torch.randn(

#     cfg_normalizer_log = type('cfg', (object,), {'type': 'log_depth', 'norm_min': -1.0, 'norm_max': 1.0, 'dmin': 0.1, 'dmax': 5.0, 'clip': True})
#     normalizer = get_depth_normalizer(cfg_normalizer_log)

#     normalized_depth = normalizer(depth_linear)
#     print("Normalized depth:", normalized_depth)

if __name__ == "__main__":
    # Example depth image of size 5x5
    h, w = 5, 5
    depth_image = torch.tensor([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.6, 0.7, 0.8, 0.9, 1.0],
        [1.1, 1.2, 1.3, 1.4, 1.5],
        [1.6, 1.7, 1.8, 1.9, 2.0],
        [2.1, 2.2, 2.3, 2.4, 2.5]
    ])

    cfg_normalizer_log = type('cfg', (object,), {'type': 'log_depth', 'norm_min': -1.0, 'norm_max': 1.0, 'dmin': 0.1, 'dmax': 2.5, 'clip': True})
    normalizer = get_depth_normalizer(cfg_normalizer_log)

    normalized_depth = normalizer(depth_image)
    print("Normalized depth:\n", normalized_depth)