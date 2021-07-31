import warnings
import logging
import torch

from DVR_renderer.scratch import Generator
from im2scene.giraffe.models import decoder, bounding_box_generator

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logging.getLogger().setLevel(logging.DEBUG)
    device = torch.device("cpu")
    # settings in default.yaml
    z_dim = 256
    z_dim_bg = 128
    bg_gen_kwargs = {"hidden_size": 64,
                     "n_blocks": 4,
                     "downscale_p_by": 12,
                     "skips": []}
    bg_gen = decoder.Decoder(z_dim=z_dim_bg, **bg_gen_kwargs)
    bounding_box_gen = bounding_box_generator.BoundingBoxGenerator(z_dim=z_dim)
    decoder = decoder.Decoder(z_dim=z_dim)

    renderer = Generator(device,
                         z_dim=z_dim,
                         z_dim_bg=z_dim_bg,
                         background_generator=bg_gen,
                         bounding_box_generator=bounding_box_gen,
                         decoder=decoder,
                         neural_renderer=None)
    rendered_img = renderer()
    print(rendered_img.shape)
