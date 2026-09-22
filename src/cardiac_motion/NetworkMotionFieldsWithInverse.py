"""Network predicting motion fields for all dynamics at once, plus their inverses."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from einops import rearrange
from .basic_network import UNetBN_attention, ConvBlockBN

class NetworkMotionFieldsWithInverse(nn.Module):
    """Attention U-Net that outputs a stationary velocity field per dynamic.

    The velocity is integrated with scaling and squaring to get a diffeomorphic forward field;
    integrating the negated velocity gives the inverse field.
    """
    # Spatial/integration helpers adapted from Voxelmorph:
    # https://github.com/voxelmorph/voxelmorph

    class SpatialTransformer(nn.Module):
        """
        N-D Spatial Transformer
        """
        def __init__(self, size,mode='bicubic'): 
            super().__init__()

            self.mode = mode
            
            # create sampling grid
            vectors = [torch.arange(0, s) for s in size]
            grids = torch.meshgrid(vectors)
            grid = torch.stack(grids)
            grid = torch.unsqueeze(grid, 0)
            grid = grid.type(torch.FloatTensor)

            # registering the grid as a buffer cleanly moves it to the GPU, but it also
            # adds it to the state dict. this is annoying since everything in the state dict
            # is included when saving weights to disk, so the model files are way bigger
            # than they need to be. so far, there does not appear to be an elegant solution.
            # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
            self.register_buffer('grid', grid)

        def get_grid(self):
            """Return the identity sampling grid."""
            return self.grid

        def forward(self, src, field, normed_grid=False):
            """Warp src (b, ch, h, w) with displacement field (b, 2, h, w) in pixels."""
            # new locations
            new_locs = self.grid + field
            shape = field.shape[2:]

            # need to normalize grid values to [-1, 1] for resampler
            for i in range(len(shape)):
                new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

            if normed_grid:
                new_locs = field

            # move channels dim to last position
            # also not sure why, but the channels need to be reversed
            if len(shape) == 2:
                new_locs = new_locs.permute(0, 2, 3, 1)
                new_locs = new_locs[..., [1, 0]]
            elif len(shape) == 3:
                new_locs = new_locs.permute(0, 2, 3, 4, 1)
                new_locs = new_locs[..., [2, 1, 0]]

            return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode, padding_mode='border')

        def apply_on_mrpro(self, img, field):
                """Apply the spatial transformer using MRpro-compatible tensor shapes.

                Shape convention:
                - img: (t, 1, 1, h, w), real or complex
                - field: (t, 2, h, w), displacement field in pixel units
                - returns: (t, 1, 1, h, w), same dtype domain as input image

                This helper keeps a consistent interface with MRpro data pipelines.
                https://github.com/PTB-MR/mrpro
                """

                # Field is expected to contain 2D displacement components.
                if field.shape[1] != 2:
                    raise ValueError('Field should have 2 as second dimension')
                
                complex_input = torch.is_complex(img)
                
                # Convert complex data to two real channels before warping.
                if complex_input:
                    img_to_warp = torch.view_as_real(img.contiguous())
                    img_to_warp = rearrange(img_to_warp, 't 1 1 h w ch -> t ch h w')
                else:
                    img_to_warp = rearrange(img, 't 1 1 h w -> t 1 h w')

                # Warp in channel-first layout expected by grid_sample.
                warped_img = self.forward(img_to_warp, field)

                # Restore MRpro layout and original real/complex representation.
                if complex_input:
                    warped_img = rearrange(warped_img, 't ch h w -> t 1 1 h w ch')
                    warped_img = torch.view_as_complex(warped_img.contiguous())
                else:
                    warped_img = rearrange(warped_img, 't 1 h w -> t 1 1 h w')
                
                return warped_img


        def get_normalized_grid(self,field,vf=True):
            """Convert a displacement field (vf=True) or absolute grid (vf=False) to [-1, 1] coordinates."""
            if vf: # from field = vector field -> normalized grid
                new_locs = self.grid + field
            else: # from field = grid (id + vf) -> normalized grid
                new_locs = field
            shape = field.shape[2:]
            for i in range(len(shape)):
                new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
            return new_locs

    class VectorIntegration(nn.Module):
        """
        Integrates a vector field via scaling and squaring.
        """

        def __init__(self, inshape, nsteps, mode='bilinear'):
            super().__init__()

            assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
            self.nsteps = nsteps
            self.scale = 1.0 / (2 ** self.nsteps)
            self.transformer = NetworkMotionFieldsWithInverse.SpatialTransformer(inshape[:], mode=mode)

        def forward(self, vec):
            vec = vec * self.scale
            for _ in range(self.nsteps):
                vec = vec + self.transformer(vec, vec)
            return vec

    class ResizeTransform(nn.Module):
        """
        Resize a transform, which involves resizing the vector field *and* rescaling it.
        """

        def __init__(self, vel_resize, ndims):
            #ndims= 2
            super().__init__()
            self.factor = 1.0 / vel_resize
            self.mode = 'linear'
            if ndims == 2:
                self.mode = 'bi' + self.mode
            elif ndims == 3:
                self.mode = 'tri' + self.mode

        def forward(self, x):
            if self.factor < 1:
                # resize first to save memory
                x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
                x = self.factor * x

            elif self.factor > 1:
                # multiply first to save memory
                x = self.factor * x
                x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

            # don't do anything if resize is 1
            return x

    def __init__(self,
                 dim,
                 inshape,
                 src_feats=54,
                 out_feats = 54,
                 ch_out_unet = 64,
                 int_steps=3,
                 int_downsize=2,
                 unet_half_res=False,
                 n_enc_stages=5,
                 n_filters = 64,
                 refinement=False
                 ):
        """
        Args:
            dim: spatial dimensions (2).
            inshape: input image shape (h, w).
            src_feats: input channels (= number of dynamics).
            out_feats: output channels (= 2 * number of dynamics).
            ch_out_unet, n_enc_stages, n_filters: U-Net size.
            int_steps: scaling and squaring steps (0 disables integration).
            int_downsize: integrate at a lower resolution by this factor.
            unet_half_res: if True, skip the downsizing before integration.
            refinement: use extra refinement convolutions (requires ``refinement_convs``).
        """
        super().__init__()
        
        # internal flag indicating whether to return field or integrated warp during inference
        self.training = True
        self.refinement = refinement
        
    
        # configure core unet model
        self.unet_model = UNetBN_attention(
            dim,
            n_ch_in=src_feats,
            n_ch_out = ch_out_unet,
            n_enc_stages=n_enc_stages, #5
            n_convs_per_stage=1, #1
            n_filters=n_filters, 
            kernel_size=3
        )

        self.times = src_feats
        self.c1 = ConvBlockBN(dim,ch_out_unet,32,3) # prima ce nerano 3, also or: 16, ... 64
        
        self.make_field = nn.Conv2d(in_channels=32,out_channels=out_feats,kernel_size=3, padding=1)

        self.make_field.weight = nn.Parameter(Normal(0, 1e-5).sample(self.make_field.weight.shape))
        self.make_field.bias = nn.Parameter(torch.zeros(self.make_field.bias.shape))

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = self.ResizeTransform(int_downsize, 2) #ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            #print(f"int_downsize = {int_downsize}")
            self.fullsize = self.ResizeTransform(1 / int_downsize, 2) #ndims)
        else:
            self.fullsize = None

      
        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape[:]]
        self.integrate = self.VectorIntegration(down_shape, int_steps) if int_steps > 0 else None

        
    def forward(self, source, log_attention=False):
        """
        Args:
            source: images (1, t, h, w).
            log_attention: also return the attention maps of the decoder.

        Returns:
            preint_fields: velocity field before integration (t, 2, h, w).
            motion_fields: stacked [forward, inverse] fields (2, t, 2, h, w).
            attention_maps (only if log_attention).
        """
        
        x = source
        # apply the unet
        if log_attention:
            x, attention_maps = self.unet_model(x, log_attention=log_attention) 
        else:
            x = self.unet_model(x, log_attention=log_attention) 
            
        #print('x.shape after unet : ', x.shape)
        x = self.c1(x)
        if self.refinement:
            x = self.refinement_convs(x)
            
        fields = self.make_field(x)
        
        fields = rearrange(fields, 'b (ch t) h w -> (b t) ch h w', t=self.times, ch=2)
        # fields is forward motion, shape (t, 2, h, w)
        
        preint_fields = fields
        pos_fields = fields # (t, 2, h, w)
        neg_fields = -pos_fields # (t, 2, h, w)
        
        if self.resize:
            pos_fields = self.resize(pos_fields)
            neg_fields = self.resize(neg_fields)
            
        if self.integrate:
            pos_fields = self.integrate(pos_fields)
            neg_fields = self.integrate(neg_fields)
            
            # resize to final resolution
            if self.fullsize:
                pos_fields = self.fullsize(pos_fields)
                neg_fields = self.fullsize(neg_fields)

        pos_fields = pos_fields.unsqueeze(0) # (1, t, 2, h, w)
        neg_fields = neg_fields.unsqueeze(0) # (1, t, 2, h, w)
        motion_fields = torch.cat([pos_fields, neg_fields], dim=0)
        
        if log_attention:
            return preint_fields, motion_fields, attention_maps
        else:
            return preint_fields, motion_fields





        