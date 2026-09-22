#%%
"""Rectangular crop around the region of interest (heart), with parameters stored as json."""
import torch
import matplotlib.pyplot as plt
import os 
import json

class MaskImgSize():
    """Crop [h_start:h_start+h_width, w_start:w_start+w_width] on the last two dimensions."""
    def __init__(self, h_start=None, w_start=None, h_width=None, w_width=None):
        self.h_start = h_start
        self.w_start = w_start
        self.h_width = h_width
        self.w_width = w_width
        
    def apply(self, x):
        """Return the cropped tensor."""
        if self.h_start is None or self.w_start is None or self.h_width is None or self.w_width is None:
            raise ValueError("Mask values are not set.")
        return x[...,
                 self.h_start:self.h_start+self.h_width, 
                 self.w_start:self.w_start+self.w_width]
    
    def inverse_mask_on_mf(self, mf, output_shape):
        """Zero-pad cropped motion fields (t, ch, h, w) back to output_shape (full image size)."""
        # mf 'b (t ch) h w'
        if self.h_start is None or self.w_start is None or self.h_width is None or self.w_width is None:
            raise ValueError("Mask values are not set.")
        
        mf_output = torch.zeros((mf.shape[0], mf.shape[1], output_shape[0], output_shape[1])).to(mf.device)
        mf_output[..., self.h_start:self.h_start+self.h_width,
                        self.w_start:self.w_start+self.w_width] = mf
        return mf_output
        
    def get_masked_shape(self):
        """Return (h_width, w_width)."""
        if self.h_width is None or self.w_width is None:
            raise ValueError("Mask values are not set.")
        return self.h_width, self.w_width
    
    def set_and_plot_mask_values(self, h_start, w_start, h_width, w_width, image=None, plot=False):
        """Set the crop and optionally show it on image."""
        self.h_start = h_start
        self.w_start = w_start
        self.h_width = h_width
        self.w_width = w_width
        if plot and image is not None:
            self.plot_mask(image)
          
            
    def plot_mask(self, image):
        """Show the cropped image (expects a 2D image)."""
        if self.h_start is None or self.w_start is None or self.h_width is None or self.w_width is None:
            raise ValueError("Mask values are not set.")
        masked_image = self.apply(image)
        # image shape t 1 1 h w
        plt.imshow(masked_image.abs().cpu().numpy())
        plt.title('Masked image')
        plt.show()
        
    def save_parameters(self, directory, file_name):
        """Save the crop to directory/file_name as json."""
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, file_name)
        parameters = {
            'h_start': self.h_start,
            'w_start': self.w_start,
            'h_width': self.h_width,
            'w_width': self.w_width
        }
        with open(file_path, 'w') as f:
            json.dump(parameters, f)
        print(f"Parameters saved to {file_path}")

    def load_parameters(self, file_path):
        """Load the crop from a json file."""
        with open(file_path, 'r') as f:
            parameters = json.load(f)
        self.h_start = parameters.get('h_start')
        self.w_start = parameters.get('w_start')
        self.h_width = parameters.get('h_width')
        self.w_width = parameters.get('w_width')
        
    def load_params(self, mask_path, img=None):
        """Load the crop from a json file (same as load_parameters, requires all keys)."""
        with open(mask_path, 'r') as file:
            mask_params = json.load(file)
            h_start = mask_params['h_start']
            w_start = mask_params['w_start']
            h_width = mask_params['h_width']
            w_width = mask_params['w_width']
        
        self.set_and_plot_mask_values(h_start, 
                                      w_start, 
                                      h_width, 
                                      w_width, 
                                      img,
                                      plot=False)

        
#%%
# import numpy as np
# from einops import rearrange    
# prefix = 'rt_cine_tvt_nr9_pe441_sl0-5_'
# filenames = [
#             # prefix + 'pat0',
#             # prefix + 'pat2',
#             # prefix + 'pat3',
#             # prefix + 'pat4',
#             # prefix + 'pat5',
#             # prefix + 'pat6',
#             # prefix + 'pat7',
#             # prefix + 'pat8',
#             # prefix + 'pat9',
#             # prefix + 'pat10',
#             # prefix + 'pat11',
#             # prefix + 'pat12',
#             # prefix + 'pat13',
#             # prefix + 'pat14',
#             # prefix + 'pat15',
#             # prefix + 'vol0',
#             prefix + 'vol1',
#             prefix + 'vol2',
#             prefix + 'vol3',
#             prefix + 'vol4',
#             prefix + 'vol5',
#             # prefix + 'vol6',
#             # prefix + 'vol7',
#             # prefix + 'vol8',
#             # prefix + 'vol9'
# ]
# #%%

# pat = 'vol4'
# kind = '1'
# #folder = f'/echo/guasti01/data/KirstenData/{prefix+pat}.npz'
# folder = f'/echo/guasti01/code/paper_1_scripts/reproducibility_exp/data/{prefix+pat}_{kind}.npz'
# img = torch.from_numpy(np.load(folder,allow_pickle=True)['icg'])
# img = rearrange(img, 'h w t-> t 1 1 h w ')
# plt.imshow(img[0,0,0].abs())

# #%%
# mask = MaskMF()

# h_start = 175
# w_start = 190
# h_width = 116
# w_width = 116
# mask.set_and_plot_mask_values(h_start, w_start, h_width, w_width, img,plot=True)
# #%%
# directory = '/echo/guasti01/code/paper_1_scripts/reproducibility_exp/masks/'
# filename = f'mask_parameters_{pat}_{kind}.json'
# mask.save_parameters(directory, filename)

#%%




























#%%
###############################################################################

# import numpy as np
# from einops import rearrange    
# from load_data import load_data_temp_simone_stack

# patient = '12-2022_14__02'
# view_type = 'transl'
# stack= '1'
# dir = f'/echo/guasti01/Simone/phdproject/Data/results/inVivo/{patient}/12stacks_5slices_4mm_gap14_transl/dynamics_cg/'
# mf_dir = f'/echo/guasti01/Simone/phdproject/Data/results/inVivo/{patient}/12stacks_5slices_4mm_gap14_transl/cardiacMotion_npy_tv/'
       
       
# rr_idx_stack, x_m_stack, sampling_times, repetition_times, mf_inv_comparison_stack = load_data_temp_simone_stack(
#         pat=patient, 
#         dir = dir,
#         mf_dir=mf_dir,
#         view_type=view_type, 
#         stack=stack,
#         dyns_name = 'dyns_cg',
#         device='cpu')

# #%%
# slice_idx = 3

# img = x_m_stack[slice_idx]
# plt.imshow(img[10,0,0].abs())
# #%%
# sum_img = x_m_stack[0] + x_m_stack[1] + x_m_stack[3] + x_m_stack[4] 
# #plt.imshow(sum_img.sum(0)[0,0].abs())
# #%%
# mask = MaskMF()

# h_start = 60
# w_start = 60
# h_width = 120
# w_width = 120
# mask.set_and_plot_mask_values(h_start, 
#                               w_start, 
#                               h_width, 
#                               w_width, 
#                               img,
#                               plot=True)
# #%%

# plt.imshow(mask.apply(img)[20,0,0].abs())



# #%%
# plt.imshow(mask.apply(img)[:,0,0,70].abs())
# ''
# #%%
# # plt.imshow(mask.apply(sum_img).sum(0)[0,0].abs())
# #%%
# directory = f'/echo/guasti01/code/simone_data/masks/{patient}/{view_type}/'
# # make the directory if not existing
# if not os.path.exists(directory):
#     os.makedirs(directory)

# filename = f'mask_parameters_stack0.json'
# mask.save_parameters(directory, filename)
#%%



















# #%%
#%%
###############################################################################
# import numpy as np
# from einops import rearrange    
# from load_data import load_data_simone_stack, load_data_temp_simone_stack


# patient =   '12-2022_14__01'
# view_type = 'transl'
# stack= '8'

# dir = f'/echo/guasti01/Simone/phdproject/Data/results/inVivo/{patient}/12stacks_5slices_4mm_gap14_{view_type}/dynamics_cg/'
# mf_dir = f'/echo/guasti01/Simone/phdproject/Data/results/inVivo/{patient}/12stacks_5slices_4mm_gap14_{view_type}/cardiacMotion_npy_tv/'

# rr_idx_stack, x_m_stack, sampling_times, repetition_times, mf_inv_comparison_stack = load_data_temp_simone_stack(
#     pat=patient, 
#     dir = dir,
#     mf_dir=mf_dir,
#     dyns_name='dyns_cg',
#     view_type=view_type, 
#     stack=stack,
#     device='cpu')


# #%%
# slice_idx = 3

# img = x_m_stack[slice_idx]
# plt.imshow(img[10,0,0].abs())

# #%%
# mask = MaskMF()

# h_start = 60
# w_start = 60
# h_width = 130
# w_width = 130
# mask.set_and_plot_mask_values(h_start, 
#                               w_start, 
#                               h_width, 
#                               w_width, 
#                               img,
#                               plot=True)
# #%%

# plt.imshow(mask.apply(img)[10,0,0].abs())

# #%%
# sum_img = img.sum(dim=0)
# plt.imshow(mask.apply(sum_img)[0,0].abs())




# #%%
# #%%
# directory = f'/echo/guasti01/code/simone_data/masks/{patient}/{view_type}/'
# # make the directory if not existing
# if not os.path.exists(directory):
#     os.makedirs(directory)

# filename = f'mask_parameters_stack0.json'
# mask.save_parameters(directory, filename)



# #%%
