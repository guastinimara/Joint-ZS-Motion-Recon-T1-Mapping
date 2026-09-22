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

