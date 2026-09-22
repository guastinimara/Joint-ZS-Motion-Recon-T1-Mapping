"""Selection of dynamics by cardiac phase (RR index) for the initial reference fit.

Starting from the phase of the first dynamic, neighbouring phases are added
([0] -> [0, 1] -> [0, 1, -1] -> ...) until the T1 fit on the selected dynamics succeeds.
"""
import torch
from datetime import datetime

def log_to_file(log_file, message):
    """Helper function to write to log file with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)


def compute_rr_mask(rr_idx, base_idx, expansion_offsets):
    """
    Compute mask based on base index and expansion offsets.
    Uses modulo arithmetic to handle values that exceed rr_idx.max().
    
    Args:
        rr_idx: tensor of RR indices
        base_idx: base RR index value (usually rr_idx[0])
        expansion_offsets: list of offsets to include (e.g., [0], [0, 1], [0, 1, -1], etc.)
    
    Returns:
        mask_condition: boolean tensor where True means the sample should be included
    """
    rr_max = rr_idx.max().item()
    target_values = []
    
    for offset in expansion_offsets:
        target_val = (base_idx + offset) % (rr_max + 1)  # Modulo to wrap around
        target_values.append(target_val)
    
    # Create mask for any of the target values
    mask_condition = torch.zeros_like(rr_idx, dtype=torch.bool)
    for val in target_values:
        mask_condition |= (rr_idx == val)
    
    return mask_condition



def get_next_expansion(current_expansion):
    """
    Get the next expansion pattern by alternating between positive and negative offsets.
    Pattern: [0] → [0,1] → [0,1,-1] → [0,1,-1,2] → [0,1,-1,2,-2] → ...
    
    Args:
        current_expansion: current list of offsets
    
    Returns:
        next_expansion: expanded list of offsets
    """
    if len(current_expansion) == 1:  # Only base index [0]
        return [0, 1]
    
    # Find the maximum absolute offset
    max_offset = max(abs(x) for x in current_expansion if x != 0)
    
    # Check if we have both positive and negative versions of max_offset
    has_positive_max = max_offset in current_expansion
    has_negative_max = -max_offset in current_expansion
    
    if has_positive_max and has_negative_max:
        # We have both +max and -max, so add next positive offset
        next_expansion = current_expansion + [max_offset + 1]
    elif has_positive_max and not has_negative_max:
        # We have +max but not -max, so add the negative version
        next_expansion = current_expansion + [-max_offset]
    else:
        # This shouldn't happen in normal flow, but just in case add next positive
        next_expansion = current_expansion + [max_offset + 1]
    
    return next_expansion



    
def find_valid_rr_threshold(rr_idx, x_m, params_estim, logger=None, max_attempts=20, recon_params=None,initial_expansion=[0,1,-1,2]):
    """
    Find valid RR threshold using expanding mask approach.
    
    Args:
        rr_idx: tensor of RR indices
        x_m: input tensor for params_estim.fit
        params_estim: T1ParameterEstimator object
        logger: Logger object for logging messages
        max_attempts: maximum number of expansion attempts
        recon_params: config dict passed to params_estim.fit
        initial_expansion: offsets to start from
    
    Returns:
        tuple: (p_0_ref, x_0_ref, expansion_offsets) or (None, None, None) if failed
    """
    base_idx = rr_idx[0].item()
    expansion_offsets = initial_expansion  # Start with the provided initial expansion
    
    if logger is not None:
        logger.log(f"Starting RR threshold search with base_idx: {base_idx}")
    
    for attempt in range(max_attempts):
        # Compute mask for current expansion
        mask_condition = compute_rr_mask(rr_idx, base_idx, expansion_offsets)
        
        # Check if we have any valid samples
        if not mask_condition.any():
            if logger is not None:
                logger.log(f"Attempt {attempt + 1}: No valid samples with expansion {expansion_offsets}, expanding...")
            expansion_offsets = get_next_expansion(expansion_offsets)
            continue
        
        # Try fitting with current mask
        num_samples = mask_condition.sum().item()
        percentage = (num_samples / rr_idx.numel()) * 100
        if logger is not None:
            logger.log(f"Attempt {attempt + 1}: Trying fitting with {num_samples} samples ({percentage:.2f}%), expansion: {expansion_offsets}")
        
        p_0_ref, x_0_ref = params_estim.fit(x_m, mask=mask_condition, recon_params=recon_params)
        
        # Check for NaN values
        if not torch.isnan(x_0_ref).any():
            if logger is not None:
                logger.log(f"Successfully computed x_ref with expansion: {expansion_offsets} (base_idx: {base_idx})")
            return p_0_ref, x_0_ref, expansion_offsets
        
        if logger is not None:
            logger.log(f"Attempt {attempt + 1}: x_0_ref contains NaN with expansion {expansion_offsets}, expanding...")
        expansion_offsets = get_next_expansion(expansion_offsets)
    
    if logger is not None:
        logger.log(f"Failed to find valid RR threshold after {max_attempts} attempts")
    return None, None, None




def update_rr_expansion_for_iteration(current_expansion):
    """
    Update the RR expansion pattern for the next iteration.
    Alternates between adding positive and negative offsets.
    
    Args:
        current_expansion: current list of offsets
    
    Returns:
        updated_expansion: expanded list of offsets for next iteration
    """
    return get_next_expansion(current_expansion)



def update_rr_expansion_and_compute_rr_mask(rr_idx, base_idx, expansion_offsets):
    """
    Expand the offsets by one step, then compute the mask as in ``compute_rr_mask``.
    
    Args:
        rr_idx: tensor of RR indices
        base_idx: base RR index value (usually rr_idx[0])
        expansion_offsets: current list of offsets (e.g., [0], [0, 1], [0, 1, -1], etc.)
    
    Returns:
        mask_condition: boolean tensor where True means the sample should be included
        expansion_offsets: the expanded list of offsets
    """
    expansion_offsets = update_rr_expansion_for_iteration(expansion_offsets)           
    
    rr_max = rr_idx.max().item()
    target_values = []
    
    for offset in expansion_offsets:
        target_val = (base_idx + offset) % (rr_max + 1)  # Modulo to wrap around
        target_values.append(target_val)
    
    # Create mask for any of the target values
    mask_condition = torch.zeros_like(rr_idx, dtype=torch.bool)
    for val in target_values:
        mask_condition |= (rr_idx == val)
    
    return mask_condition, expansion_offsets