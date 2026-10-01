"""Low-rank temporal regularization.

The dynamic series (time, ny, nx) is flattened to (time, voxels) and normalized; U from its SVD is
a temporal basis. Coefficients in this basis are found with an L2 data term plus an L1 sparsity
penalty (proximal gradient descent), giving a low-rank approximation of the images.
"""
import torch
import torch.nn as nn

import mrpro 


import torch.optim as optim
from einops import rearrange


def compute_svd(x_full,mask, divide_by_n=False):
    """Compute the temporal SVD basis inside a mask and the sparse approximation of the full images.

    Args:
        x_full: dynamic images (time, ..., ny, nx).
        mask: object with ``apply`` that crops/masks the images (e.g. around the heart).
        divide_by_n: normalize the functionals by the number of elements.

    Returns:
        image_approx: low-rank approximation (time, 1, ny, nx).
        U_op: operator applying the temporal basis.
        coefficients: coefficients in the basis (voxels, time).
    """
    device = x_full.device
    x_small = mask.apply(x_full).squeeze().unsqueeze(0)
    x_small = rearrange(x_small, "1 time ny nx  -> time 1 ny nx").to(device)

    ntime, nslices,ny, nx = x_small.shape 
    xp = rearrange(x_small, "time slices ny nx  -> time (slices ny nx)")
    dim=(-1,)
    mu, std = xp.mean(dim=dim, keepdim=True), xp.std(dim=dim, keepdim=True)
    xp = (xp - mu) / std
    U, _, _ = torch.linalg.svd(xp)
    
    U_op = mrpro.operators.EinsumOp(U).to(device)

    x_full = x_full.squeeze().unsqueeze(0)
    x_full = rearrange(x_full, "1 time ny nx  -> time 1 ny nx").to(device)

    ntime, nslices,ny, nx = x_full.shape 
    xp = rearrange(x_full, "time slices ny nx  -> time (slices ny nx)")
    mu, std = xp.mean(dim=dim, keepdim=True), xp.std(dim=dim, keepdim=True)
    xp = (xp - mu) / std
    
    target = xp.T.to(device)

    alpha = 1e0

    g = alpha * mrpro.operators.functionals.L1NormViewAsReal(divide_by_n=divide_by_n).to(device)
    l2_squared = mrpro.operators.functionals.L2NormSquared(target=target,divide_by_n=divide_by_n).to(device)
    f = l2_squared @ U_op
    coefficients, = mrpro.algorithms.optimizers.pgd(f, g, initial_value=torch.zeros_like(xp.T).to(device),  max_iterations=64, backtrack_factor=0.8)
    
    image_approx, = U_op(coefficients)
    image_approx = image_approx.T
    image_approx = image_approx * std.to(device) + mu.to(device)
    image_approx = rearrange(image_approx, " time (slices ny nx)  -> time slices ny nx", ny=ny, nx=nx, slices=nslices, time=ntime)

    return image_approx, U_op, coefficients

def update_svd(x, U_op,map=None,weight_l1=1.,weight_l2=1., divide_by_n=False):
    """Recompute the coefficients of new images in a fixed basis U_op.

    Args:
        x: dynamic images (time, ..., ny, nx).
        U_op: basis from ``compute_svd``.
        map: optional (ny, nx) voxel-wise weight for the L1 penalty.
        weight_l1, weight_l2: weights of the L1 penalty and L2 data term.
        divide_by_n: normalize the functionals by the number of elements.

    Returns:
        image_approx: approximation (time, 1, 1, 1, ny, nx), and the coefficients.
    """
    x = x.squeeze().unsqueeze(0)
    x = rearrange(x, "1 time ny nx  -> time 1 ny nx").to(x.device)
    
    ntime, nslices,ny, nx = x.shape 
    xp = rearrange(x, "time slices ny nx  -> time (slices ny nx)")
    
    mu, std = xp.mean(dim=(-1,), keepdim=True), xp.std(dim=(-1,), keepdim=True)
    xp = (xp - mu) / std
    
    target = xp.T.to(x.device)

    
    # if isinstance(alpha, (int, float)):
    #     g = alpha * mrpro.operators.functionals.L1NormViewAsReal(divide_by_n=divide_by_n).to(x.device)
    # else:
    # print('alpha before reshape ', alpha.shape)
    # print('alpha reshaped ', alpha.shape)
    # print('(xp.T) shape ', xp.T.shape)
    if map is not None:
        g = weight_l1 * mrpro.operators.functionals.L1NormViewAsReal(divide_by_n=divide_by_n,
                                                        weight=rearrange(map, 'ny nx -> (ny nx) 1')).to(x.device)
    else:
        g = weight_l1 * mrpro.operators.functionals.L1NormViewAsReal(divide_by_n=divide_by_n).to(x.device)
    
    l2_squared = (weight_l2*0.5) * mrpro.operators.functionals.L2NormSquared(target=target,divide_by_n=divide_by_n).to(x.device)
    f = l2_squared @ U_op
    coefficients, = mrpro.algorithms.optimizers.pgd(f, g, initial_value=torch.zeros_like(xp.T).to(x.device),  max_iterations=64, backtrack_factor=0.8)
    
    image_approx, = U_op(coefficients)
    image_approx = image_approx.T
    image_approx = image_approx * std.to(x.device) + mu.to(x.device)
    image_approx = rearrange(image_approx, " time (slices ny nx)  -> time slices ny nx", ny=ny, nx=nx, slices=nslices, time=ntime)

    image_approx = image_approx.squeeze().unsqueeze(1).unsqueeze(1).unsqueeze(1)
    return image_approx, coefficients


def compute_svd_trainable_lambda(x_full, mask, num_iters=100, lr=1e-2, divide_by_n=False):
    """Experimental: like ``compute_svd``, then learn the L1 weight lambda with Adam.

    Returns image_approx, U_op, coefficients and the learned lambda_param.
    """
    device = x_full.device
    x_small = mask.apply(x_full).squeeze().unsqueeze(0)
    x_small = rearrange(x_small, "1 time ny nx -> time 1 ny nx").to(device)
    ntime, nslices, ny, nx = x_small.shape
    xp = rearrange(x_small, "time slices ny nx -> time (slices ny nx)")
    mu, std = xp.mean(dim=(-1,), keepdim=True), xp.std(dim=(-1,), keepdim=True)
    xp_norm = (xp - mu) / (std + 1e-8)
    U, S, Vh = torch.linalg.svd(xp_norm)
    U_op = mrpro.operators.EinsumOp(U).to(device)

    # Prepare data for coefficients (full data)
    x_full = x_full.squeeze().unsqueeze(0)
    x_full = rearrange(x_full, "1 time ny nx -> time 1 ny nx").to(device)
    ntime, nslices, ny, nx = x_full.shape
    xp = rearrange(x_full, "time slices ny nx -> time (slices ny nx)")
    mu, std = xp.mean(dim=(-1,), keepdim=True), xp.std(dim=(-1,), keepdim=True)
    xp_norm = (xp - mu) / (std + 1e-8)
    target = xp_norm.T.to(device)  # shape: [ny*nx, time]
    
    alpha = 1e0

    g = alpha * mrpro.operators.functionals.L1NormViewAsReal(divide_by_n=divide_by_n).to(device)
    l2_squared = mrpro.operators.functionals.L2NormSquared(target=target,divide_by_n=divide_by_n).to(device)
    f = l2_squared @ U_op
    coefficients, = mrpro.algorithms.optimizers.pgd(f, g, initial_value=torch.zeros_like(xp.T).to(device), 
                                                    max_iterations=64, backtrack_factor=0.8)
    
    # Initialize coefficients and lambda
    # coefficients = nn.Parameter(coefficients)
    lambda_param = nn.Parameter(torch.tensor(1., device=device), requires_grad=True)  # Scalar

    optimizer = optim.Adam([lambda_param], lr=lr)
    loss_history = []

    for it in range(num_iters):
        optimizer.zero_grad()
        # Forward model: image_approx = U_op @ coefficients.T
        
        image_approx, = U_op(coefficients)
        # Loss: reconstruction + L1 regularization with learnable lambda
        #rec_loss = torch.mean((image_approx - target) ** 2)
        #l1_reg = lambda_param * (coefficients.abs().mean())  # mean for normalization
        
        rec_loss, = l2_squared(image_approx)
        l1_reg, = g(coefficients)
        loss = rec_loss + lambda_param * l1_reg
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if (it % 10 == 0 or it == num_iters - 1):
            print(f"Iter {it}: Loss={loss.item():.6f}, rec={rec_loss.item():.6f}, l1={l1_reg.item():.6f}, lambda={lambda_param.item():.4f}")

    # Final reconstruction
    image_approx, = U_op(coefficients)
    image_approx = image_approx.T
    image_approx = image_approx * std.to(device) + mu.to(device)
    image_approx = rearrange(image_approx, " time (slices ny nx)  -> time slices ny nx", ny=ny, nx=nx, slices=nslices, time=ntime)

    
    return image_approx, U_op, coefficients, lambda_param



def continue_svd_training(x_new, mask, U_prev=None, coefficients=None, lambda_param=None, num_iters=50, lr=1e-2, divide_by_n=False):
    """Experimental: recompute the basis on new images and keep optimizing coefficients and lambda with Adam.

    Args:
        x_new: new dynamic images.
        mask: masking object (see ``compute_svd``).
        U_prev: unused.
        coefficients, lambda_param: values to warm-start from (zeros / 1 if None).
        num_iters, lr: Adam iterations and learning rate.

    Returns image_approx, U_op, coefficients and lambda_param.
    """
    device = x_new.device
    x_small = mask.apply(x_new).squeeze().unsqueeze(0)
    x_small = rearrange(x_small, "1 time ny nx -> time 1 ny nx").to(device)
    ntime, nslices, ny, nx = x_small.shape
    xp = rearrange(x_small, "time slices ny nx -> time (slices ny nx)")
    mu, std = xp.mean(dim=(-1,), keepdim=True), xp.std(dim=(-1,), keepdim=True)
    xp_norm = (xp - mu) / (std + 1e-8)

    # Recompute SVD basis from new x
    U, S, Vh = torch.linalg.svd(xp_norm)
    U_op = mrpro.operators.EinsumOp(U).to(device)

    # Prepare target
    x_new_full = x_new.squeeze().unsqueeze(0)
    x_new_full = rearrange(x_new_full, "1 time ny nx -> time 1 ny nx").to(device)
    xp = rearrange(x_new_full, "time slices ny nx -> time (slices ny nx)")
    mu, std = xp.mean(dim=(-1,), keepdim=True), xp.std(dim=(-1,), keepdim=True)
    xp_norm = (xp - mu) / (std + 1e-8)
    target = xp_norm.T.to(device)  # [ny*nx, time]

    # Use previous coefficients and lambda if provided, else initialize
    if coefficients is None:
        coefficients = nn.Parameter(torch.zeros_like(target))
    else:
        coefficients = nn.Parameter(coefficients.detach())
    if lambda_param is None:
        lambda_param = nn.Parameter(torch.tensor(1., device=device), requires_grad=True)
    else:
        lambda_param = nn.Parameter(lambda_param.detach(), requires_grad=True)
    
    optimizer = optim.Adam([coefficients, lambda_param], lr=lr)
    loss_history = []

    alpha = 1.0
    g = alpha * mrpro.operators.functionals.L1NormViewAsReal(divide_by_n=divide_by_n).to(device)
    l2_squared = mrpro.operators.functionals.L2NormSquared(target=target,divide_by_n=divide_by_n).to(device)
    f = l2_squared @ U_op

    for it in range(num_iters):
        optimizer.zero_grad()
        image_approx, = U_op(coefficients)
        rec_loss, = l2_squared(image_approx)
        l1_reg, = g(coefficients)
        loss = rec_loss + lambda_param * l1_reg
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if (it % 10 == 0 or it == num_iters - 1):
            print(f"[Cont] Iter {it}: Loss={loss.item():.6f}, rec={rec_loss.item():.6f}, l1={l1_reg.item():.6f}, lambda={lambda_param.item():.4f}")

    image_approx, = U_op(coefficients)
    image_approx = image_approx.T
    image_approx = image_approx * std.to(device) + mu.to(device)
    image_approx = rearrange(image_approx, "time (slices ny nx) -> time slices ny nx", ny=ny, nx=nx, slices=nslices, time=ntime)

    return image_approx, U_op, coefficients, lambda_param

