"""Build the operators used for the image reconstruction (coil sensitivities, DCF, Fourier)."""
import torch
import mrpro


def compute_csm(kdata,smoothing_width=11,device='cpu'):
    """Estimate coil sensitivity maps (Walsh method) from a density-compensated adjoint recon.

    Args:
        kdata: KData used for the estimation (usually all readouts, no dynamics).
        smoothing_width: smoothing kernel size for the Walsh method.
        device: torch device.

    Returns:
        csm_operator: operator applying the coil sensitivities.
        img_size: (ny, nx) of the images.
    """

    #idx_csm = torch.as_tensor(torch.arange(0,100))[None,:]
    #kdata_csm = kdata.split_k1_into_other(idx_csm, other_label='repetition')
    #kdata = kdata_csm
    kdata = kdata.to(device)
    dcf_operator = mrpro.data.DcfData.from_traj_voronoi(kdata.traj).as_operator()
    dcf_operator = dcf_operator.to(device)
    
    fourier_operator = mrpro.operators.FourierOp.from_kdata(kdata).to(device)
    fourier_operator = dcf_operator @ fourier_operator
    adjoint_operator = fourier_operator.H
    
    
    img_coilwise = mrpro.data.IData.from_tensor_and_kheader(*adjoint_operator(kdata.data), kdata.header)
    csm = mrpro.data.CsmData.from_idata_walsh(img_coilwise, smoothing_width=smoothing_width)
    # csm = mrpro.data.CsmData.from_kdata_inati(kdata, 
    #                                           smoothing_width=smoothing_width)
    csm_operator = csm.as_operator()
    img_size = csm.data.shape[-2:]
    
    return csm_operator, img_size


def construct_reconstruction_operators(kdata_dynamic, kdata, device: str):
    """Create density compensation, Fourier, and normal-equation operators.

    Args:
        kdata_dynamic: KData split into dynamics (see ``build_dynamic_kdata``).
        kdata: full KData, used to estimate the coil sensitivities.
        device: torch device.

    Returns:
        dcf_op: density compensation operator W.
        dcf_sqrt_op: operator applying sqrt(W).
        acquisition_op: A = F S (Fourier after coil sensitivities).
        AHWA_op: normal operator A^H W A, used by CG.
        rhs: right-hand side A^H W y.
        img_size: (ny, nx) of the images.
    """
    # Coil sensitivities are estimated once from full k-space data.
    with torch.no_grad():
        csm_op, img_size = compute_csm(kdata, smoothing_width=15, device=device)
    csm_op.to(device)
    torch.cuda.empty_cache()
    
    dcf = mrpro.data.DcfData.from_traj_voronoi(kdata_dynamic.traj)
    dcf_op = dcf.as_operator()
    dcf_sqrt_op = mrpro.operators.DensityCompensationOp(torch.sqrt(dcf.data))

    fourier_op = mrpro.operators.FourierOp.from_kdata(kdata_dynamic).to(device)

    # A = F S, and AHWA_op is the weighted normal operator used by CG.
    acquisition_op = fourier_op @ csm_op
    AHWA_op = (dcf_sqrt_op @ acquisition_op).gram
    (rhs,) = (acquisition_op.H @ dcf_op)(kdata_dynamic.data)

    return dcf_op, dcf_sqrt_op, acquisition_op, AHWA_op, rhs, img_size
