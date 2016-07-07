"""Model evaluation tools."""

import numpy as np


def fd_jacobian(model, angle, p, T, lnq, perturbation=0.001):
    """Calculate Jacobians by finite differencing.
    
    Returns a reference calculation (direct model output) and approximated
    Jacobians for T and lnq. Only one angle at a time is possible.
    """
    reference = model(p=p, T=T, lnq=lnq, angles=angle)
    test_reference = model.forward(angle, p=p, T=T, lnq=lnq)
    assert np.isclose(reference.fwd, test_reference)
    # Determine perturbation
    perturb_T = perturbation * np.min(np.abs(T))
    perturb_lnq = perturbation * np.min(np.abs(lnq))
    # Calculate temperature Jacobian with centered difference
    fd_dT = np.zeros_like(reference.dT)
    for i in range(len(T)):
        TT = T.copy()
        TT[i] += perturb_T
        upper = model.forward(angle, p=p, T=TT, lnq=lnq)
        TT[i] -= 2*perturb_T
        lower = model.forward(angle, p=p, T=TT, lnq=lnq)
        fd_dT[i] = (upper - lower) / (2*perturb_T)
    # Calculate humidity Jacobian with centered difference
    fd_dlnq = np.zeros_like(reference.dlnq)
    for i in range(len(T)):
        lnqq = lnq.copy()
        lnqq[i] += perturb_lnq
        upper = model.forward(angle, p=p, T=T, lnq=lnqq)
        lnqq[i] -= 2*perturb_lnq
        lower = model.forward(angle, p=p, T=T, lnq=lnqq)
        fd_dlnq[i] = (upper - lower) / (2*perturb_lnq)
    return reference, fd_dT, fd_dlnq

