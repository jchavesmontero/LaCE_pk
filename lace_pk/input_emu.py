import numpy as np


def get_params_4_emu(data, n_std=1):
    means = np.mean(data, axis=0)
    cov = np.cov(data.T)
    stds = np.sqrt(np.diag(cov))
    corrs = []
    for ii in range(data.shape[1]):
        for jj in range(ii):
            corrs.append(cov[ii, jj] / np.sqrt(cov[ii, ii] * cov[jj, jj]))
    corrs = np.array(corrs)
    return means, stds, corrs


def get_input_emulator(folder_input, ntot, file_out):
    for ii in range(ntot):
        file = (
            folder_input
            + "temp_data/fits/fit_indsim_"
            + str(ii)
            + "_kmax3d_5_noise3d_0.075_kmax1d_5_noise1d_0.01.npz"
        )
        fil = np.load(file)
        par_chain = fil["chain"].copy()
        par_chain[:, 0] = np.abs(par_chain[:, 0])
        bp = par_chain[pos]
        means, stds, corrs = get_params_4_emu(par_chain)
        np.savez(file_out, means=means, stds=stds, corrs=corrs)
