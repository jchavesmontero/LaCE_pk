import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import emcee
from pyDOE2 import lhs


class FitPk(object):
    """Fit measured P3D with Arinyo model."""

    def __init__(
        self,
        data,
        model,
        fit_type="p3d",
        fit_k_Mpc_3d_max=10,
        noise_floor_3d=0.05,
        min_counts_3d=0,
        err_p3d=None,
        fit_k_Mpc_1d_max=10,
        noise_floor_1d=0.05,
        err_p1d=None,
        priors=None,
        smooth_mu=True,
    ):
        """Setup P3D flux power model and measurement.
        Inputs:
         - data: measured P3D
         - model: theoretical model for 3D flux power
         - fit_k_Mpc_3d_max: fit only modes with k_Mpc_3d < fit_k_Mpc_3d_max
         - noise_floor: down-weight high-k modes in fit"""

        # store data and model
        self.data = data
        self.model = model
        self.noise_floor_3d = noise_floor_3d
        self.priors = priors
        self.smooth_mu = smooth_mu

        # p3d or both
        self.fit_type = fit_type

        # identify bins included in fit
        k_Mpc_3d = self.data["k_Mpc_3d"]

        # first, consider only bins that have been measured precisely (counts>min_counts)
        # and bins that have k_Mpc < fit_k_Mpc_max
        good_3d = (self.data["counts_3d"] > min_counts_3d) & (
            self.data["k_Mpc_3d"] < fit_k_Mpc_3d_max
        )
        self.fit_bins_3d = np.array(good_3d)

        # relative errors with noise floor
        if err_p3d is not None:
            self.estimate_err_pkmu(
                err_p3d, kmax=fit_k_Mpc_3d_max, noise_floor_3d=noise_floor_3d
            )
            self.err_p3d[~good_3d] = np.inf
        else:
            rel_err = np.empty_like(self.data["counts_3d"])
            rel_err[good_3d] = (
                np.sqrt(2.0 / self.data["counts_3d"][good_3d]) + noise_floor_3d
            )
            self.err_p3d = rel_err * self.data["p3d_Mpc"]
            self.err_p3d[~good_3d] = np.inf

        if self.smooth_mu:
            self.data["smooth_mu"] = np.zeros_like(self.data["mu"])
            for ii in range(self.data["mu"].shape[1]):
                self.data["smooth_mu"][:, ii] = np.nanmedian(self.data["mu"][:, ii])

        # same for p1d
        if fit_type == "both":
            k_Mpc_1d = self.data["k_Mpc_1d"]
            good_1d = k_Mpc_1d < fit_k_Mpc_1d_max
            self.fit_bins_1d = np.array(good_1d)

            # relative errors with noise floor
            if err_p1d is not None:
                self.estimate_err_p1d(
                    err_p1d, kmax=fit_k_Mpc_1d_max, noise_floor_1d=noise_floor_1d
                )
                self.err_p1d[~good_1d] = np.inf
            else:
                rel_err = np.empty_like(self.data["k_Mpc_1d"])
                rel_err[good_1d] = noise_floor_1d
                rel_err[~good_1d] = np.inf
                self.err_p1d = rel_err * self.data["p1d_Mpc"]

    def get_model_3d(self, parameters={}):
        """Model for the 3D flux power spectrum"""

        # identify (k,mu) for bins included in fit
        k = self.data["k_Mpc_3d"][self.fit_bins_3d]
        z = self.data["z"]
        if self.smooth_mu:
            mu = self.data["smooth_mu"][self.fit_bins_3d]
        else:
            mu = self.data["mu"][self.fit_bins_3d]

        return self.model.P3D_Mpc(z, k, mu, parameters)

    def get_model_1d(
        self,
        parameters={},
        k_perp_min=0.001,
        k_perp_max=60,
        n_k_perp=40,
    ):
        """Model for the 1D flux power spectrum"""
        res = self.model.P1D_Mpc(
            self.data["z"],
            self.data["k_Mpc_1d"][self.fit_bins_1d],
            parameters=parameters,
            k_perp_min=k_perp_min,
            k_perp_max=k_perp_max,
            n_k_perp=n_k_perp,
        )
        return res

    def estimate_err_pkmu(self, sigma_pkmu, order=2, kmax=10, noise_floor_3d=0.05):
        """Returns P3D(k, mu) errors estimated from simulation
        - order: polinomial fit order"""

        self.err_p3d = np.zeros_like(self.data["p3d_Mpc"])
        self.fit_epk3d = np.zeros((self.data["k_Mpc_3d"].shape[1], order + 1))
        for ii in range(self.data["k_Mpc_3d"].shape[1]):
            _ = np.isfinite(self.data["k_Mpc_3d"][:, ii]) & (
                self.data["k_Mpc_3d"][:, ii] < kmax
            )
            logk = np.log10(self.data["k_Mpc_3d"][_, ii])
            self.fit_epk3d[ii, :] = np.polyfit(logk, np.log10(sigma_pkmu[_, ii]), order)
            pfit = np.poly1d(self.fit_epk3d[ii, :])

            _ = np.isfinite(self.data["k_Mpc_3d"][:, ii])
            logk = np.log10(self.data["k_Mpc_3d"][_, ii])
            noise_floor = noise_floor_3d * self.data["p3d_Mpc"][_, ii]
            self.err_p3d[_, ii] = 10 ** pfit(logk) + noise_floor

    def estimate_err_p1d(self, sigma_pk1d, order=3, kmax=40, noise_floor_1d=0.05):
        """Returns P1D(k, mu) errors estimated from simulation
        - order: polinomial fit order"""

        if "k_Mpc_1d" in self.data:
            pass
        else:
            return "No k_Mpc_1d key in data"

        self.err_p1d = np.zeros_like(self.data["p1d_Mpc"])
        _ = (
            (self.data["k_Mpc_1d"] > 0)
            & (self.data["k_Mpc_1d"] < kmax)
            & (sigma_pk1d != 0)
        )
        logk = np.log10(self.data["k_Mpc_1d"][_])
        self.fit_epk1d = np.polyfit(logk, np.log10(sigma_pk1d[_]), order)
        pfit = np.poly1d(self.fit_epk1d)

        _ = self.data["k_Mpc_1d"] > 0
        logk = np.log10(self.data["k_Mpc_1d"][_])
        noise_floor = noise_floor_1d * self.data["p1d_Mpc"][_]
        self.err_p1d[_] = 10 ** pfit(logk) + noise_floor
        self.err_p1d[0] = self.err_p1d[1] * 2

    def get_chi2(self, parameters={}, return_npoints=False):
        """Compute chi squared for a particular P3D model.
        - parameters: dictionary with parameters to use
        - return_points: return number of data points used"""

        # get P3D measurement for bins included in fit
        data = self.data["p3d_Mpc"][self.fit_bins_3d]

        # compute model for these wavenumbers
        theory = self.get_model_3d(parameters=parameters)

        # get absolute error
        error = self.err_p3d[self.fit_bins_3d]

        # compute chi2
        chi2 = np.sum(((data - theory) / error) ** 2) / np.sum(self.fit_bins_3d)

        if self.fit_type == "both":
            # get P1D measurement for bins included in fit
            data = self.data["p1d_Mpc"][self.fit_bins_1d]

            # compute model for these wavenumbers
            theory = self.get_model_1d(parameters=parameters)

            # compute absolute error
            error = self.err_p1d[self.fit_bins_1d]

            # compute chi2
            chi2_p1d = np.sum(((data - theory) / error) ** 2) / np.sum(self.fit_bins_1d)
            chi2 += chi2_p1d

        if return_npoints:
            npoints = len(data)
            return chi2, npoints
        else:
            return chi2

    def get_log_like(self, parameters={}):
        """Compute log likelihood (ignoring determinant)"""

        return -0.5 * self.get_chi2(parameters, return_npoints=False)

    def _log_like(self, values, parameter_names):
        """Function passed to scipy minimizer:
        - values: array of initial values of parameters
        - parameter_names: should have same size than values above"""

        Np = len(values)
        assert Np == len(parameter_names), "inconsistent inputs in _log_like"

        # create dictionary with parameters that models can understand
        # also check if all parameters within priors
        out_priors = 0
        parameters = {}
        for ii in range(Np):
            parameters[parameter_names[ii]] = values[ii]
            if self.priors is not None:
                if (values[ii] < self.priors[parameter_names[ii]][0]) | (
                    values[ii] > self.priors[parameter_names[ii]][1]
                ):
                    out_priors += 1

        if out_priors != 0:
            return -np.inf
        else:
            return self.get_log_like(parameters)

    def maximize_likelihood(self, parameters):
        """Run minimizer and return best-fit values"""

        ndim = len(parameters)
        names = list(parameters.keys())
        values = np.array(list(parameters.values()))

        # generate random initial value
        ini_values = values * (1 + 0.05 * np.random.randn(ndim))

        # lambda function to minimize
        minus_log_like = lambda *args: -self._log_like(*args)

        # get max likelihood values
        results = minimize(
            minus_log_like,
            ini_values,
            args=(names),
            method="Nelder-Mead",
            options={"maxiter": 10000, "fatol": 1e-2},
        )

        # update parameters dictionary
        best_fit_parameters = {}
        for ip in range(ndim):
            best_fit_parameters[names[ip]] = results.x[ip]

        return results, best_fit_parameters

    def log_like_emcee(self, params):
        """Compute log likelihood"""

        out_priors = 0
        parameters = {}
        for ii in range(self.ndim):
            parameters[self.names[ii]] = params[ii]
            if (params[ii] < self.priors[self.names[ii]][0]) | (
                params[ii] > self.priors[self.names[ii]][1]
            ):
                out_priors += 1

        if out_priors != 0:
            return -np.inf
        else:
            return -0.5 * self.get_chi2(parameters, return_npoints=False)

    def explore_likelihood(
        self,
        parameters,
        seed=0,
        nwalkers=20,
        nsteps=100,
        nburn=0,
        plot=False,
        attraction=0.3,
    ):
        """Run emcee and return best-fitting chain"""

        self.ndim = len(parameters)
        self.names = list(parameters.keys())
        values = np.array(list(parameters.values()))

        # generate random initial value
        ini_values = init_chains(
            parameters,
            nwalkers,
            self.priors,
            seed=seed,
            attraction=attraction,
        )

        sampler = emcee.EnsembleSampler(
            nwalkers,
            self.ndim,
            self.log_like_emcee,
        )
        sampler.run_mcmc(ini_values, nsteps)

        chain = sampler.get_chain(discard=nburn)
        lnprob = sampler.get_log_prob(discard=nburn)

        # print(chain.shape, lnprob.shape)
        # (nsteps, nwalkers, ndim) (nsteps, nwalkers)

        if plot:
            for ii in range(nwalkers):
                plt.plot(lnprob[:, ii])
            plt.show()

        minval = np.nanmedian(lnprob, axis=0)
        minval = np.nanmax(minval) - 5
        keep = purge_chains(lnprob, minval=minval)

        chain = chain[:, keep, :].reshape(-1, self.ndim)
        lnprob = lnprob[:, keep].reshape(-1)

        return lnprob, chain


def purge_chains(ln_prop_chains, nsplit=5, abs_diff=5, minval=-1000):
    """Purge emcee chains that have not converged"""
    # split each walker in nsplit chunks
    split_arr = np.array_split(ln_prop_chains, nsplit, axis=0)
    # compute median of each chunck
    split_med = []
    for ii in range(nsplit):
        split_med.append(split_arr[ii].mean(axis=0))
    # (nwalkers, nchucks)
    split_res = np.array(split_med).T
    # compute median of chunks for each walker ()
    split_res_med = split_res.mean(axis=1)

    # step-dependence convergence
    # check that average logprob does not vary much with step
    # compute difference between chunks and median of each chain
    keep1 = (np.abs(split_res - split_res_med[:, np.newaxis]) < abs_diff).all(axis=1)
    # total-dependence convergence
    # check that average logprob is close to minimum logprob of all chains
    # check that all chunks are above a target minimum value
    keep2 = (split_res > minval).all(axis=1)

    # combine both criteria
    keep = keep1 & keep2

    return keep


def init_chains(
    parameters,
    nwalkers,
    bounds,
    criterion="c",
    seed=0,
    attraction=1,
    min_attraction=0.05,
):

    parameter_names = list(parameters.keys())
    parameter_values = np.array(list(parameters.values()))
    nparams = len(parameter_names)
    design = lhs(
        nparams,
        samples=nwalkers,
        criterion=criterion,
        random_state=seed,
    )

    if attraction > 1:
        attraction = 1
    elif attraction < min_attraction:
        attraction = min_attraction

    for ii in range(nparams):
        buse = bounds[parameter_names[ii]]
        lbox = (buse[1] - buse[0]) * attraction

        # design sample using lh as input, attracted to best-fitting solution
        design[:, ii] = (
            lbox * (design[:, ii] - 0.5) + buse[0] * attraction + parameter_values[ii]
        )

        # make sure that samples do not get out of prior range
        _ = design[:, ii] >= buse[1]
        design[_, ii] -= lbox * 0.999
        _ = design[:, ii] <= buse[0]
        design[_, ii] += lbox * 0.999

    return design
