import numpy as np
import scipy
from lace.cosmo import camb_cosmo
from scipy.integrate import simpson


def get_nmod(k, dk, Lbox):
    Vs = 4 * np.pi**2 * k**2 * dk * (1 + 1 / 12 * (dk / k) ** 2)
    kf = 2 * np.pi / Lbox
    Vk = kf**3
    Nk = Vs / Vk
    return Nk


def get_linP_interp(cosmo, zs, camb_results, camb_kmax_Mpc=30):
    """Ask CAMB for an interpolator of linear power."""

    if camb_results is None:
        camb_results = camb_cosmo.get_camb_results(
            cosmo, zs=zs, camb_kmax_Mpc=camb_kmax_Mpc
        )

    # get interpolator from CAMB
    # meaning of var1 and var2 here
    # https://camb.readthedocs.io/en/latest/transfer_variables.html#transfer-variables
    linP_interp = camb_results.get_matter_power_interpolator(
        nonlinear=False,
        var1=8,
        var2=8,
        hubble_units=False,
        k_hunit=False,
        log_interp=True,
    )

    return linP_interp


class ArinyoModel(object):
    """Flux power spectrum model from Arinyo-i-Prats et al. (2015)"""

    def __init__(
        self,
        cosmo,
        zs,
        camb_results=None,
        default_bias=-0.18,
        default_beta=1.3,
        default_d1_q1=0.4,
        default_d1_q2=0.0,
        default_d1_kvav=0.58,
        default_d1_av=0.29,
        default_d1_bv=1.55,
        default_d1_kp=10.5,
        camb_kmax_Mpc=100.0,
    ):
        """Set up flux power spectrum model.
        Inputs:
         - cosmo: CAMB params object defining cosmology
         - zs: redshifts for which we want predictions
         - camb_results: if already computed, it can be used
         - default_bias: starting value for the flux bias
         - default_beta: RSD parameter for the flux
         - default_d1_{}: parameters in non-linear model
         - default_d1_kvav: units (1/Mpc)^(av)
         - default_d1_kp: units 1/Mpc"""

        # get a linear power interpolator
        self.linP_interp = get_linP_interp(
            cosmo, zs, camb_results, camb_kmax_Mpc=camb_kmax_Mpc
        )

        # store bias parameters
        self.default_bias = default_bias
        self.default_beta = default_beta
        self.default_d1_q1 = default_d1_q1
        self.default_d1_q2 = default_d1_q2
        self.default_d1_kvav = default_d1_kvav
        self.default_d1_av = default_d1_av
        self.default_d1_bv = default_d1_bv
        self.default_d1_kp = default_d1_kp

    def linP_Mpc(self, z, k_Mpc):
        """get linear power at input redshift and wavenumber"""

        return self.linP_interp.P(z, k_Mpc)

    def P3D_Mpc(self, z, k, mu, parameters={}):
        """Compute model for 3D flux power (units of Mpc^3)"""

        # evaluate linear power at input (z,k)
        linP = self.linP_Mpc(z, k)

        # model large-scales biasing for delta_flux(k)
        lowk_bias = self.lowk_biasing(mu, parameters)

        # model small-scales correction (D_NL in Arinyo-i-Prats 2015)
        D_NL = self.small_scales_correction(z, k, mu, parameters)

        return linP * lowk_bias**2 * D_NL

    def lowk_biasing(self, mu, parameters={}):
        """Compute model for the large-scales biasing of delta_flux"""

        # extract bias and beta from dictionary with parameter values
        if "bias" in parameters:
            bias = parameters["bias"]
        else:
            bias = self.default_bias

        if "beta" in parameters:
            beta = parameters["beta"]
        else:
            beta = self.default_beta

        linear_rsd = 1 + beta * mu**2

        return bias * linear_rsd

    def small_scales_correction(self, z, k, mu, parameters={}):
        """Compute small-scales correction to delta_flux biasing"""

        # extract parameters from dictionary of parameter values
        if "d1_q1" in parameters:
            d1_q1 = parameters["d1_q1"]
        else:
            d1_q1 = self.default_d1_q1
        if "d1_q2" in parameters:
            d1_q2 = parameters["d1_q2"]
        else:
            d1_q2 = self.default_d1_q2
        if "d1_kvav" in parameters:
            d1_kvav = parameters["d1_kvav"]
        else:
            d1_kvav = self.default_d1_kvav
        if "d1_av" in parameters:
            d1_av = parameters["d1_av"]
        else:
            d1_av = self.default_d1_av
        if "d1_bv" in parameters:
            d1_bv = parameters["d1_bv"]
        else:
            d1_bv = self.default_d1_bv
        if "d1_kp" in parameters:
            d1_kp = parameters["d1_kp"]
        else:
            d1_kp = self.default_d1_kp

        # get linear power (required to get delta squared)
        linP = self.linP_Mpc(z, k)
        delta2 = (1 / (2 * (np.pi**2))) * k**3 * linP
        nonlin = d1_q1 * delta2 + d1_q2 * (delta2**2)

        d1 = np.exp(
            nonlin * (1 - ((k**d1_av) / d1_kvav) * (mu**d1_bv)) - (k / d1_kp) ** 2
        )

        return d1

    def _P3D_kperp2(self, z, ln_k_perp, k_par, parameters={}):
        """Function to be integrated to compute P1D"""

        # compute k and mu from ln_k_perp and k_par
        k_perp = np.exp(ln_k_perp)
        k = np.sqrt(k_par**2 + k_perp**2)
        mu = k_par / k

        # get P3D
        p3d = self.P3D_Mpc(z, k, mu, parameters)

        return (1 / (2 * np.pi)) * k_perp**2 * p3d

    def _P1D_lnkperp(self, z, ln_k_perp, kpars, parameters={}):
        """Compute P1D by integrating P3D in terms of ln(k_perp)"""

        # get interval for integration
        dlnk = ln_k_perp[1] - ln_k_perp[0]

        # for each value of k_par, integrate P3D over ln(k_perp) to get P1D
        p1d = np.empty_like(kpars)
        for i in range(kpars.size):
            # get function to be integrated
            p3d_fix_k_par = self._P3D_kperp2(z, ln_k_perp, kpars[i], parameters)
            # perform numerical integration
            p1d[i] = simpson(p3d_fix_k_par, ln_k_perp, dx=dlnk)

        return p1d

    def _P1D_lnkperp_fast(self, z, ln_k_perp, kpars, parameters={}):
        """Compute P1D by integrating P3D in terms of ln(k_perp)"""

        # get interval for integration
        dlnk = ln_k_perp[1] - ln_k_perp[0]

        # get function to be integrated
        # it is equivalent of the inner loop of _P1D_lnkperp
        k_perp = np.exp(ln_k_perp)
        k = np.sqrt(kpars[np.newaxis, :] ** 2 + k_perp[:, np.newaxis] ** 2)
        mu = kpars[np.newaxis, :] / k
        k = k.swapaxes(0, 1)
        mu = mu.swapaxes(0, 1)

        fact = (1 / (2 * np.pi)) * k_perp[:, np.newaxis] ** 2
        fact = fact.swapaxes(0, 1)
        p3d_fix_k_par = self.P3D_Mpc(z, k, mu, parameters) * fact

        # perform numerical integration
        p1d = simpson(p3d_fix_k_par, ln_k_perp, dx=dlnk, axis=1)

        return p1d

    def P1D_Mpc(
        self,
        z,
        k_par,
        k_perp_min=0.001,
        k_perp_max=100,
        n_k_perp=99,
        parameters={},
    ):
        """Returns P1D for specified values of k_par, with the option to
        specify values of k_perp to be integrated over
        - k_par: array or list of values for which P1D is to be computed
        - k_perp_min: lower bound of integral
        - k_perp_max: upper bound of integral
        - n_k_perp: number of points in integral (Simpson's rule requires
                odd number, otherwise use trapezoidal)"""

        ln_k_perp = np.linspace(np.log(k_perp_min), np.log(k_perp_max), n_k_perp)

        p1d = self._P1D_lnkperp_fast(z, ln_k_perp, k_par, parameters)

        return p1d

    def rel_err_P1d(
        self,
        z,
        k_par,
        Lbox,
        res_los,
        nskewers,
        n_k_perp=1000,
        parameters={},
    ):
        n_k_par = k_par.shape[0]

        # array of k_perp that fit within simulation box (important!)
        nk_max = int(Lbox / res_los / 2)
        k_perp_min = 2 * np.pi / Lbox
        k_perp_max = k_perp_min * nk_max
        ln_k_perp = np.linspace(np.log(k_perp_min), np.log(k_perp_max), n_k_perp)
        k_perp = np.exp(ln_k_perp)

        # get p3d and number of modes as a function of k_par and k_per
        p3d = np.zeros((n_k_par, n_k_perp))
        nmod = np.zeros((n_k_par, n_k_perp))
        for ii in range(n_k_perp):
            k = np.sqrt(k_par**2 + k_perp[ii] ** 2)
            dk = k[1:] - k[:-1]
            dk = np.concatenate([dk, np.atleast_1d(dk[-1])])
            mu = k_par / k
            p3d[:, ii] = self.P3D_Mpc(z, k, mu)
            nmod[:, ii] = get_nmod(k, dk, Lbox)

        ## estimate error from cosmic variance (new, JCM)

        # sigma_p3d = np.sqrt(2/Nmodes) * P
        sigma_p3d = np.sqrt(2 / nmod) * p3d

        # p1d = 1/(2*pi) int_kmin^kmax dkper kper p3d
        # I am not sure whether should be pi or 2*pi, depends on the convention?
        # we use 2*pi above so let's go with that
        p1d = self.P1D_Mpc(z, k_par)

        # in practise, we only need to evaluate the derivatives and error at kper=kmax
        # dp1d/dp3d = dP1d/dkpar (dkpar/dp3d)_kmin^kmax
        # dp1d_dkpar = (1/pi) int_kmin^kmax dkper kper dp3d/dkpar
        dp3d_dkpar = np.gradient(p3d, k_par, axis=0)
        dp1d_dkpar = simpson(dp3d_dkpar * k_perp, k_perp, axis=1) / (2 * np.pi)
        dp1d_dp3d = dp1d_dkpar / dp3d_dkpar[:, -1]

        # err_p1d = |dp1d/dp3d| * sigma_p3d_kmin^kmax
        sigma_p1d = np.abs(dp1d_dp3d) * sigma_p3d[:, -1]

        # err_p1d__p1d
        sigma_p1d__p1d = sigma_p1d / p1d

        ## estimate error from sampling variance (Zhan et al. 2005)
        sampling_sigma__p1d = (k_par[:] * 0 + 1) / np.sqrt(nskewers)

        ## combine both
        sigma_both = np.sqrt(sigma_p1d__p1d**2 + sampling_sigma__p1d**2)

        return sigma_both, sigma_p1d__p1d, sampling_sigma__p1d
