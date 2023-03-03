import numpy as np
import scipy
import matplotlib.pyplot as plt
from lace.cosmo import camb_cosmo
from scipy.integrate import simpson


def kpar(k, mu):
    return k * mu


def kper(k, mu):
    return k * np.sqrt(1 - mu**2)


def dkpar_dk(k, mu):
    return mu


def dkpar_dmu(k, mu):
    return k


def dkper_dk(k, mu):
    return np.sqrt(1 - mu**2)


def dkper_dmu(k, mu):
    return -k * mu / np.sqrt(1 - mu**2)


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
        lowk_bias = self.kaiser(mu, parameters)

        # model small-scales correction (D_NL in Arinyo-i-Prats 2015)
        D_NL = self.DNL(z, k, mu, parameters)

        return linP * lowk_bias**2 * D_NL

    def kaiser(self, mu, parameters={}):
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

    def DNL1(self, z, k, parameters={}):
        if "d1_q1" in parameters:
            d1_q1 = parameters["d1_q1"]
        else:
            d1_q1 = self.default_d1_q1
        if "d1_q2" in parameters:
            d1_q2 = parameters["d1_q2"]
        else:
            d1_q2 = self.default_d1_q2

        # get linear power (required to get delta squared)
        linP = self.linP_Mpc(z, k)
        delta2 = (1 / (2 * (np.pi**2))) * k**3 * linP
        nonlin = d1_q1 * delta2 + d1_q2 * (delta2**2)

        return nonlin

    def DNL2(self, k, mu, parameters={}):
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

        return 1 - ((k**d1_av) / d1_kvav) * (mu**d1_bv)

    def DNL3(self, k, parameters={}):
        if "d1_kp" in parameters:
            d1_kp = parameters["d1_kp"]
        else:
            d1_kp = self.default_d1_kp

        return -((k / d1_kp) ** 2)

    def DNL(self, z, k, mu, parameters={}):
        nonlin = self.DNL1(z, k, parameters)
        vel = self.DNL2(k, mu, parameters)
        pres = self.DNL3(k, parameters)

        return np.exp(nonlin * vel + pres)

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

    ########################
    # def p1d_4_p3d(self, z, k, mu, parameters={}):
    #     pre_fact = 1 / 2 / np.pi

    #     int_k = np.zeros((k.shape[0], mu.shape[0]))
    #     int_mu = np.zeros((k.shape[0], mu.shape[0]))
    #     for ii in range(k.shape[0]):
    #         int_k[ii] = k[ii] * (1 - mu**2) * self.P3D_Mpc(z, k[ii], mu, parameters)
    #         int_mu[ii] = -(k[ii] ** 2) * mu * self.P3D_Mpc(z, k[ii], mu, parameters)

    #     p1d_k = simpson(int_k, x=k, axis=0)
    #     p1d_mu = simpson(int_mu, x=mu, axis=1)

    #     return

    # def dp1d_dp3d(
    #     self,
    #     z,
    #     k_par,
    #     k_perp_min=0.001,
    #     k_perp_max=100,
    #     n_k_perp=99,
    #     parameters={},
    # ):
    #     ln_k_perp = np.linspace(np.log(k_perp_min), np.log(k_perp_max), n_k_perp)
    #     k_perp = np.exp(ln_k_perp)
    #     k = np.sqrt(k_par**2 + k_perp**2)
    #     mu = k_par / k

    #     dp1d_dkpar = self.dp1d_dkpar(z, k, mu, parameters)
    #     dp3d_dkpar = self.dp3d_dkpar(z, k, mu, parameters)
    #     return dp1d_dkpar / dp3d_dkpar

    def dp1d_dkpar(self, z, k, mu, parameters={}):
        res = self.dp1d_dk(z, k, mu, parameters) / dkpar_dk(k, mu) + self.dp1d_dmu(
            z, k, mu, parameters
        ) / dkpar_dmu(k, mu)

        return res

    def dp1d_dk(self, z, k, mu, parameters={}):
        pre_fact = -1 / 2 / np.pi
        int_mu = mu * (
            2 * k * self.P3D_Mpc(z, k, mu, parameters)
            + k**2 * self.dp3d_dk(z, k, mu, parameters)
        )
        sint_mu = simpson(int_mu, x=mu)

        return pre_fact * sint_mu

    def dp1d_dmu(self, z, k, mu, parameters={}):
        pre_fact = 1 / 2 / np.pi
        int_k = k * (
            -2 * mu * self.P3D_Mpc(z, k, mu, parameters)
            + (1 - mu**2) * self.dp3d_dk(z, k, mu, parameters)
        )
        sint_k = simpson(int_k, x=k, axis=0)

        return pre_fact * sint_k

    def dp3d_dkpar(self, z, k, mu, parameters={}):
        res = self.dp3d_dk(z, k, mu, parameters) / dkpar_dk(k, mu) + self.dp3d_dmu(
            z, k, mu, parameters
        ) / dkpar_dmu(k, mu)
        return res

    def dp1d_dp3d(
        self,
        z,
        k_par,
        k_perp_min=0.001,
        k_perp_max=100,
        n_k_perp=99,
        parameters={},
    ):
        ln_k_perp = np.linspace(np.log(k_perp_min), np.log(k_perp_max), n_k_perp)
        k_perp = np.exp(ln_k_perp)
        k = np.sqrt(k_par**2 + k_perp**2)
        mu = k_par / k

        p3d = self.P3D_Mpc(z, k, mu, parameters)
        dp3d_dkper = self.dp3d_dkper(z, k, mu, parameters)
        # integrate in log(kper), dividing everything by kper

        f2int = 1 + p3d / dp3d_dkper / k_perp
        int_kper = simpson(f2int, x=ln_k_perp)

        # f2int = k_perp + p3d / dp3d_dkper
        # int_kper = simpson(f2int, x=k_perp)

        plt.plot(ln_k_perp, f2int, label=str(k_par))

        return int_kper / np.pi

    def dp3d_dkper(self, z, k, mu, parameters={}):
        dfdk = self.dp3d_dk(z, k, mu, parameters) / dkper_dk(k, mu)
        dfdmu = self.dp3d_dmu(z, k, mu, parameters) / dkper_dmu(k, mu)

        return dfdk + dfdmu

    def dp3d_dk(self, z, k, mu, parameters={}):
        kaiser = self.kaiser(mu, parameters)
        res1 = self.dplin_dk(z, k) * self.DNL(z, k, mu, parameters)
        res2 = self.linP_Mpc(z, k) * self.dDNL_dk(z, k, mu, parameters)
        return kaiser**2 * (res1 + res2)

    def dp3d_dmu(self, z, k, mu, parameters={}):
        if "bias" in parameters:
            bias = parameters["bias"]
        else:
            bias = self.default_bias

        if "beta" in parameters:
            beta = parameters["beta"]
        else:
            beta = self.default_beta

        pre = bias**2 * (1 + beta * mu**2) * self.linP_Mpc(z, k)

        res1 = 4 * beta * mu * self.DNL(z, k, mu, parameters)
        res2 = (1 + beta * mu**2) * self.dDNL_dmu(z, k, mu, parameters)
        return pre * (res1 + res2)

    def dDNL_dk(self, z, k, mu, parameters={}):
        pre_fact = self.DNL(z, k, mu, parameters)
        sum1 = self.DNL2(k, mu, parameters) * self.dDNL1_dk(z, k, parameters)
        sum2 = self.DNL1(z, k, parameters) * self.dDNL2_dk(k, mu, parameters)
        sum3 = self.dDNL3_dk(k, parameters)
        return pre_fact * (sum1 + sum2 + sum3)

    def dDNL_dmu(self, z, k, mu, parameters={}):
        pre_fact = self.DNL(z, k, mu, parameters)
        res = self.DNL1(z, k, parameters) * self.dDNL2_dmu(k, mu, parameters)
        return pre_fact * res

    def dDNL1_dk(self, z, k, parameters={}):
        if "d1_q1" in parameters:
            d1_q1 = parameters["d1_q1"]
        else:
            d1_q1 = self.default_d1_q1
        if "d1_q2" in parameters:
            d1_q2 = parameters["d1_q2"]
        else:
            d1_q2 = self.default_d1_q2

        plin = self.linP_Mpc(z, k)
        dplin_dk = self.dplin_dk(z, k)

        sum1a = k**3 / 2 / np.pi**2 * dplin_dk
        sum1b = 3 * k**2 / 2 / np.pi**2 * plin

        sum2a = k**6 / 2 / np.pi**4 * plin * dplin_dk
        sum2b = 6 * k**5 / 4 / np.pi**4 * plin**2

        return d1_q1 * (sum1a + sum1b) + d1_q2 * (sum2a + sum2b)

    def dDNL2_dk(self, k, mu, parameters={}):
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

        return -d1_av * k ** (d1_av - 1) / d1_kvav * mu**d1_bv

    def dDNL2_dmu(self, k, mu, parameters={}):
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

        return -d1_bv * (k**d1_av / d1_kvav) * mu ** (d1_bv - 1)

    def dDNL3_dk(self, k, parameters={}):
        if "d1_kp" in parameters:
            d1_kp = parameters["d1_kp"]
        else:
            d1_kp = self.default_d1_kp

        return -2 * k / d1_kp**2

    def dplin_dk(self, z, k):
        # implement
        plin = self.linP_Mpc(z, k)
        return np.gradient(plin)
