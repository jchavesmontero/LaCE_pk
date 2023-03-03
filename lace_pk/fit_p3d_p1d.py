import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class FitP3D_P1D(object):
    """Fit measured P3D and P1D with Arinyo model."""

    def __init__(
        self,
        data,
        model,
        fit3d_k_Mpc_max=10,
        fit1d_k_Mpc_max=10,
        noise_floor_3d=0.05,
        noise_floor_1d=0.05,
        min_counts=0,
    ):
        """Setup P3D flux power model and measurement.
        Inputs:
         - data: measured P3D and P1D
         - model: theoretical model for 3D flux power
         - fit_k_Mpc_max: fit only modes with k_Mpc < fit_k_Mpc_max
         - noise_floor: down-weight high-k modes in fit"""

        # store data and model
        self.data = data
        self.model = model
        self.fit3d_k_Mpc_max = fit3d_k_Mpc_max
        self.fit1d_k_Mpc_max = fit1d_k_Mpc_max

        # identify bins included in fit
        k3d_Mpc = self.data["k3d_Mpc"]
        k1d_Mpc = self.data["k1d_Mpc"]
        # first, consider only bins that have been measured (counts>0)
        counts = self.data["counts"]
        good3d = counts > min_counts
        self.fit3d_bins = np.array(good3d)
        # and bins that have k_Mpc < fit_k_Mpc_max
        self.fit3d_bins[good3d] = k3d_Mpc[good3d] < fit3d_k_Mpc_max
        good1d = k1d_Mpc < fit1d_k_Mpc_max
        self.fit1d_bins = np.array(good1d)

        # relative errors with noise floor (pretty sure we have a 2 here)
        self.rel_error3d = np.empty_like(counts)
        self.rel_error3d[good3d] = np.sqrt(2.0 / counts[good3d]) + noise_floor_3d
        self.rel_error3d[~good3d] = np.inf
        self.rel_error1d = np.empty_like(k1d_Mpc) + noise_floor_1d
        self.rel_error1d[good1d] = noise_floor_1d
        self.rel_error1d[~good1d] = np.inf

    def get_model3d(self, parameters={}):
        """Model for the 3D flux power spectrum"""

        # identify (k,mu) for bins included in fit
        k = self.data["k3d_Mpc"][self.fit3d_bins]
        mu = self.data["mu"][self.fit3d_bins]
        z = self.data["z"]

        return self.model.P3D_Mpc(z, k, mu, parameters)

    def get_model1d(
        self,
        parameters={},
        k_perp_min=0.001,
        k_perp_max=60,
        n_k_perp=200,
    ):
        """Model for the 1D flux power spectrum"""

        # identify (k) for bins included in fit
        k = self.data["k1d_Mpc"][self.fit1d_bins]
        z = self.data["z"]
        res = self.model.P1D_Mpc(
            z,
            k,
            parameters=parameters,
            k_perp_min=k_perp_min,
            k_perp_max=k_perp_max,
            n_k_perp=n_k_perp,
        )
        return res

    def get_chi2(self, parameters={}, return_npoints=False):
        """Compute chi squared for a particular P3D model.
        - parameters: dictionary with parameters to use
        - return_points: return number of data points used"""

        # get P3D measurement for bins included in fit
        data3d = self.data["p3d_Mpc"][self.fit3d_bins]
        data1d = self.data["p1d_Mpc"][self.fit1d_bins]

        # compute model for these wavenumbers
        theory3d = self.get_model3d(parameters)
        theory1d = self.get_model1d(parameters)

        # compute absolute error
        error3d = data3d * self.rel_error3d[self.fit3d_bins]
        error1d = data1d * self.rel_error1d[self.fit1d_bins]

        # compute chi2
        chi2_3d = np.sum(((data3d - theory3d) / error3d) ** 2) / np.sum(self.fit3d_bins)
        chi2_1d = np.sum(((data1d - theory1d) / error1d) ** 2) / np.sum(self.fit1d_bins)
        # print(chi2_3d, chi2_1d)
        chi2 = chi2_3d + chi2_1d

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
        parameters = {}
        for i in range(Np):
            name = parameter_names[i]
            value = values[i]
            parameters[name] = value

        return self.get_log_like(parameters)

    def maximize_likelihood(self, parameters):
        """Run minimizer and return best-fit values"""

        ndim = len(parameters)
        names = list(parameters.keys())
        values = np.array(list(parameters.values()))

        # generate random initial value
        ini_values = values * (1 + 0.1 * np.random.randn(ndim))

        # lambda function to minimize
        minus_log_like = lambda *args: -self._log_like(*args)

        # get max likelihood values
        results = minimize(
            minus_log_like,
            ini_values,
            args=(names),
            method="Nelder-Mead",
            options={"maxiter": 10000},
        )

        # update parameters dictionary
        best_fit_parameters = {}
        for ip in range(ndim):
            best_fit_parameters[names[ip]] = results.x[ip]

        return results, best_fit_parameters

    def plot_p3d_ratio(
        self,
        parameters={},
        downsample_mu=1,
        divide_by="linP",
        extra_parameters=None,
        label="base",
        extra_label="extra",
    ):
        """Plot measured P3D, with different options:
        - parameters: dictionary specifying parameters values for model
        - downsample_mu: plot only several mu bins (less cluttered)
        - divide_by: one can use different denominators
            - linP: linear power at (z,k)
            - lowk: low-k limit of model at (z,mu) using input parameters
            - model: full p3d model at (z,k,mu) using input parameters
        - extra_parameters: parameter values for extra_model
        - label: label to use for main model
        - extra_label: label to use for extra model"""

        # store data and redshift
        data = self.data
        z = data["z"]

        # get colormap for pretty plots
        cm = plt.get_cmap("hsv")

        # figure out number of (k,mu) bins
        n_k, n_mu = data["counts"].shape
        mu_bin_edges = np.linspace(0.0, 1.0, n_mu + 1)

        # make a plot for several mu bins
        for i in range(0, n_mu, downsample_mu):
            col = cm(1.0 * i / n_mu)
            # mask bins not measured
            mask = data["counts"][:, i] == 0
            keep = ~mask
            iP = data["p3d_Mpc"][:, i][keep]
            ik = data["k_Mpc"][:, i][keep]
            imu = data["mu"][:, i][keep]
            # compute errorbar (including noise floor)
            irel_error = self.rel_error[:, i][keep]
            if divide_by == "linP":
                # interpolate linear power to the precise value of k here
                denom = self.model.linP_Mpc(z=z, k_Mpc=ik)
                ylabel = r"$P_F(k,\mu) / P_L(k)$"
            elif divide_by == "lowk":
                # evaluate low-k limit of P3D / linP, i.e., (Kaiser term)^2
                lowk_ratio = self.model.lowk_P3D_over_linP(
                    z=z, mu=imu, lowk=1e-3, parameters=parameters
                )
                linP = self.model.linP_Mpc(z=z, k_Mpc=ik)
                denom = lowk_ratio * linP
                ylabel = r"$D_{NL}(k,\mu)$"
            elif divide_by == "model":
                # evaluate full p3d model
                denom = self.model.P3D_Mpc(z=z, k=ik, mu=imu, parameters=parameters)
                ylabel = r"residual $P_F(k,\mu)$"
            else:
                raise ValueError("wrong value of divide_by " + divide_by)
            # compute ratio over linear power and error
            power_ratio = iP / denom
            ratio_error = irel_error * power_ratio
            # plot ratio
            plt.errorbar(
                ik,
                power_ratio,
                yerr=ratio_error,
                capsize=3,
                ecolor=col,
                color=col,
                marker="x",
                label=r"%.2f $\leq \mu \leq$ %.2f"
                % (mu_bin_edges[i], mu_bin_edges[i + 1]),
            )

            # model power ratio
            imodel = self.model.P3D_Mpc(z=z, k=ik, mu=imu, parameters=parameters)
            model_ratio = imodel / denom
            if extra_parameters is not None:
                iextra = self.model.P3D_Mpc(
                    z=z, k=ik, mu=imu, parameters=extra_parameters
                )
                extra_ratio = iextra / denom
                if i == 0:
                    plt.plot(ik, model_ratio, ls="-", color="gray", label=label)
                    plt.plot(ik, extra_ratio, ls=":", color="gray", label=extra_label)
                plt.plot(ik, extra_ratio, ls=":", color=col)
            plt.plot(ik, model_ratio, ls="-", color=col)

        plt.title(r"Flux biasing z=%.2f" % (z))
        plt.axvline(x=self.fit_k_Mpc_max, ls="--", color="gray")
        plt.xlabel("k (1/Mpc)")
        plt.ylabel(ylabel)
        plt.legend(loc="best", numpoints=1, fancybox=True, fontsize="small")
        plt.xscale("log")
        plt.yscale("log")
