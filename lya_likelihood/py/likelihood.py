import numpy as np
import matplotlib.pyplot as plt
import data_PD2013
import lya_theory
import likelihood_parameter


class Likelihood(object):
    """Likelihood class, holds data, theory, and knows about parameters"""

    def __init__(self,data=None,theory=None,emulator=None,
                    free_parameters=None,verbose=False,
                    prior_Gauss_rms=0.2,
                    min_kp_kms=None,ignore_emu_cov=False):
        """Setup likelihood from theory and data. Options:
            - if prior_Gauss_rms is None it will use uniform priors
            - ignore k-bins with k > min_kp_kms
            - ignore_emu_cov will ignore emulator covariance in likelihood."""

        self.verbose=verbose
        self.prior_Gauss_rms=prior_Gauss_rms
        self.ignore_emu_cov=ignore_emu_cov

        if data:
            self.data=data
        else:
            if self.verbose: print('use default data')
            self.data=data_PD2013.P1D_PD2013(blind_data=True)

        # (optionally) get rid of low-k data points
        self.data._cull_data(min_kp_kms)

        if theory:
            self.theory=theory
        else:
            zs=self.data.z
            if self.verbose: print('use default theory')
            self.theory=lya_theory.LyaTheory(zs,emulator=emulator)

        # setup parameters
        if not free_parameters:
            free_parameters=['ln_tau_0']
        self.set_free_parameters(free_parameters)

        if self.verbose: print(len(self.free_params),'free parameters')

        return


    def set_free_parameters(self,free_parameter_names):
        """Setup likelihood parameters that we want to vary"""

        # setup list of likelihood free parameters
        self.free_params=[]

        # get all parameters in theory, free or not
        params = self.theory.get_parameters()

        # select parameters using input list of names
        for par in params:
            if par.name in free_parameter_names:
                self.free_params.append(par)

        Nfree=len(self.free_params)
        Nin=len(free_parameter_names)

        assert (Nfree==Nin), 'could not setup free paremeters'

        if self.verbose:
            print('likelihood setup with {} free parameters'.format(Nfree))

        return


    def parameters_from_sampling_point(self,values):
        """Translate input array of values (in cube) to likelihood parameters"""

        assert len(values)==len(self.free_params),'size mismatch'
        Npar=len(values)
        like_params=[]
        for ip in range(Npar):
            par = self.free_params[ip].get_new_parameter(values[ip])
            like_params.append(par)

        return like_params


    def get_p1d_kms(self,k_kms=None,values=None,return_covar=False):
        """Compute theoretical prediction for 1D P(k)"""

        if k_kms is None:
            k_kms=self.data.k

        # translate sampling point (in unit cube) to parameter values
        if values is not None:
            like_params= self.parameters_from_sampling_point(values)
        else:
            like_params=[]

        return self.theory.get_p1d_kms(k_kms,like_params=like_params,
                                            return_covar=return_covar)


    def get_chi2(self,values=None):
        """Compute chi2 using data and theory, without adding
            emulator covariance"""

        log_like=self.get_log_like(values,ignore_log_det_cov=True,
                                    add_emu_cov=False)
        if log_like is None:
            return None
        else:
            return -2.0*log_like


    def get_log_like(self,values=None,ignore_log_det_cov=True,
                        add_emu_cov=False):
        """Compute log(likelihood), including determinant of covariance
            unless you are setting ignore_log_det_cov=True.
            If add_emu_cov, include emulator uncertainty to the covariance."""

        # get measured bins from data
        k_kms=self.data.k
        zs=self.data.z
        Nz=len(zs)

        # ask emulator prediction for P1D in each bin
        emu_p1d,emu_covar = self.get_p1d_kms(k_kms,values,return_covar=True)
        if self.verbose: print('got P1D from emulator')

        # compute log like contribution from each redshift bin
        log_like=0

        for iz in range(Nz):
            # acess data for this redshift
            z=zs[iz]
            # make sure that theory is valid
            if emu_p1d[iz] is None:
                if self.verbose: print(z,'theory did not emulate p1d')
                return None
            if self.verbose: print('compute chi2 for z={}'.format(z))
            # get data
            p1d=self.data.get_Pk_iz(iz)
            data_cov=self.data.get_cov_iz(iz)
            # add covariance from emulator
            if add_emu_cov:
                cov = data_cov + emu_covar[iz]
            else:
                cov = data_cov

            # compute chi2 for this redshift bin
            icov = np.linalg.inv(cov)
            diff = (p1d-emu_p1d[iz])
            chi2_z = np.dot(np.dot(icov,diff),diff)
            # check whether to add determinant of covariance as well
            if ignore_log_det_cov:
                log_like_z = -0.5*chi2_z
            else:
                (_, log_det_cov) = np.linalg.slogdet(cov)
                log_like_z = -0.5*(chi2_z + log_det_cov)
            log_like += log_like_z
            if self.verbose: print('added {} to log_like'.format(log_like_z))

        return log_like


    def log_prob(self,values):
        """Return log likelihood plus log priors"""

        # compute log_prior
        log_prior=self.get_log_prior(values)

        # compute log_like (option to ignore emulator covariance)
        log_like=self.get_log_like(values,ignore_log_det_cov=False,
                                    add_emu_cov=not self.ignore_emu_cov)

        if log_like is None:
            if self.verbose: print('was not able to emulate at least on P1D')
            return -np.inf

        return log_like + log_prior


    def get_log_prior(self,values):
        """Compute logarithm of prior"""

        assert len(values)==len(self.free_params),'size mismatch'

        # Always force parameter to be within range (for now)
        if max(values) > 1.0:
            return -np.inf
        if min(values) < 0.0:
            return -np.inf

        # Add Gaussian prior around fiducial parameter values
        if self.prior_Gauss_rms is None:
            return 0
        else:
            rms=self.prior_Gauss_rms
            fid_values=[p.value_in_cube() for p in self.free_params]
            log_prior=-np.sum((np.array(fid_values)-values)**2/(2*rms**2))
            return log_prior


    def go_silent(self):
        """Turn off verbosity on all objects in likelihood object"""

        self.verbose=False
        self.theory.verbose=False
        self.theory.cosmo.verbose=False
        self.theory.emulator.verbose=False
        self.theory.emulator.arxiv.verbose=False


    def go_loud(self):
        """Turn on verbosity on all objects in likelihood object"""

        self.verbose=True
        self.theory.verbose=True
        self.theory.cosmo.verbose=True
        self.theory.emulator.verbose=True
        self.theory.emulator.arxiv.verbose=True


    def plot_p1d(self,values=None,values2=None,plot_every_iz=1):
        """Plot P1D in theory vs data. If plot_every_iz >1,
            plot only few redshift bins"""

        # get measured bins from data
        k_kms=self.data.k
        zs=self.data.z
        Nz=len(zs)

        # ask emulator prediction for P1D in each bin
        emu_p1d, emu_cov = self.get_p1d_kms(k_kms,values,return_covar=True)
        if values2 is not None:
            emu_p1d_2, emu_cov_2 = self.get_p1d_kms(k_kms,values2,return_covar=True)

        if self.verbose: print('got P1D from emulator')

        # plot only few redshifts for clarity
        for iz in range(0,Nz,plot_every_iz):
            # acess data for this redshift
            z=zs[iz]
            p1d_data=self.data.get_Pk_iz(iz)
            p1d_cov=self.data.get_cov_iz(iz)
            p1d_theory=emu_p1d[iz]
            cov_theory=emu_cov[iz]
            
            if p1d_theory is None:
                if self.verbose: print(z,'emulator did not provide P1D')
                continue
            # plot everything
            col = plt.cm.jet(iz/(Nz-1))
            plt.errorbar(k_kms,p1d_data*k_kms/np.pi,color=col,
                    yerr=np.sqrt(np.diag(p1d_cov))*k_kms/np.pi,label='z=%.1f'%z)
            #plt.plot(k_kms,p1d_theory*k_kms/np.pi,color=col,
            #        linestyle="--")
            plt.errorbar(k_kms,p1d_theory*k_kms/np.pi,color=col,
                    yerr=np.sqrt(np.diag(cov_theory))*k_kms/np.pi,linestyle="dashed")
            if values2 is not None:
                p1d_theory_mcmc=emu_p1d_2[iz]
                cov_theory_mcmc=emu_cov_2[iz]
                plt.errorbar(k_kms,p1d_theory_mcmc*k_kms/np.pi,
                    yerr=np.sqrt(np.diag(cov_theory_mcmc))*k_kms/np.pi,color=col,
                    linestyle=":")
                
        if values2 is not None:
            plt.plot(-10,-10,linestyle="-",label="Data",color="k")
            plt.plot(-10,-10,linestyle="--",label="Likelihood fit",color="k")
            plt.plot(-10,-10,linestyle=":",label="MCMC best fit",color="k")
        plt.yscale('log')
        plt.legend()
        plt.xlabel('k [s/km]')
        plt.ylabel(r'$k_\parallel \, P_{\rm 1D}(z,k_\parallel) / \pi$')
        plt.ylim(0.005,0.6)
        plt.xlim(min(k_kms)-0.001,max(k_kms)+0.001)
        plt.tight_layout()
        plt.show()

        return


    def overplot_emulator_calls(self,param_1,param_2,values=None,
                                tau_scalings=True,temp_scalings=True):
        """For parameter pair (param1,param2), overplot emulator calls
            with values stored in arxiv, color coded by redshift"""

        # mask post-process scalings (optional)
        emu_data=self.theory.emulator.arxiv.data
        Nemu=len(emu_data)
        if not tau_scalings:
            mask_tau=[x['scale_tau']==1.0 for x in emu_data]
        else:
            mask_tau=[True]*Nemu
        if not temp_scalings:
            mask_temp=[(x['scale_T0']==1.0) 
                        & (x['scale_gamma']==1.0) for x in emu_data]
        else:
            mask_temp=[True]*Nemu

        # figure out values of param_1,param_2 in arxiv
        emu_1=np.array([emu_data[i][param_1] for i in range(Nemu) if (
                                                  mask_tau[i] & mask_temp[i])])
        emu_2=np.array([emu_data[i][param_2] for i in range(Nemu) if (
                                                  mask_tau[i] & mask_temp[i])])

        # translate sampling point (in unit cube) to parameter values
        if values is not None:
            like_params= self.parameters_from_sampling_point(values)
        else:
            like_params=[]
        emu_calls=self.theory.get_emulator_calls(like_params=like_params)
        # figure out values of param_1,param_2 called
        call_1=[emu_call[param_1] for emu_call in emu_calls]
        call_2=[emu_call[param_2] for emu_call in emu_calls]

        # overplot
        zs=self.data.z
        emu_z=np.array([emu_data[i]['z'] for i in range(Nemu) if (
                                                  mask_tau[i] & mask_temp[i])])
        zmin=min(min(emu_z),min(zs))
        zmax=max(max(emu_z),max(zs))
        plt.scatter(emu_1,emu_2,c=emu_z,s=1,vmin=zmin, vmax=zmax)
        plt.scatter(call_1,call_2,c=zs,s=50,vmin=zmin, vmax=zmax)
        cbar=plt.colorbar()
        cbar.set_label("Redshift", labelpad=+1)
        plt.xlabel(param_1)
        plt.ylabel(param_2)
        plt.show()

        return
