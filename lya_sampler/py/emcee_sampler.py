import numpy as np
import sys
import os
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import cProfile
import emcee
import corner
# our own modules
import simplest_emulator
import linear_emulator
import gp_emulator
import data_PD2013
import mean_flux_model
import thermal_model
import lya_theory
import likelihood


class EmceeSampler(object):
    """Wrapper around an emcee sampler for Lyman alpha likelihood"""

    def __init__(self,like=None,emulator=None,free_parameters=None,
        									nwalkers=None,verbose=True):
        """Setup sampler from likelihood, or use default"""

        self.verbose=verbose

        if like:
            if self.verbose: print('use input likelihood')
            self.like=like
            if free_parameters:
                self.like.set_free_parameters(free_parameters)
        else:
            if self.verbose: print('use default likelihood')
            data=data_PD2013.P1D_PD2013(blind_data=True)
            zs=data.z
            theory=lya_theory.LyaTheory(zs,emulator=emulator)
            self.like=likelihood.Likelihood(data=data,theory=theory,
                            free_parameters=free_parameters,verbose=verbose)

        # number of free parameters to sample
        self.ndim=len(self.like.free_params)
        # number of walkers
        if nwalkers:
            self.nwalkers=nwalkers
        else:
            self.nwalkers=10*self.ndim

        # setup sampler
        self.sampler = emcee.EnsembleSampler(self.nwalkers,self.ndim,
                                                            self.log_prob)

        # setup walkers
        self.p0=self.get_initial_walkers()

        if self.verbose:
            print('done setting up sampler')
        

    def run_burn_in(self,nsteps):
        """Start sample from initial points, for nsteps"""

        if self.verbose: print('start burn-in, will do',nsteps,'steps')
        pos, prob, state = self.sampler.run_mcmc(self.p0,nsteps)
        if self.verbose: print('finished burn-in')

        self.burnin_nsteps=nsteps
        self.burnin_pos=pos
        self.burnin_prob=prob
    
        return


    def run_chains(self,nsteps,nprint=20):
        """Run actual chains, starting from end of burn-in"""

        # reset and run actual chains
        self.sampler.reset()

        pos=self.burnin_pos
        for i, result in enumerate(self.sampler.sample(pos,iterations=nsteps)):
            if i % nprint == 0:
                print(i,result[0][0])

        return


    def get_initial_walkers(self):
        """Setup initial states of walkers in sensible points"""

        ndim=self.ndim
        nwalkers=self.nwalkers

        if self.verbose: 
            print('set %d walkers with %d dimensions'%(nwalkers,ndim))

        p0=np.random.rand(ndim*nwalkers).reshape((nwalkers,ndim))

        # make sure that all walkers are within the convex hull
        for iw in range(nwalkers):
            walker=p0[iw]
            if self.verbose: print(iw,'walker',walker)
            test=self.log_prob(walker)
            while (test == -np.inf):
                if self.verbose: print(iw,'bad walker',walker)
                walker = np.random.rand(ndim)
                if self.verbose: print(iw,'try walker',walker)
                test=self.log_prob(walker)
            if self.verbose: print(iw,'good walker',walker,' log_prob=',test)
            p0[iw]=walker

        return p0


    def log_prob(self,values):
        """Function that will actually be called by emcee"""

        test_log_prob=self.like.log_prob(values=values)
        if np.isnan(test_log_prob):
            if self.verbose:
                print('parameter values outside hull',values)
                return -np.inf
        return test_log_prob


    def go_silent(self):
        self.verbose=False
        self.like.go_silent()


    def plot_histograms(self,cube=False):
        """Make histograms for all dimensions, using re-normalized values if
            cube=True"""

        for ip in range(self.ndim):
            param=self.like.free_params[ip]
            if cube:
                values=self.sampler.flatchain[:,ip]
                title=param.name+' in cube'
            else:
                cube_values=self.sampler.flatchain[:,ip]
                values=param.value_from_cube(cube_values)
                title=param.name

            plt.hist(values, 100, color="k", histtype="step")
            plt.title(title)
            plt.show()

        return


    def plot_corner(self,cube=False):
        """Make corner plot, using re-normalized values if cube=True"""

        labels=[]
        for p in self.like.free_params:
            if cube:
                labels.append(p.name+' in cube')
            else:
                labels.append(p.name)

        if cube:
            values=self.sampler.flatchain
        else:
            cube_values=self.sampler.flatchain
            list_values=[self.like.free_params[ip].value_from_cube(
                                cube_values[:,ip]) for ip in range(self.ndim)]
            values=np.array(list_values).transpose()

        corner.corner(values,labels=labels)
        plt.show()

        return