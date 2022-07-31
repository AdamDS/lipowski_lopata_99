import sys
import copy
import numpy as np


class ll_99(object):
    def __init__(self, n_trials, T, L, d, r_s, rng_seed=None):
        '''
        INPUTS
        ------
        n_trials (int) : number of monte carlo runs for each setup
        T (int) : max epochs to allow simulation
        L (int) : size of one dimension of the lattice
        d (int) : dimension of lattice
        r_s (list of float) : 
        rng_seed (float) : OPTIONAL seed for numpy.random
        
        ATTRIBUTES
        ----------
        n_trials : n_trials of INPUTS
        max_epochs : T of INPUTS
        length : L of INPUTS
        dimension : d of INPUTS
        shape (tuple) : lattice shape given by length and dimension
        rng (np.random) : random number generator
        interactions is an np.ndarray of frustration contributions (w_i,j)
        r_s (list of float) : control parameters, thresholds dictating extinction
        r (float) : current control parameter
        '''
        self.n_trials = n_trials
        self.max_epochs = T
        self.length = L
        self.dimension = d
        self.shape = tuple([self.length]*(self.dimension))
        self.rng = np.random.default_rng(rng_seed)  # arg sets seed of RNG
        
        self.interactions = np.zeros(self.shape)
        self.randomize()
        
        self.r_s = r_s
        self.r = self.r_s[0]
        
        self.extinct_species = None

    # random w_i,j
    def randomize(self):
        _ = np.ravel(self.interactions)
        for i in np.arange(0, len(_)):
            _[i] = self.rng.random()
        self.interactions = _.reshape(self.shape)


    def random_species(self):
        site = [int(self.length*self.rng.random()) \
            for _ in np.arange(0, self.dimension)]
        return site


    def neighbor_interaction(self, site, i, left_right):
        # frustration from "left_right" neighbors
        _neighbor = copy.deepcopy(site)
        _neighbor[i] = site[i] + left_right
        if _neighbor[i] < 0:  # periodic boundaries
            _neighbor[i] = self.length - 1
        elif _neighbor[i] >= self.length:
            _neighbor[i] = 0

        interaction = self.interactions[tuple(_neighbor)]

        return interaction


    # calculate species frustration omega_j
    def species_interact(self, site):
        '''
        site is a tuple indicating site of interest (i,j of w_i,j)
        frustration is the sum of neighbor interactions (omega_j)
        '''
        frustration = 0
        for i in np.arange(0, self.dimension):
            # frustration from "left" neighbors
            frustration += self.neighbor_interaction(site, i, -1)
            # frustration from "right" neighbors
            frustration += self.neighbor_interaction(site, i, +1)

        return frustration


    def replace_extinct_species(self, random_species):
        if len(random_species):
            self.interactions[random_species] = self.rng.random()


    def random_species_extinction(self, random_site, frustration):
        if frustration < self.r:
            self.replace_extinct_species(random_site)
            return True
        return False


    def epoch(self):
        # "Choose a site i at random"
        a_species = self.random_species()
        # "Calculate sum_j(w_i,j), where summation is over all 
        #    nearest neighbors j"
        frustration = self.species_interact(a_species)
        # "If omega < r, then the chosen species, due to too 
        #    much frustration, becomes extinct and the site 
        #    becomes occupied by a new species with the 
        #    interactions w_i,j chosen anew. If omega > r, 
        #    the species at the site i survives"
        went_extinct = self.random_species_extinction(a_species, frustration)
        
        if went_extinct:
            return a_species
        return None


    def steady_state_simulation(self):
        self.randomize()  # initialize lattice
        extinct_species = np.zeros((self.max_epochs,), dtype=list)
        for t in np.arange(0, self.max_epochs):
            extinct_species[t] = self.epoch()
        return extinct_species


    def steady_state(self):
        '''
        "we studied the density p of active sites ~i.e., those with 
        omega < r in the steady state for the system size L = 10^4 
        and L = 10^5 and with initial interactions chosen randomly."
        '''
        self.results = np.zeros((self.n_trials, self.max_epochs), dtype=list)
        for trial in np.arange(0, self.n_trials):
            extinct_species = self.steady_state_simulation()
            self.results[trial] = extinct_species.T
        
        
    def one_random_species(self):
        _ = np.ravel(self.interactions)
        _[0] = self.rng.random()
        self.interactions = _.reshape(self.shape)
        
        
    def single_seed_simulation(self):
        self.one_random_species()  # initialize lattice
        extinct_species = np.zeros((self.max_epochs,), dtype=list)
        for t in np.arange(0, self.max_epochs):
            extinct_species[t] = self.epoch()
        return extinct_species


    def single_seed(self):
        '''
        "we studied the density p of active sites ~i.e., those with 
        omega < r in the steady state for the system size L = 10^4 
        and L = 10^5 and with initial interactions chosen randomly."
        '''
        self.results = np.zeros((self.n_trials, self.max_epochs), dtype=list)
        for trial in np.arange(0, self.n_trials):
            extinct_species = self.steady_state_simulation()
            self.results[trial] = extinct_species.T
        
        
    def study(self):
        self.probability_survival = np.sum(np.array([[0 if x is None else 1 \
            for x in exp.results[i]] for i in range(0, 10)]), axis=0)

        
#def main():
#    # "for the system size L = 10^4 and L = 10^5"
#    length = 10000

#    # "To examine the properties of the one-dimensional version 
#    #    of this model, we used Monte Carlo simulations"
#    dimension = 1  

#    # "Our statistics is based on runs up to t = 2x10^4"
#    T = 20000
#    # "we usually made 2x10^4 independent runs"
#    n_trials = 20000

#    # clearly indicate the existence of the phase transition 
#    #    in the present model around r~0.44, which separates 
#    #    the active (p > 0) and the absorbing (p = 0) phases
#    r_s = [0.44, 0.4405, 0.441, 0.4415]  

#    experiment = ll_99(n_trials, T, L, d, r_s, rng_seed=None)

#    experiment.steady_state()

# sys.exit(main())