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
        
        self.r_s = r_s
        self.r = self.r_s[0]
        
        self.active_species = None
        self.init_active_species()

        
    def init_active_species(self):
        self.active_species = np.zeros((self.max_epochs,), dtype=int)
        
        
    # random w_i,j
    def init_interactions(self):
        '''
        The authors indicate that each species experiences frustration 
        from neighboring species. And it's the interactions that are 
        reset when replacing a species. There's no indication that when 
        a species is extinct and replaced that it affects the interactions 
        expected from the neighboring species perspective. w_i,j is the 
        interaction of j with i; does w_i,j == w_j,i? Must w_i,j == w_i+1,j?
        I think the data structure should be different. The interactions 
        should be a dict of species with interactions for each dimension. 
        The interactions of each dimension should define the 
        'left' and 'right' interactions.
        '''
        # self.interactions = np.zeros(self.shape)
        # _ = np.ravel(self.interactions)
        # for i in np.arange(0, len(_)):
        #     _[i] = self.rng.random()
        # self.interactions = _.reshape(self.shape)
        self.init_active_species()
        self.interactions = np.ndarray(self.shape, dtype=dict)
        _ = np.ravel(self.interactions)
        for i in np.arange(0, np.prod(self.shape)):
            _[i] = dict()
            site = _[i]
            self.randomize_species_ravel(site)
            starts_active = self.is_active(site)
            if starts_active:
                self.active_species[0] += 1
        self.interactions = _.reshape(self.shape)


    def randomize_species_ravel(self, site):
        site['w_ij'] = np.zeros((self.dimension, 2), dtype=float)
        for d in np.arange(0, self.dimension):
            site['w_ij'][d] = self.random_dim_interactions()
        site['omega'] = np.sum(site['w_ij'])
        
        
    def random_dim_interactions(self):
        return [self.rng.random(), self.rng.random()]
        
        
    def is_active(self, site):
        # "active sites (i.e., those with omega > r)"
        if site['omega'] > self.r:
            return True
        return False

        
    def randomize_species(self, lattice_site):
        species = self.interactions[lattice_site]
        for d in np.arange(0, self.dimension):
            species['w_ij'][d] = self.random_dim_interactions()
        species['omega'] = np.sum(species['w_ij'])


    def periodic_boundaries(self, site):
        if site < 0:  # off left
            site = self.length - 1
        elif site >= self.length:  # off right
            site = 0
        # implicit else, not at boundary
        
        return site
        
            
    def update_neighbor(self, neighbor_site, d, left_right):
        neighbor_species = self.interactions[neighbor_site]
        neighbor_species['w_ij'][d][left_right] = self.rng.random()
        
        
    def update_neighbor_interactions(self, lattice_site, d, left_right):
        _neighbor = list(copy.deepcopy(lattice_site))
        _neighbor[d] = self.periodic_boundaries(_neighbor[d] + left_right)
        self.update_neighbor(tuple(_neighbor), d, -1*left_right)

        
    def replace_extinct_species(self, lattice_site):
        # refresh extinct species
        self.randomize_species(lattice_site)
        
        # reset neighbor interactions
        for d in np.arange(0, self.dimension):
            # left neighbor
            self.update_neighbor_interactions(lattice_site, d, -1)
            # right neighbor
            self.update_neighbor_interactions(lattice_site, d, 1)


    def get_a_random_species(self):
        site = [int(self.length*self.rng.random()) \
            for _ in np.arange(0, self.dimension)]
        
        return tuple(site)


    # calculate species frustration omega_j
    def species_interact(self, lattice_site):
        '''
        site is a tuple indicating site of interest (i,j of w_i,j)
        frustration is the sum of neighbor interactions (omega_j)
        '''
        frustration = self.interactions[lattice_site]['omega']
        
        return frustration
    

    def random_species_extinction(self, lattice_site, frustration):
        if frustration > self.r:
            self.replace_extinct_species(lattice_site)
    

    def epoch(self):
        # "Choose a site i at random"
        lattice_site = self.get_a_random_species()
        # "Calculate sum_j(w_i,j), where summation is over all 
        #    nearest neighbors j"
        frustration = self.species_interact(lattice_site)
        # "If omega < r, then the chosen species, due to too 
        #    much frustration, becomes extinct and the site 
        #    becomes occupied by a new species with the 
        #    interactions w_i,j chosen anew. If omega > r, 
        #    the species at the site i survives"
        self.random_species_extinction(lattice_site, frustration)
        
        species_activity = self.is_active(self.interactions[lattice_site])
        
        return species_activity


    # ============
    # Steady State 
    # ============
    def steady_state_simulation(self):
        self.init_interactions()  # initialize lattice
        total_active = self.active_species[0]
        for t in np.arange(0, self.max_epochs):
            species_activity = self.epoch()
            if species_activity:
                total_active += 1
            self.active_species[t] = total_active


    def steady_state(self):
        '''
        "we studied the density p of active sites ~i.e., those with 
        omega < r in the steady state for the system size L = 10^4 
        and L = 10^5 and with initial interactions chosen randomly."
        '''
        self.results = np.zeros((self.n_trials, self.max_epochs), dtype=list)
        for trial in np.arange(0, self.n_trials):
            self.steady_state_simulation()
            self.results[trial] = self.active_species.T
        
    
    # ===========
    # Single Seed
    # ===========
    def certain_active_species(self, species):
        species['w_ij'] = np.zeros((self.dimension, 2), dtype=float)
        species['w_ij'] += 2*self.dimension
        species['omega'] = np.sum(species['w_ij'])
        
        
    def init_single_seed(self):
        # "w_i,i+1 = r0 for i = 3,4,...,L and 2r0 < r_c"
        certainly_absorbed = 2*self.dimension
        self.init_interactions()  # initiales every site
        #self.interactions = np.ndarray(self.shape, dtype=dict)
        _ = np.ravel(self.interactions)
        for site in _:
            site['w_ij'] = np.zeros((self.dimension, 2), dtype=float)
            site['omega'] = np.sum(site['w_ij'])
            
        self.certain_active_species(_[1])
        
        # "assign w_1,2 = w_2,3 = 0.23"
        for d in np.arange(0, self.dimension):
            # left neighbor
            _[0]['w_ij'] = np.zeros((self.dimension, 2), dtype=float)
            
            # right neighbor
            _[2]['w_ij'] = np.zeros((self.dimension, 2), dtype=float)
        _[0]['omega'] = np.sum(_[0]['w_ij'])
        _[2]['omega'] = np.sum(_[2]['w_ij'])
        
        self.interactions = _.reshape(self.shape)
        the_seed = tuple([1 for _ in np.arange(0, self.dimension)])
        return the_seed        
        
        
    def single_seed_simulation(self):
        the_species = self.init_single_seed()  # initialize lattice
        time_to_absorb = self.max_epochs
        for t in np.arange(0, self.max_epochs):
            # measure epochs for single active site to absorb
            extinct_species = self.epoch()
            if extinct_species == the_species:
                if self.interactions[the_species]['omega'] > self.r:
                    time_to_absorb = t
                    break
                    
        return time_to_absorb


    def single_seed(self):
        '''
        "a single active site and L-1 inactive sites. Thus we 
        assign w_1,2 = w_2,3 = 0.23 and w_i,i+1 = r_0 for i = 3,4,...,L 
        and 2r_0 < r_c. With such an assignment and for r close 
        to r_c only the site with i = 2 is active. Since the 
        periodic boundary conditions are imposed in our simulations, 
        the system is translationally invariant and the initial 
        location of the active site is obviously irrelevant.
        
        From the description, I assume they continue testing the 
        1-D problem. w_1,2 is the site left of 
        '''
        
        self.results = np.zeros((self.n_trials, self.max_epochs), dtype=list)
        for trial in np.arange(0, self.n_trials):
            times_to_absorb = self.single_seed_simulation()
            self.results[trial] = times_to_absorb
        
    
    # ========
    # Analysis
    # ========
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