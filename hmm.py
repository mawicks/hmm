from tools import cumsum, find
from random import randrange, uniform

def draw(cdf):
    r = uniform(0, 1)
    x = find(cdf, r)

    if x != None:
        return x

    # Assuming the CDF is actually a CDF, it's still possible the last value isn't exactly 1 in which case x could be None.
    # In that case, return the index of the last value
    if len(cdf) != 0:
        return(len(cdf)-1)
    
    return None

def random_distribution(n):
    l = [uniform(0, 1) for _ in range(n)]
    s = sum(l)
    return tuple(x / s for x in l)

class HMM:
    def compute_cums(self):
        self._cum_transitions = tuple(tuple(cumsum(row)) for row in self._transitions)
        self._cum_emissions = tuple(tuple(cumsum(row)) for row in self._emissions)
        self._cum_initial = tuple(cumsum(self._initial))
        
    def random_parameters(self, n_states, emission_counts = None):
        '''Set a random set of initial parameters.  Counts of observations are used to estimate an initial set of emission probabilities.
State transitions are set randomly.'''
        m = len(emission_counts)
        sequence_length = sum(emission_counts)

        emissions = tuple( float(count+1) / float(sequence_length+m) for count in emission_counts )
        print(emissions)
        
        self._emissions = tuple(emissions for _ in range(n_states))
        self._transitions = tuple(random_distribution(n_states) for _ in range(n_states))
        self._initial = random_distribution(n_states)

        self.compute_cums()
    
    def set_parameters(self, transitions=None, emissions=None, initial=None):
        if len(transitions) == 0:
            raise Exception('Must have at least one state')
        
        row_lengths = [len(row) for row in transitions]
        mx,mn = (max(row_lengths), min(row_lengths))
        if mx != mn or mx != len(transitions):
            raise Exception('Transition matrix is not square')
        
        if len(transitions) != len(emissions):
            raise Exception('transition matrix and emissions length have different numbers of rows')
        
        if len(transitions) != len(initial):
            raise Exception('transition matrix and initial state probability distribution have different numbers of rows')
        
        self.compute_cums()
        
    def sim(self, n):
        '''emissions, states = sim(n):  Simulate n iterations and return the emission and state trajectories'''
        path = []
        state = draw(self._cum_initial)
        for _ in range(n):
            emission = draw(self._cum_emissions[state])
            path.append((emission, state))
            
            print('state: {0}, emission: {1}'.format(state, emission))

            # Update for next iteration
            state = draw(self._cum_transitions[state])
        return zip(*path)
