from tools import cumsum, find
from random import randrange, uniform
import numpy
import operator

def draw(cdf):
    '''Draw a random sample from the distribution defined by cdf.  The values in a cdf must be non-descreasing, but need not be normalized.
They can represent cumulative counts, for example, rather than cumulative probabilities.'''
    if len(cdf) > 0:
        return find(cdf, uniform(0, cdf[-1]))
    else:
        return None

def random_distribution(n):
    l = [uniform(0, 1) for _ in range(n)]
    s = sum(l)
    return tuple(x / s for x in l)

def row_sum(a, b):
    return (a_i + b_i for a_i, b_i in zip(a, b))

def sum_rows_of_log_likelihoods(a):
    col_max = numpy.max(a, axis=0)

    result = numpy.log(numpy.sum(numpy.exp(numpy.array([c - m for c, m in zip(a.T, col_max)])),
                       axis=1)) + col_max

    return result
    
class HMM:
    def set_parameters(self, transitions=None, emissions=None, initial=None):
        if len(transitions) == 0:
            raise Exception('Must have at least one state')
        
        row_lengths = [len(row) for row in transitions]
        mx,mn = (max(row_lengths), min(row_lengths))
        if mx != mn or mx != len(transitions):
            raise Exception('Transition matrix is not square')
        
        if len(transitions) != len(emissions):
            raise Exception('Transition matrix and emissions length have different numbers of rows')
        
        if len(transitions) != len(initial):
            raise Exception('Transition matrix and initial state probability distribution have different numbers of rows')

        self._n_states = len(transitions)
        self._n_symbols = len(emissions[0])
        
        self._transitions = numpy.array(transitions)
        self._log_transitions = numpy.log(self._transitions)
        
        self._emissions = numpy.array(emissions)
        self._log_emissions = numpy.log(self._emissions)
        
        self._initial = numpy.array(initial)
        self._log_initial = numpy.log(self._initial)
        
    def random_parameters(self, n_states, emission_counts):
        '''Set a random set of initial parameters.  Counts of observations are used to estimate an initial set of emission probabilities.
State transitions are set randomly.'''
        m = len(emission_counts)
        sequence_length = sum(emission_counts)

        emissions = tuple( float(count+1) / float(sequence_length+m) for count in emission_counts )
        
        self.set_parameters(tuple(random_distribution(n_states) for _ in range(n_states)),
                            tuple(emissions for _ in range(n_states)),
                            random_distribution(n_states))
        
    def sim(self, n):
        '''emissions, states = sim(n):  Simulate n iterations and return the emission and state trajectories'''
        cum_transitions = tuple(tuple(cumsum(row)) for row in self._transitions)
        cum_emissions = tuple(tuple(cumsum(row)) for row in self._emissions)
        cum_initial = tuple(cumsum(self._initial))
        
        path = []
        state = draw(cum_initial)
        for _ in range(n):
            emission = draw(cum_emissions[state])
            path.append((emission, state))
            
            # Update for next iteration
            state = draw(cum_transitions[state])
        return zip(*path)

    def forward(self, observations):
        log_alpha = self._log_initial + self._log_emissions[:,observations[0]]
        log_alpha_series = [log_alpha]

        for y in observations[1:]:
            log_alpha = (self._log_emissions[:,y] +
                         sum_rows_of_log_likelihoods(numpy.array([log_alpha_i + log_trans_i
                                                                  for log_alpha_i, log_trans_i
                                                                  in zip(log_alpha, self._log_transitions)])))
            log_alpha_series.append(log_alpha)

        return numpy.array(log_alpha_series)
    
    def backward(self, observations):
        log_beta = numpy.array([ 0.0 ] * self._n_states)
        log_beta_series = [log_beta]
        
        for y in reversed(observations[1:]):
            log_beta = sum_rows_of_log_likelihoods(numpy.array([log_beta_j + log_em_j[y] + log_trans_j
                                                                for log_beta_j, log_trans_j, log_em_j
                                                                in zip(log_beta, self._log_transitions.T, self._log_emissions)]))
            log_beta_series.append(log_beta)
            
        return numpy.array(list(reversed(log_beta_series)))
        
    def update(self, observations):
        log_alpha = self.forward(observations)
        log_beta = self.backward(observations)

        gamma_num = log_alpha + log_beta
        gamma_den = sum_rows_of_log_likelihoods(gamma_num.T)
        
        log_gamma = numpy.array([n-d for n,d in zip(gamma_num, gamma_den)])
        gamma = numpy.exp(log_gamma)
        
        print('\nalpha:\n{0}'.format(numpy.exp(log_alpha)))
        print('\nbeta:\n{0}'.format(numpy.exp(log_beta)))
        print('\ngamma:\n{0}'.format(gamma))
        
        new_transition_num = numpy.zeros((self._n_states, self._n_states))
        new_transition_den = numpy.zeros(self._n_states)

        for gamma_k, log_alpha_k, log_beta_k, y in zip(gamma[:-1], log_alpha[:-1], log_beta[1:], observations[1:]):
            log_xi_num = numpy.empty((self._n_states, self._n_states))
            for i in range(self._n_states):
                for j in range(self._n_states):
                    log_xi_num[i,j] = log_alpha_k[i] + log_beta_k[j] + self._log_transitions[i][j] + self._log_emissions[j][y]
            
            # This clumsy because numpy turns single rows/colums into vectors.
            # sum_rows_of_log_likelihoods requires a matrix argument, so we force it to be a matrix.
            log_xi_den = sum_rows_of_log_likelihoods(numpy.array([[r] for r in sum_rows_of_log_likelihoods(log_xi_num)]))
            
            xi = numpy.exp(log_xi_num - log_xi_den)
            
            new_transition_num += xi
            new_transition_den += gamma_k
            
            
        log_initial = log_gamma[0]
        initial = numpy.exp(log_initial)

        print('\nold_initial: {0}'.format(self._initial))
        print('new_initial: {0}'.format(initial))

        new_transition = (new_transition_num.T/new_transition_den).T
        
        print('\nold_transition:\n{0}'.format(self._transitions))
        print('\nnew_transition:\n{0}'.format(new_transition))

        new_emission_den = new_transition_den + gamma[-1]

        ind = numpy.array(range(self._n_symbols))
        new_emission_num = sum((numpy.outer(gamma_k, (ind == yk)) for yk, gamma_k in zip(observations, gamma)))
        new_emission = (new_emission_num.T/new_emission_den).T

        print('\nold_emission:\n{0}'.format(self._emissions))
        print('\nnew_emission:\n{0}'.format(new_emission))

