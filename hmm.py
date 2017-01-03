import logging
from tools import cumsum, find
from random import randrange, uniform
import numpy
import operator

logger = logging.getLogger()

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

def log_sum_exp(a):
    '''Compute log(sum[exp(row) for row in a]) without underflows'''
    col_max = numpy.max(a, axis=0)

    result = numpy.log(numpy.sum(numpy.exp(numpy.array([c - m for c, m in zip(a.T, col_max)])),
                       axis=1)) + col_max

    return result

def print_large(array, name):
    print ('{0}:'.format(name))
    
    if len(array) < 10:
        print(array)
    else:
        for r in array[:5]:
            print ('\t{0}'.format(r))
        print ('\t...')
        for r in array[-5:]:
            print ('\t{0}'.format(r))
    
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
        self._log_transitions = numpy.log(self._transitions+1e-30)
        
        self._emissions = numpy.array(emissions)
        self._log_emissions = numpy.log(self._emissions+1e-30)
        
        self._initial = numpy.array(initial)
        self._log_initial = numpy.log(self._initial+1e-30)
        
    def random_parameters(self, n_states, n_symbols):
        '''Set a random set of initial parameters.  Counts of observations are used to estimate an initial set of emission probabilities.
State transitions are set randomly.'''
        self.set_parameters(tuple(random_distribution(n_states) for _ in range(n_states)),
                            tuple(random_distribution(n_symbols) for _ in range(n_states)),
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
        '''Apply so-called "forward procedure" and return sequence of alpha probabilities'''
        logger.info('forward() entered')
        
        log_alpha = self._log_initial + self._log_emissions[:,observations[0]]
        log_alpha_series = [log_alpha]

        for y in observations[1:]:
            log_alpha = (self._log_emissions[:,y] +
                         log_sum_exp(numpy.array([log_alpha_i + log_trans_i
                                                                  for log_alpha_i, log_trans_i
                                                                  in zip(log_alpha, self._log_transitions)])))
            log_alpha_series.append(log_alpha)

        logger.info('forward() exited')
        return numpy.array(log_alpha_series)
    
    def backward(self, observations):
        '''Apply so-called "backward procedure" and return sequence of beta probabilities'''
        logger.info('backward() entered')

        log_beta = numpy.array([ 0.0 ] * self._n_states)
        log_beta_series = [log_beta]
        
        for y in reversed(observations[1:]):
            log_beta = log_sum_exp(numpy.array([log_beta_j + log_em_j[y] + log_trans_j
                                                                for log_beta_j, log_trans_j, log_em_j
                                                                in zip(log_beta, self._log_transitions.T, self._log_emissions)]))
            log_beta_series.append(log_beta)
            
        logger.info('backward() exited')
        return numpy.array(list(reversed(log_beta_series)))
        
    def update(self, observations):
        '''Perform one iteration of Baum-Welch algorithm and return new parameter estimate and observation likelihood:
        transition, emission, pi = update(observations)
The observation likelihood is for the observed sequence given the original parameters, not the new parameters.
'''
        logger.info('update() entered')

        log_alpha = self.forward(observations)
        log_beta = self.backward(observations)

        gamma_num = log_alpha + log_beta
        gamma = numpy.exp(numpy.array([n-d
                                       for n,d
                                       in zip(gamma_num,
                                              log_sum_exp(gamma_num.T))]))
        
#        print_large(log_alpha, 'log alpha')
#        print_large(log_beta, 'log beta')
#        print_large(gamma, 'gamma')
        
        new_transition_num = numpy.zeros((self._n_states, self._n_states))
        new_transition_den = numpy.zeros(self._n_states)

        for gamma_k, log_alpha_k, log_beta_k, y in zip(gamma[:-1], log_alpha[:-1], log_beta[1:], observations[1:]):
            log_xi_num = numpy.empty((self._n_states, self._n_states))
            for i in range(self._n_states):
                for j in range(self._n_states):
                    log_xi_num[i,j] = log_alpha_k[i] + log_beta_k[j] + self._log_transitions[i][j] + self._log_emissions[j][y]
            
            # This clumsy because numpy turns single rows/colums into vectors.
            # log_sum_exp requires a matrix argument, so we force it to be a matrix.
            log_xi_den = log_sum_exp(numpy.array([[r] for r in log_sum_exp(log_xi_num)]))
            
            xi = numpy.exp(log_xi_num - log_xi_den)
#            print('xi:\n{0}'.format(xi))
            
            new_transition_num += xi
            new_transition_den += gamma_k
            
        new_transition = (new_transition_num.T/new_transition_den).T

        ind = numpy.array(range(self._n_symbols))
        new_emission_num = sum((numpy.outer(gamma_k, (ind == yk)) for yk, gamma_k in zip(observations, gamma)))
        new_emission_den = new_transition_den + gamma[-1]
        new_emission = (new_emission_num.T/new_emission_den).T

        new_initial = gamma[0]

        # Same clumsy trick
        observation_likelihood = log_sum_exp(numpy.array([[la] for la in log_alpha[-1]]))
        
        logger.info('update() exited')
        return new_transition, new_emission, new_initial, float(observation_likelihood)

