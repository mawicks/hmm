import hmm

# Example from: http://www.indiana.edu/~iulg/moss/hmmcalculations.pdf
example1 = hmm.HMM()
example1.set_parameters(transitions = [[0.3, 0.7], [0.1, 0.9]],
                   emissions = [[0.4, 0.6], [0.5, 0.5]],
                   initial = [0.85, 0.15])
e1 = [0, 1, 1, 0]
e2 = [1, 0, 1]

example1.update(e1)
example1.update(e2)

# Example 2
print('********* Example 2 ********')

# Set some known parameters and use them to generate an observation sequence.
example2 = hmm.HMM()
example2.set_parameters(transitions = [[0.95, 0.05], [0.50, 0.50]],
                        emissions = [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]],
                        initial = [0.25, 0.75])
e, s = example2.sim(10000)

d = {}
for o in e:
    d[o] = d.get(o, 0) + 1

print('counts: {0}'.format(d))
counts = [d[o] for o in sorted(d.keys())]
print('counts: {0}'.format(counts))

# Set incorrect parameters as an initial guess and try to estimate the original parameters
# from the observations
example3 = hmm.HMM()
example3.set_parameters(transitions = [[0.6, 0.4], [0.4, 0.6]],
                        emissions = [[0.33, 0.33, 0.34], [0.05, 0.55, 0.40]],
                        initial = [0.3, 0.7])
example3.random_parameters(2, 3)

print('observations: {0}'.format(e[:50]))
print('actual state sequence: {0}'.format(s[:50]))

for i in range(50):
    print (' ***** ITERATION {0} *****'.format(i))
    tr, em, init, likelihood = example3.update(e)
    if i == 0:
        original_likelihood = likelihood

    print('current log likelihood: {0}'.format(likelihood))
    example3.set_parameters(transitions = tr, emissions = em, initial = init)

print('initial log likelihood: {0}'.format( original_likelihood))
print('final log likelihood: {0}'.format(likelihood))
    
print('final parameters:')
print('transitions:\n{0}'.format(tr))
print('emissions:\n{0}'.format(em))
print('initial:\n{0}'.format(init))
