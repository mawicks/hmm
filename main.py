import hmm

foo = hmm.HMM()
foo.set_parameters(transitions = [[0.3, 0.7], [0.1, 0.9]],
                   emissions = [[0.4, 0.6], [0.5, 0.5]],
                   initial = [0.85, 0.15])
e, s = foo.sim(10)
e1 = [0, 1, 1, 0]
e2 = [1, 0, 1]

foo.update(e1)

print('***************')

foo.update(e2)



