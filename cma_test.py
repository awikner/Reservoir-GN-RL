import cma
from cma.fitness_functions import sphere  # cannot be an instance method
from cma.optimization_tools import EvalParallel2
## Test multiprocessing
es = cma.CMAEvolutionStrategy(2 * [0.0], 1.0, {'popsize':50})  # doctest:+ELLIPSIS
print('Testing Multiprocess')
with EvalParallel2(sphere, 5) as eval_all:
    while not es.stop():
        X = es.ask()
        es.tell(X, eval_all(X))
        # es.disp()
        # es.logger.add()  # doctest:+ELLIPSIS
## Test monoprocessing
es = cma.CMAEvolutionStrategy(2 * [0.0], 1.0, {'popsize':50})  # doctest:+ELLIPSIS
print('Testing Monoprocess')
with EvalParallel2(sphere, 0) as eval_all:
    while not es.stop():
        X = es.ask()
        es.tell(X, eval_all(X))
        # es.disp()
        # es.logger.add()  # doctest:+ELLIPSIS
## Test monoprocessing 2
print('Testing Monoprocess 2')
es = cma.CMAEvolutionStrategy(2 * [0.0], 1.0, {'popsize':50})  # doctest:+ELLIPSIS
es.optimize(sphere)
