import numpy as np
import array
import random
import collections
from deap import algorithms
from deap import base
from deap import creator
from deap import tools


def run_GA(fitness_func, num_syls, strip_total_length, mut_amount=500, mut_prob=0.2, pop_size=500, num_gens=50):

    # single objective fitness (only optimize on one criteria, minimizing it)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    def jitter_duplicates(ind, jit_amt=20):
        while len(set(ind)) != len(ind):
            dups = collections.defaultdict(list)
            for i, x in enumerate(ind):
                dups[x].append(i)

            indices_to_jitter = []

            for key in dups.keys():
                indices_to_jitter += dups[key][1:]

            for i in indices_to_jitter:
                ind[i] += random.randint(-1 * jit_amt, jit_amt)
                ind[i] = max(0, ind[i])
                ind[i] = min(strip_total_length - 1, ind[i])

        return ind

    def individual_init(container):
        ind = np.random.randint(0, strip_total_length - 1, num_syls)
        ind = sorted(ind)
        return container(list(ind))

    def mut_jitter(ind, max_amt, indpb):
        for i in range(len(ind) - 1):
            if random.uniform(0, 1) > indpb:
                continue

            ind[i] += random.randint(-1 * max_amt, max_amt)

            # keep all positions within valid indices
            ind[i] = min(0, ind[i])
            ind[i] = max(strip_total_length - 1, ind[i])

        ind = jitter_duplicates(ind)
        ind.sort()
        return (ind,)

    def crossover_and_sort(ind1, ind2, minsize=3):
        pt2 = random.randint(3, len(ind1))
        pt1 = random.randint(0, pt2 - 3)

        slice1 = ind1[pt1:pt2]
        slice2 = ind2[pt1:pt2]

        ind1[pt1:pt2] = slice2
        ind2[pt1:pt2] = slice1

        ind1 = jitter_duplicates(ind1)
        ind2 = jitter_duplicates(ind2)

        ind1.sort()
        ind2.sort()

        return ind1, ind2

    # Structure initializers
    # toolbox.register("individual", tools.initRepeat, creator.Individual, init_geom, num_gaps)
    toolbox.register("individual", individual_init, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fitness_func)
    toolbox.register("mate", crossover_and_sort)
    # toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.25)
    toolbox.register("mutate", mut_jitter, max_amt=mut_amount, indpb=mut_prob)
    toolbox.register("select", tools.selTournament, tournsize=3)

    random.seed(64)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.55, mutpb=0.2, ngen=num_gens,
                                   stats=stats, halloffame=hof, verbose=True)

    return pop, log, hof


if __name__ == "__main__":
    main()
