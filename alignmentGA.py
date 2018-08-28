import numpy as np
import array
import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools


def run_GA(fitness_func, num_gaps, room_for_gaps, pop_size=100, num_gens=30):
    avg_gap_size = int(room_for_gaps / num_gaps)

    # single objective fitness (only optimize on one criteria, minimizing it)
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    def individual_init(container):
        ind = np.random.uniform(0, 1, num_gaps)
        ind = [int(round(x * room_for_gaps / sum(ind))) for x in ind]

        # correct for rounding errors
        diff = room_for_gaps - sum(ind)
        for i in range(abs(diff)):
            temp = -1
            while temp < 1:
                rand_index = random.randint(0, num_gaps - 1)
                temp = ind[rand_index]
            ind[rand_index] += np.sign(diff)
        return container(list(ind))

    def mut_shift_adjacent(ind, max_amt, indpb):
        for i in range(len(ind) - 1):
            if random.uniform(0, 1) > indpb:
                continue

            sub_index = random.randint(0, 1)
            add_index = 1 - sub_index
            sub_index += i
            add_index += i

            amt = random.randint(0, min(max_amt, ind[sub_index]))
            ind[sub_index] -= amt
            ind[add_index] += amt

        return (ind,)

    def keep_sum_crossover(ind1, ind2, minsize=3):
        pt2 = random.randint(3, len(ind1))
        pt1 = random.randint(0, pt2 - 3)

        slice1 = ind1[pt1:pt2]
        slice2 = ind2[pt1:pt2]
        sum_slice1 = max(sum(slice1), 1)
        sum_slice2 = max(sum(slice2), 1)
        # shrink or expand slices so that they each fit in the other's sequence while maintaining the sum
        slice1 = [int(round(x * sum_slice2 / sum_slice1)) for x in slice1]
        slice2 = [int(round(x * sum_slice1 / sum_slice2)) for x in slice2]

        ind1[pt1:pt2] = slice2
        ind2[pt1:pt2] = slice1

        return ind1, ind2

    # Structure initializers
    # toolbox.register("individual", tools.initRepeat, creator.Individual, init_geom, num_gaps)
    toolbox.register("individual", individual_init, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fitness_func)
    toolbox.register("mate", keep_sum_crossover)
    # toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.25)
    toolbox.register("mutate", mut_shift_adjacent, max_amt=avg_gap_size, indpb=0.15)
    toolbox.register("select", tools.selTournament, tournsize=3)

    random.seed(64)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.75, mutpb=0.25, ngen=num_gens,
                                   stats=stats, halloffame=hof, verbose=True)

    return pop, log, hof


if __name__ == "__main__":
    main()
