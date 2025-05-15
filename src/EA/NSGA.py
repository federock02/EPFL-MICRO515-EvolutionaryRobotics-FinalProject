import copy
import os
from typing import Dict

import numpy as np

from src.utils.Filesys import search_file_list

NSGA_opts = {
    "min": -4,
    "max": 4,
    "num_parents": 16,
    "num_generations": 100,
    "mutation_prob": 0.3,
    "crossover_prob": 0.1,
}


class NSGAII():
    def __init__(self, n_pop, n_params, opts: Dict = NSGA_opts, output_dir: str = "./results/NSGAII"):
        """
        Evolutionary Strategy [INCOMPLETE]

        :param n_pop: population size
        :param n_params: number of parameters
        :param opts: algorithm options
        :param output_dir: output directory Default = "./results/NSGAII"
        """
        # % EA options
        self.n_params = n_params
        self.n_pop = n_pop
        self.n_gen = opts["num_generations"]
        self.n_parents = opts["num_parents"]
        self.min = opts["min"]
        self.max = opts["max"]

        self.current_gen = 0
        self.F = opts["mutation_prob"]
        self.Cr = opts["crossover_prob"]

        # % bookkeeping
        self.directory_name = output_dir
        self.full_x = []
        self.full_fitness = []
        self.x_best_so_far = None
        self.f_best_so_far = -np.inf
        self.x = None
        self.f = None


    def ask(self):
        if self.current_gen==0:
            new_population = self.initialise_x0()
        else:
            new_population = self.create_children(self.n_pop)
        new_population = np.clip(new_population, self.min, self.max)
        return new_population

    def tell(self, solutions, function_values, save_checkpoint=True):
        parents_population, parents_fitness = self.sort_and_select_parents(
            solutions, function_values, self.n_parents
        )


        #% Some bookkeeping
        self.full_fitness.append(function_values)
        self.full_x.append(solutions)
        self.x = parents_population
        self.f = parents_fitness

        aggregate_vals = np.sum(function_values, axis=1)
        if np.max(aggregate_vals) > np.sum(self.f_best_so_far):
            best_index = np.argmax(aggregate_vals)
            self.f_best_so_far = function_values[best_index]
            self.x_best_so_far = solutions[best_index]

        if self.current_gen % 5 == 0:
            print(f"Best fitness in generation {self.current_gen}: {self.f_best_so_far}\n"
                  f"Mean pop fitness: {self.f.mean()} +- {self.f.std()}\n"
                  )

        if save_checkpoint:
            self.save_checkpoint()
        self.current_gen += 1


    def initialise_x0(self):
        return np.random.uniform(self.min, self.max, size=(self.n_pop, self.n_params))

    def create_children(self, population_size):
        children = []
        parent_indices = np.arange(self.n_parents)
        crossover_factor = 2.0
        mutation_factor = 5.0

        while len(children) < population_size:
            # Select parents for crossover
            idx1, idx2 = np.random.choice(parent_indices, 2, replace=False)
            p1 = self.x[idx1]
            p2 = self.x[idx2]

            # Crossover
            if np.random.rand() < self.Cr:
                x_1 = (p1 + p2) / 2
                x_2 = np.abs((p1 - p2) / 2)

                u = np.random.uniform(0, 1, size=self.n_params) # Generate u per dimension
                beta = np.empty(self.n_params)

                # Calculate beta based on u for each dimension
                beta[u <= 0.5] = (2 * u[u <= 0.5]) ** (1 / (crossover_factor + 1))
                beta[u > 0.5] = (2 * (1 - u[u > 0.5])) ** (-1 / (crossover_factor + 1))

                c1 = x_1 + beta * x_2
                c2 = x_1 - beta * x_2
            else:
                c1 = p1.copy()
                c2 = p2.copy()

            # Polynomial Mutation for child 1
            for j in range(self.n_params):
                if np.random.rand() < self.F: # Apply mutation gene by gene based on prob
                    u = np.random.uniform()
                    if u <= 0.5:
                        delta = (2 * u) ** (1 / (mutation_factor + 1)) - 1
                        c1[j] += delta * (c1[j] - self.min)
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (mutation_factor + 1))
                        c1[j] += delta * (self.max - c1[j])

            # Polynomial Mutation for child 2
            for j in range(self.n_params):
                 if np.random.rand() < self.F: # Apply mutation gene by gene based on prob
                    u = np.random.uniform()
                    if u <= 0.5:
                        delta = (2 * u) ** (1 / (mutation_factor + 1)) - 1
                        c2[j] += delta * (c2[j] - self.min)
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (mutation_factor + 1))
                        c2[j] += delta * (self.max - c2[j])

            children.append(c1)
            if len(children) < population_size:
                children.append(c2)
        return np.array(children)



    def sort_and_select_parents(self, solutions, function_values, n_parents):
        fronts, population_rank = self.fast_nondominated_sort(function_values)

        prob_weight = len(fronts) - np.array(population_rank)
        dist = prob_weight / np.sum(prob_weight)

        draw_ind = np.random.choice(np.arange(len(solutions)), n_parents,
                      p=dist)
        return solutions[draw_ind], function_values[draw_ind]


    def dominates(self, individual, other_individual):
        return all(x >= y for x, y in zip(individual, other_individual)) and any(
            x > y for x, y in zip(individual, other_individual)
        )

    def fast_nondominated_sort(self, fitness):
        n_pop = len(fitness)
        domination_lists = [[] for _ in range(n_pop)]
        pareto_fronts = [[]]
        domination_counts = np.zeros(n_pop, dtype=int)
        population_rank = np.zeros(n_pop, dtype=int)

        for i in range(n_pop):
            for j in range(n_pop):
                if i == j:
                    continue

                # does individual i dominate individual j?
                if self.dominates(fitness[i], fitness[j]):
                    # Add j to the list of individuals dominated by i
                    domination_lists[i].append(j)

                # does individual j dominate individual i?
                elif self.dominates(fitness[j], fitness[i]):
                    # Increment the domination counter of i
                    domination_counts[i] += 1

            # If domination_counts[i] is 0, it means i is not dominated by any other individual
            if domination_counts[i] == 0:
                population_rank[i] = 0 # Rank 0 for the first front
                pareto_fronts[0].append(i)

        # Build subsequent fronts
        i = 0
        while pareto_fronts[i]:
            next_front = []
            # Iterate through individuals in the current front (pareto_fronts[i])
            for p_idx in pareto_fronts[i]:
                # Iterate through individuals dominated by p_idx
                for q_idx in domination_lists[p_idx]:
                    domination_counts[q_idx] -= 1
                    # If q_idx is no longer dominated, add it to the next front
                    if domination_counts[q_idx] == 0:
                        population_rank[q_idx] = i + 1
                        next_front.append(q_idx)

            i += 1
            # Add the newly identified front to the list of fronts
            if next_front: # Avoid adding empty lists if the last front was processed
                 pareto_fronts.append(next_front)
            else:
                 break # Exit if the next front is empty

        return pareto_fronts, population_rank

    def save_checkpoint(self):
        curr_gen_path = os.path.join(self.directory_name, str(self.current_gen))
        os.makedirs(curr_gen_path, exist_ok=True)
        np.save(os.path.join(self.directory_name, 'full_f'), np.array(self.full_fitness))
        np.save(os.path.join(self.directory_name, 'full_x'), np.array(self.full_x))
        np.save(os.path.join(curr_gen_path, 'f_best'), np.array(self.f_best_so_far))
        np.save(os.path.join(curr_gen_path, 'x_best'), np.array(self.x_best_so_far))
        np.save(os.path.join(curr_gen_path, 'x'), np.array(self.x))
        np.save(os.path.join(curr_gen_path, 'f'), np.array(self.f))

    def load_checkpoint(self):
        dir_path = search_file_list(self.directory_name, 'f_best.npy')
        assert len(dir_path) > 0;
        "No files are here, check the directory_name!!"

        self.current_gen = int(dir_path[-1].split('/')[-2])
        curr_gen_path = os.path.join(self.directory_name, str(self.current_gen))

        self.full_fitness = np.load(os.path.join(self.directory_name, 'full_f.npy'))
        self.full_x = np.load(os.path.join(self.directory_name, 'full_x.npy'))
        self.f_best_so_far = np.load(os.path.join(curr_gen_path, 'f_best.npy'))
        self.x_best_so_far = np.load(os.path.join(curr_gen_path, 'x_best.npy'))
        self.x = np.load(os.path.join(curr_gen_path, 'x.npy'))
        self.f = np.load(os.path.join(curr_gen_path, 'f.npy'))
