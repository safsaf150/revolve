import numpy as np
import time

class Learning:
    def __init__(self, muation_rate=0.2, crossover_rate=0.8):
        """
        :param muation_rate:
        :param crossover_rate:
        """
        self.muation_rate = muation_rate
        self.crossover_rate = crossover_rate

    def vectorize_brain(self, brain):
        """
        Turn brain parameters into vector of values
        :param brain: BrainNN object to turn into vector
        :return: vector values and references to position in brain
        """        
        param_references = {}
        vector_values = []
        
        # create list of parameters
        for node in brain.nodes:
            if node in brain.params:
                param_references[node + '_period'] = len(vector_values)
                vector_values.append(brain.params[node].period)
                param_references[node + '_offset'] = len(vector_values)
                vector_values.append(brain.params[node].phase_offset)
                param_references[node + '_amplitude'] = len(vector_values)            
                vector_values.append(brain.params[node].amplitude)        
        
        return vector_values, param_references

    def devectorize_brain(self, vecorized_brain, brain_references, brain):
        """
        Cast vectorized values back into original brain
        :param vecorized_brain: vector values
        :param brain_references: references to position in brain
        :param brain: BrainNN object
        :return: devectorized brain
        """
        for node in brain.nodes:
            if node in brain.params:
                brain.params[node].period = vecorized_brain[brain_references[node + '_period']]
                brain.params[node].offset = vecorized_brain[brain_references[node + '_offset']]
                brain.params[node].amplitude = vecorized_brain[brain_references[node + '_amplitude']]     
        
        return brain   

    def construct_mutated_brains(self, individual, mutate_single_brain, brain_population_size=10):
        """
        :param individual: individual
        :param brain_population_size: population size of brains to evolve 
        :return: mutated brains
        """
        if individual.phenotype is None:
            return

        brain = individual.phenotype._brain
        
        vector_values, param_references = self.vectorize_brain(brain)

        if len(vector_values) > 0:
            # duplicate list to population size
            brains = np.tile(vector_values, (brain_population_size, 1))

            # mutate brain parameters uniformly according to normal distributions
            for i in range(brain_population_size):
                mutation_vals = np.random.normal(0, 1, len(vector_values) + 1)
                mutation_prob = np.random.uniform(0, 1, len(vector_values) + 1)        
                
                for j in range(len(vector_values)):
                    if self.muation_rate >= mutation_prob[j]:
                        mutated_param = brains[i][j] + mutation_vals[j]
                        if mutated_param > 10:
                            brains[i][j] = 10
                        elif mutated_param < 1:
                            brains[i][j] = 1
                        else:
                            brains[i][j] = mutated_param
                
                # mutate mutation probability
                if self.muation_rate >= mutation_prob[-1]:
                    mutated_prob = self.muation_rate + mutation_vals[-1]
                    if mutated_prob > 1:
                        self.muation_rate = 1
                    elif mutated_prob < 0:
                        self.muation_rate = 0
                    else:
                        self.muation_rate = mutated_prob
                if mutate_single_brain:
                    break

        # construct vectors back to brain objects
        mutated_brains = []
        for _ in range(brain_population_size):            
            constructed_brain = self.devectorize_brain(vector_values, param_references, individual.phenotype._brain)
            mutated_brains.append(constructed_brain)
            if mutate_single_brain:
                individual.phenotype._brain = mutated_brains[0]
                return individual
        return mutated_brains

    def crossover_brain(self, parents):
        """
        Perform crossover between parents and return offspring
        :param parents: list of parents
        :return: child resulting from crossover
        """
        vectorized_brains = []
        brains_param = []
        
        # construct list of vectorized values
        for parent in parents:
            vector_values, param_references = self.vectorize_brain(parent.phenotype._brain)
            vectorized_brains.append(vector_values)
            brains_param.append(param_references)

        # perform crossover on parents
        crossover_offspring = self.crossover(vectorized_brains)

        # turn vectorized offspring back into type of Individual
        devectorized_offspring = []
        for child_vector in crossover_offspring:
            child_brain = self.devectorize_brain(child_vector, brains_param[0], parents[0].phenotype._brain)
            child = parents[0]
            child.phenotype._brain = child_brain
            devectorized_offspring.append(child)

        return devectorized_offspring

    def crossover(self, vector_of_parents, offspring_size=1):
        """
        Perform Uniform crossover
        :param vector_of_parents: list of list of vector values of parents brain
        :param offspring_size: amount of children
        """
        offspring = []
        parent1, parent2 = vector_of_parents[0], vector_of_parents[1]
        
        # perform uniform crossover on parents to create offspring
        for _ in range(offspring_size):
            crossover_probs = np.random.uniform(0, 1, len(parent1))
            child = []
            for i in range(len(parent1)):
                if crossover_probs[i] > self.crossover_rate:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])
        
        return offspring
