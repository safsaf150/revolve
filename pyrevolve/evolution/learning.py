import asyncio
import os
import sys
import cma
from pyrevolve.gazebo.manage import WorldManager as World
from pyrevolve.SDF.math import Vector3


class Learning:
    def __init__(self, individual, max_age_robot=10):
        """
        :param individual: individual to perform learning on brain
        """
        self.individual = individual
        self.vector_values = None
        self.param_references = None
        self.max_age_robot = max_age_robot

    def vectorize_brain(self):
        """
        Cast brain parameters into vector of values
        """
        self.param_references = {}
        self.vector_values = []
        brain = self.individual.phenotype._brain

        if brain is None:
            return

        # vectorize parameter values
        for node in brain.nodes:
            if node in brain.params:
                self.param_references[node +
                                      '_period'] = len(self.vector_values)
                self.vector_values.append(brain.params[node].period)
                self.param_references[node +
                                      '_offset'] = len(self.vector_values)
                self.vector_values.append(brain.params[node].phase_offset)
                self.param_references[node +
                                      '_amplitude'] = len(self.vector_values)
                self.vector_values.append(brain.params[node].amplitude)

    def devectorize_brain(self, vecorized_brain, brain_references):
        """
        Cast vectorized values back into original brain parameters
        """
        brain = self.individual.phenotype._brain

        if brain is None or self.vector_values is None or self.param_references is None:
            return

        # cast vector values back into brain
        for node in brain.nodes:
            if node in brain.params:
                brain.params[node].period = self.vector_values[self.param_references[node + '_period']]
                brain.params[node].offset = self.vector_values[self.param_references[node + '_offset']]
                brain.params[node].amplitude = self.vector_values[self.param_references[node + '_amplitude']]

        self.individual.phenotype._brain = brain

    async def evaluate_single_robot(self, individual):
        """
        Evaluate single robot in simulation and return fitness
        :param individual:
        :return: fitness of evaluated individual
        """
        world = await World.create()
        insert_future = await world.insert_robot(individual.phenotype, Vector3(0, 0, 0.25))
        robot_manager = await insert_future
        await world.pause(False)
        while robot_manager.age() < self.max_age_robot:
            pass
        print('fitness: {}'.format(robot_manager.fitness()))
        return robot_manager.fitness()

    def cma_es_evaluate_vector(self, vector):
        """
        :param vector: list of vector values to evaluate in individual
        :return: fitness of vector
        """
        if self.param_references is None:
            return

        self.devectorize_brain(vector, self.param_references)

        # TODO: PERFORM EVALUTATION ON INDIVIDUAL
        self.individual.fitness = self.evaluate_single_robot(self.individual)

        return self.individual.fitness

    def vector_cma_es(self):
        """
        Covariance matrix adaptation evolution strategy
        :return: best vector result
        """
        if self.individual.phenotype._brain is None:
            return

        self.vectorize_brain()

        # cma evolution strategy with sigma set to 0.1
        cma_strategy = cma.CMAEvolutionStrategy(self.vector_values, 0.1)
        cma_strategy.optimize(self.cma_es_evaluate_vector)

        # return best vector
        return cma_strategy.result.xbest

    def learn_brain_through_cma_es(self):
        """
        Learn brain parameters by optimization through CMA-ES
        :return: individual with learned brain parameters
        """
        best_vector = self.vector_cma_es()
        self.devectorize_brain(best_vector, self.param_references)
        return self.individual