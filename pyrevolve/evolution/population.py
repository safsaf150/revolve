# [(G,P), (G,P), (G,P), (G,P), (G,P)]

from pyrevolve.evolution.individual import Individual
from pyrevolve.evolution import selection
from pyrevolve.evolution.learning import Learning
from pyrevolve.SDF.math import Vector3
import time
import asyncio


class PopulationConfig:
    def __init__(self,
                 population_size: int,
                 genotype_constructor,
                 genotype_conf,
                 fitness_function,
                 mutation_operator,
                 mutation_conf,
                 crossover_operator,
                 crossover_conf,
                 selection,
                 parent_selection,
                 population_management,
                 population_management_selector,
                 evaluation_time,
                 learning,
                 learning_brain_pop_size,
                 learning_num_generations,
                 offspring_size=None):
        """
        Creates a PopulationConfig object that sets the particular configuration for the population

        :param population_size: size of the population
        :param genotype_constructor: class of the genotype used
        :param genotype_conf: configuration for genotype constructor
        :param fitness_function: function that takes in a `RobotManager` as a parameter and produces a fitness for the robot
        :param mutation_operator: operator to be used in mutation
        :param mutation_conf: configuration for mutation operator
        :param crossover_operator: operator to be used in crossover
        :param selection: selection type
        :param parent_selection: selection type during parent selection
        :param population_management: type of population management ie. steady state or generational
        :param learning: perform learning on brain
        :param learning_brain_pop_size: population size for brain learning
        :param learning_num_generations: number of generation for brain learning        
        :param offspring_size (optional): size of offspring (for steady state)
        """
        self.population_size = population_size
        self.genotype_constructor = genotype_constructor
        self.genotype_conf = genotype_conf
        self.fitness_function = fitness_function
        self.mutation_operator = mutation_operator
        self.mutation_conf = mutation_conf
        self.crossover_operator = crossover_operator
        self.crossover_conf = crossover_conf
        self.parent_selection = parent_selection
        self.selection = selection
        self.population_management = population_management
        self.population_management_selector = population_management_selector
        self.evaluation_time = evaluation_time
        self.offspring_size = offspring_size
        self.learning = learning
        self.learning_brain_pop_size = learning_brain_pop_size
        self.learning_num_generations = learning_num_generations


class Population:
    def __init__(self, conf: PopulationConfig, simulator):
        """
        Creates a Population object that initialises the
        individuals in the population with an empty list
        and stores the configuration of the system to the
        conf variable.

        :param conf: configuration of the system
        :param simulator: connection to the simulator
        """
        self.conf = conf
        self.individuals = []
        self.simulator = simulator

    async def init_pop(self):
        """
        Populates the population (individuals list) with Individual objects that contains their respective genotype.
        """
        for i in range(self.conf.population_size):
            individual = Individual(self.conf.genotype_constructor(self.conf.genotype_conf))
            self.individuals.append(individual)

        await self.evaluate(self.individuals, 0)

    async def evolve_genotype_only(self, individual):
        """
        Learn brain parameters through evolution
        :param individual: individual to evolve brain
        :param generations: number of generations
        :return: individual with learned brain
        """
        # generate identical (phenotype) population with mutated genotype
        learning = Learning()
        brains = learning.construct_mutated_brains(individual, False, self.conf.learning_brain_pop_size)
        individuals_with_mutated_brain = []
        for brain in brains:
            new_individual = individual
            new_individual.phenotype._brain = brain
            individuals_with_mutated_brain.append(new_individual)

        # evalutate mutated population
        evaluated_individuals = []
        for new_individual in individuals_with_mutated_brain:
            await self.evaluate_single_robot(new_individual)
            evaluated_individuals.append(new_individual)

        # perform evolution step
        gen_num = 0
        population = evaluated_individuals
        while gen_num < self.conf.learning_num_generations:
            gen_num += 1
            population = await self.next_gen_brain_learning(gen_num, population)
        
        # return best learned individual
        best_learned_individual = selection.tournament_selection(population, len(population))
        return best_learned_individual

    async def next_gen_brain_learning(self, gen_num, individuals):
        """
        Creates next generation of the population through selection, mutation, crossover for genotype evolution
        :param gen_num: generation number
        :return: new population        
        """       
        new_individuals = []
        learning = Learning()

        for _i in range(self.conf.offspring_size):
            # parent selection and crossover
            parents = self.conf.parent_selection(individuals)
            child = self.conf.crossover_operator(parents, self.conf.crossover_conf)

            # mutate offspring
            mutated_child = learning.construct_mutated_brains(child, True)
            new_individuals.append(mutated_child)
        
        # survivor selection
        if self.conf.population_management_selector is not None:
            new_individuals = self.conf.population_management(self.individuals, new_individuals,
                                                              self.conf.population_management_selector)
        else:
            new_individuals = self.conf.population_management(self.individuals, new_individuals)

        return new_individuals

    async def next_gen(self, gen_num):
        """
        Creates next generation of the population through selection, mutation, crossover

        :param gen_num: generation number

        :return: new population
        """

        new_individuals = []

        for _i in range(self.conf.offspring_size):
            # Selection operator (based on fitness)
            # Crossover
            if self.conf.crossover_operator is not None:
                parents = self.conf.parent_selection(self.individuals)
                child = self.conf.crossover_operator(parents, self.conf.crossover_conf)
            else:
                child = self.conf.selection(self.individuals)
            # Mutation operator
            child_genotype = self.conf.mutation_operator(child.genotype, self.conf.mutation_conf)
            # Insert individual in new population
            individual = Individual(child_genotype)
            new_individuals.append(individual)

        # evaluate new individuals
        await self.evaluate(new_individuals, gen_num)

        # create next population
        if self.conf.population_management_selector is not None:
            new_individuals = self.conf.population_management(self.individuals, new_individuals,
                                                              self.conf.population_management_selector)
        else:
            new_individuals = self.conf.population_management(self.individuals, new_individuals)
        # return self.__class__(self.conf, new_individuals)
        new_population = Population(self.conf, self.simulator)
        new_population.individuals = new_individuals
        return new_population

    async def evaluate(self, new_individuals, gen_num):
        """
        Evaluates each individual in the new gen population

        :param new_individuals: newly created population after an evolution itertation
        :param gen_num: generation number
        """
        # Parse command line / file input arguments
        await self.simulator.pause(True)
        # await self.simulator.reset(rall=True, time_only=False, model_only=False)
        # await asyncio.sleep(2.5)

        for individual in new_individuals:
            print(f'---\nEvaluating individual (gen {gen_num}) {individual.genotype.id} ...')
            individual.develop()
            
            if self.conf.learning:
                # learn individual's brain
                individual = await self.evolve_genotype_only(individual)

            await self.evaluate_single_robot(individual)
            print(f'Evaluation complete! Individual {individual.genotype.id} has a fitness of {individual.fitness}.')

    async def evaluate_single_robot(self, individual):
        """
        Evaluate an individual

        :param individual: an individual from the new population
        """
        # Insert the robot in the simulator
        insert_future = await self.simulator.insert_robot(individual.phenotype, Vector3(0, 0, 0.25))
        robot_manager = await insert_future

        # Resume simulation
        await self.simulator.pause(False)

        # Start a run loop to do some stuff
        max_age = self.conf.evaluation_time # + self.conf.warmup_time
        while robot_manager.age() < max_age:
            individual.fitness = self.conf.fitness_function(robot_manager)
            await asyncio.sleep(1.0 / 5) # 5= state_update_frequency

        await self.simulator.pause(True)
        delete_future = await self.simulator.delete_robot(robot_manager)
        await delete_future
        # await self.simulator.reset(rall=True, time_only=False, model_only=False)
