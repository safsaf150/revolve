# (G,P)

class Individual:
	def __init__(self, genotype, phenotype=None):
		"""
		Creates an Individual object with the given genotype and optionally the phenotype.

		:param genotype: genotype of the individual
		:param phenotype (optional): phenotype of the individual
		"""
		self.genotype = genotype
		self.phenotype = phenotype
		self.fitness = None
		self.parents = None

	def develop(self):
		"""
		Develops genotype into a intermediate phenotype

		"""
		if self.phenotype is None:
			self.phenotype = self.genotype.develop()
