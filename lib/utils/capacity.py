class capacity_control:
	def __init__(self, cont_capacity, disc_capacity):
		self.C_cont = cont_capacity[0]
		self.C_disc = disc_capacity[0]
		self.C_cont_max = cont_capacity[1]
		self.C_disc_max = disc_capacity[1]
		self.C_cont_step = (self.C_cont_max - self.C_cont) / cont_capacity[2]
		self.C_disc_step = (self.C_disc_max - self.C_disc) / disc_capacity[2]
		self.cont_gamma = cont_capacity[3]
		self.disc_gamma = disc_capacity[3]
	
	def step(self):
		if self.C_cont < self.C_cont_max:
			self.C_cont += self.C_cont_step
		if self.C_disc < self.C_disc_max:
			self.C_disc += self.C_disc_step

	def get_paras(self):
		return self.cont_gamma, self.C_cont, self.disc_gamma, self.C_disc