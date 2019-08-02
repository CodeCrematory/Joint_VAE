import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as numpy
import random
from math import floor

class VAE(nn.Module):
	def __init__(self, img_size, latent_space, num_input_channel, encoder_num_feature_map = [1, 32, 64, 64], decoder_num_feature_map = [64, 32, 32, 1],
				 depth = 3, hidden_dim = 256, data_parallel = False, drop_rate = 0.0, use_gpu = False):
		super(VAE, self).__init__()
		self.is_continous = 'cont' in latent_space
		self.is_discrete = 'disc' in latent_space
		#paras saved
		self.img_size = img_size
		self.latent_space = latent_space
		self.num_input_channel = num_input_channel
		self.encoder_num_feature_map = encoder_num_feature_map # must be a depth+1 array
		self.decoder_num_feature_map = decoder_num_feature_map
		self.depth = depth
		self.hidden_dim = hidden_dim
		self.data_parallel = data_parallel
		self.drop_rate = drop_rate
		self.temperature = 1

		self.feature_map_size = img_size  # H_new = (H+2*padding-kernel_size)/stride + 1
		for i in range(depth):
			self.feature_map_size = floor((self.feature_map_size + 2*1 - 4)/2 + 1) 
		self.use_gpu = use_gpu
		assert self.feature_map_size > 1, 'the depth is too big.'

		#latent dimensions
		self.latent_cont_dim = 0
		self.latent_disc_dim = 0
		self.num_disc_latents = 0
		if self.is_continous:
			self.latent_cont_dim = latent_space['cont']
		if self.is_discrete:
			self.latent_disc_dim += sum([dim for dim in latent_space['disc']])
			self.num_disc_latents = len(latent_space['disc'])
		self.latent_dim = self.latent_cont_dim + self.latent_disc_dim
		#modules
		self.encoder = nn.Sequential()
		self.encoder_to_hidden = nn.Sequential()
		self.decoder_latent2feature = nn.Sequential()
		self.decoder = nn.Sequential()
		#encoder
		for i in range(depth):
			self.encoder.add_module("conv%d"%(i+1), nn.Conv2d(self.encoder_num_feature_map[i], self.encoder_num_feature_map[i + 1], kernel_size = 4, stride = 2, padding = 1, bias = True))
			self.encoder.add_module("relu%d"%(i+1), nn.ReLU())
		
		self.encoder_to_hidden.add_module("fc", nn.Linear(self.encoder_num_feature_map[depth] * self.feature_map_size * self.feature_map_size, hidden_dim))
		self.encoder_to_hidden.add_module("relu", nn.ReLU())
		#encode to distribution paras
		if self.is_continous:
			self.cont_mean = nn.Linear(hidden_dim, self.latent_cont_dim)
			self.cont_log_var = nn.Linear(hidden_dim, self.latent_cont_dim)
		if self.is_discrete:
			alphas = []
			for dim in latent_space['disc']:
				alphas.append(nn.Linear(hidden_dim, dim))
			self.alphas = nn.ModuleList(alphas)
		#latent variabels to features
		self.decoder_latent2feature.add_module("fc1", nn.Linear(self.latent_dim, hidden_dim))
		self.decoder_latent2feature.add_module("relu1", nn.ReLU())
		self.decoder_latent2feature.add_module("fc2", nn.Linear(hidden_dim, decoder_num_feature_map[0] * self.feature_map_size * self.feature_map_size))
		self.decoder_latent2feature.add_module("relu2", nn.ReLU())
		
		#decoder
		for i in range(depth):
			self.decoder.add_module("convt%d"%(i+1), nn.ConvTranspose2d(decoder_num_feature_map[i], decoder_num_feature_map[i+1], kernel_size = 4, stride = 2, padding = 1, bias = True))
			if i != depth-1:
				self.decoder.add_module("relu%d"%(i+1), nn.ReLU())
		self.decoder.add_module("sigmoid", nn.Sigmoid())
		pass
	def encode_phase(self, x):
		feature_maps = self.encoder(x)
		features = self.encoder_to_hidden(feature_maps.view(feature_maps.size()[0],-1))

		latents_paras = {}
		if self.is_continous:
			latents_paras['cont'] = [self.cont_mean(features), self.cont_log_var(features)]
		if self.is_discrete:
			latents_paras['disc'] = []
			for alpha in self.alphas:
				latents_paras['disc'].append(F.softmax(alpha(features), dim = 1))

		return latents_paras

	def sample_normal(self, mean, logvar):
		if self.training:
			std = torch.exp(0.5 * logvar)
			eps = torch.zeros(std.size()).normal_()
			if self.use_gpu:
				eps = eps.cuda()
			return mean + std * eps
		else:
			return mean
	
	def sample_gumbel_softmax(self, alpha):
		EPS = 1e-12
		if self.training:
			unif = torch.rand(alpha.size())
			if self.use_gpu:
				unif = unif.cuda()
			gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
			log_alpha = torch.log(alpha + EPS)
			logit = (log_alpha + gumbel)/self.temperature
			return F.softmax(logit, dim = 1)
		else:
			_, max_alpha = torch.max(alpha, dim = 1)
			one_hot_samples = torch.zeros(alpha.size())
			one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
			if self.use_gpu:
				one_hot_samples = one_hot_samples.cuda()
			return one_hot_samples

	def reparameterize_phase(self, latents_paras):
		latents_sample = []

		if self.is_continous:
			mean, logvar = latents_paras['cont']
			cont_sample = self.sample_normal(mean, logvar)
			latents_sample.append(cont_sample)
		if self.is_discrete:
			for alpha in latents_paras['disc']:
				disc_sample = self.sample_gumbel_softmax(alpha)
				latents_sample.append(disc_sample)
		return torch.cat(latents_sample, dim = 1)

	def decode_phase(self, latents_sample):
		features = self.decoder_latent2feature(latents_sample)
		feature_maps =  features.view(-1, self.encoder_num_feature_map[self.depth], self.feature_map_size, self.feature_map_size)
		return self.decoder(feature_maps)

	def forward(self, x):
		latents_paras = self.encode_phase(x)
		latents_sample = self.reparameterize_phase(latents_paras)
		reconstruction = self.decode_phase(latents_sample)
		return reconstruction, latents_paras


if __name__ == "__main__":
	model = VAE(32, {'cont': 10, 'disc': [10]}, 1)
	a = torch.rand(64,1,32,32)
	a = model.encoder(a)
	print(a.size())
	pass