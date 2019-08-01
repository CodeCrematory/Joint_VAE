import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class joint_vae_loss(nn.Module):
	def __init__(self, image_size, use_gpu = False):
		super(joint_vae_loss, self).__init__()
		self.rebuild_error = 0
		self.cont_kl_error = 0
		self.disc_kl_error = 0
		self.num_pixels = image_size**2
		self.use_gpu = use_gpu
		
	def forward(self, origin_data, rebuild_data, latent_paras, gamma_cont = 1, gamma_disc = 1, C_cont = 0, C_disc = 0):
		#self.rebuild_error = F.mse_loss(origin_data, rebuild_data, reduction = "mean") / 2
		self.rebuild_error = F.binary_cross_entropy(rebuild_data.view(-1, self.num_pixels),origin_data.view(-1, self.num_pixels))
		self.rebuild_error *= self.num_pixels
		#cont kl
		if 'cont' in latent_paras:
			mean, logvar = latent_paras['cont']
			self.cont_kl_error = self._cont_kl_with_norm(mean, logvar)
		#disc kl
		if 'disc' in latent_paras:
			self.disc_kl_error = self._multiple_disc_kl(latent_paras['disc'])
		total = self.rebuild_error + gamma_cont * torch.abs(self.cont_kl_error - C_cont) + gamma_disc * torch.abs(self.disc_kl_error - C_disc)
		return total

	def _cont_kl_with_norm(self, mean, logvar):
		kl_values = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
		kl_means = torch.mean(kl_values, dim = 0)
		return torch.sum(kl_means)

	def _multiple_disc_kl(self, alphas):
		loss = [self._single_disc_kl(alpha) for alpha in alphas]
		losses = torch.sum(torch.cat(loss))
		return losses

	def _single_disc_kl(self, alpha):
		EPS = 1e-12
		disc_dim = int(alpha.size()[-1])
		log_dim = torch.Tensor([np.log(disc_dim)])
		if self.use_gpu:
			log_dim = log_dim.cuda()
		entropy = torch.sum(alpha * torch.log(alpha + EPS), dim = 1)
		mean_entropy = torch.mean(entropy, dim = 0)
		return log_dim + mean_entropy

if __name__ == "__main__":
	a = torch.Tensor([[1,2,3],[4,5,6]])
	b = torch.Tensor([[1,2,3],[4,5,6]])
	latent_paras = {'cont': [torch.Tensor([[2,3],[3,3]]), torch.Tensor([[0.1,0.1],[0.1,0.1]])], 'disc': [torch.Tensor([[0.3,0.7],[0.9,0.1]])]}
	loss = joint_vae_loss()
	print(loss(a,b,latent_paras))
	pass

