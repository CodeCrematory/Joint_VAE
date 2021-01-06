from model.mine_vae import VAE
from lib.criterion import joint_vae_loss
from lib.dataloader import mnist_dataset, get_sampler
from lib.utils.capacity import capacity_control

import os
from os import path
import time
import datetime
import shutil
import ast

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import utils
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
import torch.nn as nn

class config:
	def __init__(self):
		self.base_path = 'D:/DataSource/basepath'
		self.dataset = "MNIST"
		self.image_size = 32

		self.workers = 4
		self.batch_size = 64
		self.number_mark = 1
		self.epochs = 200
		self.start_epoch = 0
		self.data_paralle = False #not used
		self.print_freq = 50

		self.resume = ''
		self.resume_arg = False

		self.mixup = False #not used
		self.manifold_mixup = False #not used
		self.mixup_layer_list = [] #not used
		self.mixup_alpha = 0 #not used

		self.net_name = 'VAE' 
		self.depth = 3
		self.hidden_dim = 256
		self.encoder_num_feature_maps = [1, 32, 64, 64]
		self.decoder_num_feature_maps = [64, 32, 32, 1]
		self.num_input_channel = 1
		self.cont_latent_num = 10
		self.disc_latent_size = [10]
		self.cont_capacity = [0.0, 5.0, 25000, 30.0]  # Starting at a capacity of 0.0, increase this to 5.0
                                         			# over 25000 iterations with a gamma of 30.0
		self.disc_capacity = [0.0, 5.0, 25000, 30.0]

		self.drop_rate = 0
		self.optimizer = 'SGD'
		self.learning_rate = 0.0005
		self.momentum = 0.9
		self.nesterov = True
		self.adjust_lr = [60,120,160]
		self.lr_decay_ratio = 0.2
		self.weight_decay = 5e-4
		self.warm_up_lr = 0.0005
		self.gpu = "0"
		pass

args = config()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def main(args = args):
	#build dataset
	if args.dataset == "MNIST":
		dataset_base_path = path.join(args.base_path, "dataset")
		train_dataset = mnist_dataset(dataset_base_path)
		test_dataset = mnist_dataset(dataset_base_path, train_flag = False)
		sampler_train = get_sampler(len(train_dataset))

		train_loader = DataLoader(train_dataset, batch_size = args.batch_size, num_workers = args.workers, pin_memory = True, sampler = sampler_train)
		test_loader = DataLoader(test_dataset, batch_size = args.batch_size, num_workers = args.workers, pin_memory = True)
	else:
		raise NotImplementError("Dataset {} Not Implemented".format(args.dataset))

	#model config
	use_gpu = False
	if args.gpu:
		use_gpu = True
	if args.net_name == "VAE":
		latent_space = {}
		if args.cont_latent_num != 0:
			latent_space['cont'] = args.cont_latent_num
		if args.disc_latent_size != []:
			latent_space['disc'] = args.disc_latent_size
		model = VAE(args.image_size, latent_space, args.num_input_channel, args.encoder_num_feature_maps, args.decoder_num_feature_maps,
					depth = args.depth, hidden_dim = args.hidden_dim, data_parallel = args.data_paralle,
					drop_rate = args.drop_rate, use_gpu = use_gpu)
	else:
		raise NotImplementError("Model {} Not Implemented".format(args.net_name))

	if use_gpu:
		model = model.cuda()
	
	#optim config
	input("Begin the {} time's training, Dataset:{}".format(args.number_mark, args.dataset))
	criterion = joint_vae_loss(args.image_size, use_gpu)

	if args.optimizer == 'SGD':
		optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate, momentum = args.momentum, weight_decay = args.weight_decay, nesterov = args.nesterov)
	else:
		raise NotImplementError("{} Not Found".format(args.optimizer))
	scheduler = MultiStepLR(optimizer, milestones = args.adjust_lr, gamma = args.lr_decay_ratio)

	#resume model
	writer_log_dir = "{}/{}/runs/number_mark:{}".format(args.base_path, args.dataset, args.number_mark)
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			if args.resume_arg:
				args = checkpoint['args']
				args.start_epoch = checkpoint['epoch']
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})"
					.format(args.resume, checkpoint['epoch']))
		else:
			raise FileNotFoundError("Checkpoint Resume File {} Not Found".format(args.resume))
	else:
		if os.path.exists(writer_log_dir):
			flag = input("{} number_mark:{} will be removed, input yes to continue:".format(
				args.dataset, args.number_mark))
			if flag == "yes":
				shutil.rmtree(writer_log_dir, ignore_errors=True)
	writer = SummaryWriter(log_dir=writer_log_dir)

	#run epochs
	cap = capacity_control(args.cont_capacity, args.disc_capacity)
	for epoch in range(args.start_epoch, args.epochs):
		scheduler.step(epoch)
		if epoch == 0:
			#warm up
			modify_lr_rate(opt = optimizer, lr = args.warm_up_lr)
		
		train(train_loader, model, criterion, optimizer, epoch, writer, cap, use_gpu)
		test(test_loader, model, criterion, epoch, writer, use_gpu)
		save_checkpoint({
				'epoch': epoch + 1,
				'args': args,
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict()
		})
		if epoch == 0:
			modify_lr_rate(opt = optimizer, lr = args.learning_rate)

#batches train
def train(train_loader, model, criterion, optimizer, epoch, writer, cap_ctrl, use_gpu):
	model.train()

	loss_sum = 0
	optimizer.zero_grad()
	for i, (image, _) in enumerate(train_loader):
		image = image.float()
		if use_gpu:
			image = image.cuda()
		rebuild, latent_paradict = model(image)

		gamma_cont, C_cont, gamma_disc, C_disc = cap_ctrl.get_paras()
		loss = criterion(image, rebuild, latent_paradict, gamma_cont = gamma_cont, gamma_disc = gamma_disc, C_cont = C_cont, C_disc = C_disc)
		cap_ctrl.step()

		loss.backward()
		loss_sum += loss.detach().cpu().numpy()

		optimizer.step()
		optimizer.zero_grad()

		if i % args.print_freq == 0:
			train_text = 'Epoch: [{0}][{1}/{2}]\t' \
						 'Time {sys_time}\t' \
						 'VAE Train Loss {cls_loss:.4f}'.format(
					epoch, i + 1, len(train_loader), sys_time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),cls_loss=loss.detach().cpu().numpy())
			print(train_text)
	writer.add_scalar(tag="Train/vae_loss", scalar_value=loss_sum / len(train_loader), global_step=epoch + 1)
	return

#test
def test(test_loader, model, criterion, epoch, writer, use_gpu):
	model.eval()

	loss_sum = 0
	for i, (image, _) in enumerate(test_loader):
		image = image.float()
		if use_gpu:
			image = image.cuda()
		rebuild, latent_paradict = model(image)
		loss = criterion(image, rebuild, latent_paradict)
		loss_sum += loss.detach().cpu().numpy()

	test_text = 'Epoch: [{0}][---/---]\t'\
				 'Time {sys_time}\t' \
				 'VAE test Loss {cls_loss:.4f}'.format(
				epoch, sys_time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),cls_loss=loss_sum / len(test_loader))
	print(test_text)
	writer.add_scalar(tag="test/vae_loss", scalar_value=loss_sum / len(test_loader), global_step=epoch + 1)

	return

#utils functions
def save_checkpoint(state, filename='checkpoint.pth.tar'):
	filefolder = "{}/{}/parameter/number_mark:{}".format(args.base_path, args.dataset, args.number_mark)
	if not path.exists(filefolder):
		os.makedirs(filefolder)
	torch.save(state, path.join(filefolder, filename))

def modify_lr_rate(opt, lr):
	for param_group in opt.param_groups:
		param_group['lr'] = lr

if __name__ == "__main__":
	#main run
	main()
	pass