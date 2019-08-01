from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch

def mnist_dataset(data_base_path, train_flag = True):
	#transform normalize to do...
	transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ])
	dataset = datasets.MNIST(root=data_base_path, train = train_flag, transform = transform, download = False)
	return dataset

def get_sampler(samples_num):
	#no valid set
	sampler_train = [i for i in range(samples_num)]
	sampler_train = SubsetRandomSampler(sampler_train)
	return sampler_train

if __name__ == "__main__":
	transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ])
	dataset = datasets.MNIST(root="D:\\DataSource\\basepath\\dataset", train = True, download = False, transform = transform)
	dataset = [k[0] for k in dataset]
	dataset = torch.cat(dataset)
	print(dataset.size())
	pass