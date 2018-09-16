import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Net(nn.Module):

	def __init__(self, conv_layers, maxPool, num_clases, img_size):
		super(Net, self).__init__()

		self.img_size    = img_size
		self.maxPool     = maxPool
		convOut          = self.compute_convolution_output_size()

		self.conv_layers  = conv_layers
		self.classifier   = nn.Sequential(
			nn.Linear((convOut ** 2) * 64 * (2 ** (self.maxPool)), 1000),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(1000, num_clases)
		)

		self._initialize_weights()

	def forward(self, x):

		x = self.conv_layers(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)

		return x

	def compute_convolution_output_size(self):

		'''
		Calcula la altura y longitud de la red neuronal
		previo a la capa de clasificación.
		'''

		convOut = self.img_size
		for i in range(self.maxPool):
			convOut = int(((convOut - 2)/2) + 1)
		return convOut

	def _initialize_weights(self):

		for m in self.modules():

			if isinstance(m, nn.Conv2d):
				nn.init.xavier_normal_(m.weight)
				nn.init.constant_(m.bias, 0)

			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

def make_layers(num_layers, num_maxPool, batch_norm):

	maxPoolFlag   = False
	prev_channels = 3
	channels      = 3
	num_filters   = 64
	layers        = []
	indexes       = maxPool_layers(num_layers, num_maxPool)

	for i in range(num_layers):

		conv2d = nn.Conv2d(channels, num_filters, kernel_size=3, padding=1)
		channels = num_filters

		if batch_norm:
			layers += [conv2d, nn.BatchNorm2d(num_filters), nn.ReLU()]
		else:
			layers += [conv2d, nn.ReLU()]

		if i in indexes:
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			num_filters *= 2

	return nn.Sequential(*layers)

def maxPool_layers(num_layers, num_maxPool):

	'''
	Regresa una lista que contiene los índices en 
	los cuales aplicar una capa de muestreo
	'''

	spacing   = int(num_layers / (num_maxPool + 1))
	remainder = num_layers % (num_maxPool + 1)

	indexes   = [ spacing for _ in range(num_maxPool)]

	for i in range(remainder):
		indexes[i] += 1

	for i in range((len(indexes)-1), -1, -1):
		indexes[i] = sum(indexes[0:i+1])

	indexes = [ element - 1 for element in indexes]

	return indexes

def simpleNet(num_layers, num_maxPool, num_clases, img_size, batch_norm=False):

	layers = make_layers(num_layers, num_maxPool, batch_norm)
	model  = Net(layers, num_maxPool, num_clases, img_size)

	return model