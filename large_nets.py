import torch
import torchvision

import numpy as np
import torch.nn as nn
import simpleNet as sp
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torchvision.transforms as transforms


def compute_error(data_loader, network):

	total   = 0
	correct = 0

	with torch.no_grad():

		for data in data_loader:

				images, labels = data

				images = images.to(device)
				labels = labels.to(device)

				output = network(images)
				_, predicted = torch.max(output.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

	return 100 * (correct/total)

def validate(testloader, criterion):

	'''
	Computa el valor de la función de costo
	sobre el conjunto de prueba.
	'''

	val_loss = 0
	for i, data in enumerate(testloader, 0):

		input_, label = data

		label  = label.to(device)
		input_ = input_.to(device)

		optimizer.zero_grad()

		output = net(input_)
		loss   = criterion(output, label)

		val_loss += loss.item()

	return val_loss

# Hiperparámetros de la red e información de la base de datos.
epochs        = 100
img_size      = 32
num_layers    = 50
num_maxPool   = 3
num_clases    = 10
learning_rate = 0.1

# Cargamos la base de datos.
print("Loading data...")
transform   = transforms.Compose([
								transforms.ToTensor(),
								transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset    = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset     = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader  = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

classes     = {"plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"}

# Creamos red.
net = sp.simpleNet(num_layers, num_maxPool, num_clases, img_size, batch_norm=True)
print(net)

# Si es posible enviamos a GPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if torch.cuda.device_count() > 1:
	net = nn.DataParallel(net)

net.to(device)

# Fase de entrenamiento.
print("Training...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
scheduler = lrs.ReduceLROnPlateau(optimizer, verbose=True, patience=3)

test_error  = []
train_error = []
total_loss  = []

test_error.append(compute_error(testloader, net))
train_error.append(compute_error(trainloader, net))

for epoch in range(epochs):

	batch_loss = 0
	for i, data in enumerate(trainloader, 0):

		input_, label = data

		label  = label.to(device)
		input_ = input_.to(device)

		optimizer.zero_grad()

		output = net(input_)
		loss   = criterion(output, label)
		loss.backward()
		optimizer.step()



		batch_loss += loss.item()

		if i % 50 == 49:    # print every 2000 mini-batches
			print('[%d, %5d] loss: %f' %
				(epoch + 1, i + 1, batch_loss / 2000))
			total_loss.append(batch_loss)
			batch_loss = 0.0

	val_loss = validate(testloader, criterion)
	scheduler.step(val_loss)

	test_error.append(compute_error(testloader, net))
	train_error.append(compute_error(trainloader, net))

	print("Epoch: %d, Loss: %f, Val_Loss %f" % (epoch, batch_loss, val_loss))

# Guardamos los resultado.
np.save("train_error.npy", np.array(train_error))
np.save("test_error.npy", np.array(test_error))
np.save("total_loss.npy", np.array(total_loss))