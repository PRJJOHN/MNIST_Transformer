import torch
import os
from ImageTransDecoder import ImageTransformer
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms as transforms
# read the image
img_stack = torch.tensor([])
for i in range(10):
    filename = '{:02d}.bmp'.format(i)
    image = Image.open(filename)
    # convert to a PyTorch tensor
    tensor = transforms.ToTensor()(image)
    img_stack = torch.cat((img_stack, tensor),dim = 0)
img_stack = img_stack.cuda()
N_EPHOCS = 500
BATCH_SIZE = 128


#model = model.cuda()
N_OUTPUTS = 10
def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()
# Model instance
def main():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = ImageTransformer(in_channels=1, size=(28, 28), nfeatures=75, nclasses=10, nheads=1, dropout=0.0).to(device)
    if os.path.exists('mnist_trans.pt'):
        model= torch.load('mnist_trans.pt')
        model = model.cpu()
    else:
        model = ImageTransformer( size=(28, 28), nfeatures=90, nclasses=10, nheads=3, dropout=0.0).to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    #criterion = nn.KLDivLoss(size_average=False)
    criterion = criterion.cuda()
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(N_EPHOCS):  # loop over the dataset multiple times
        train_running_loss = 0.0
        train_acc = 0.0
        model.train()

        # TRAINING ROUND
        for i, data in enumerate(trainloader):
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            label_stack = torch.tensor([]).cuda()
            for idx in range( labels.size(0)):
                label_stack= torch.cat((label_stack, img_stack[labels[idx]].unsqueeze(0)),dim = 0)
            # forward + backward + optimize
            outputs = model(inputs)
            if outputs.size(0) == labels.size(0):
                loss = criterion(outputs, label_stack)
                loss.backward()
                optimizer.step()

                train_running_loss += loss.detach().item()
                # train_acc += get_accuracy(outputs, labels, BATCH_SIZE)
            
        model.eval()
        # print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f'  % (epoch, train_running_loss / i, train_acc/i))
        print('Epoch:  %d | Loss: %.4f '  % (epoch, train_running_loss / i))
        torch.save(model,'mnist_trans.pt')
if __name__ == '__main__':
    main()

