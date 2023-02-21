import torch
from ImageTransformer import ImageTransformer
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
N_EPHOCS = 30
BATCH_SIZE = 128

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model = model.cuda()
N_OUTPUTS = 10
def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()
# Model instance
model = ImageTransformer(in_channels=1, size=(28, 28), nfeatures=56, nclasses=10, nheads=1, dropout=0.0).to(device)
criterion = nn.CrossEntropyLoss()
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
        # forward + backward + optimize
        outputs = model(inputs)
        if outputs.size(0) == labels.size(0):
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          train_running_loss += loss.detach().item()
          train_acc += get_accuracy(outputs, labels, BATCH_SIZE)
         
    model.eval()
    print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f'  % (epoch, train_running_loss / i, train_acc/i))
    torch.save(model,'mnist_trans.pt')