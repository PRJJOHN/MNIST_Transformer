import torch
import torchvision
import torchvision.transforms as transforms
N_EPHOCS = 30
BATCH_SIZE = 128
def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()
def main():
    transform = transforms.Compose([transforms.ToTensor()])
    model= torch.load('mnist_trans.pt')

    # trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    test_acc = 0.0
    model = model.cpu()
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        #inputs = inputs.view(-1, 28, 28)

    #    zerofill = torch.zeros(inputs.size(0), 7, 28)
    #    inputs[:,0:7,:] = zerofill
        #imshow(inputs[0,:,:])
        outputs = model(inputs)

        test_acc += get_accuracy(outputs, labels, BATCH_SIZE)
            
    print('Test Accuracy: %.2f'%( test_acc/i))
if __name__ == '__main__':
    main()