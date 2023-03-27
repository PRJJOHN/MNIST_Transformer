import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
N_EPHOCS = 30
BATCH_SIZE = 128
N_DIV = 1
def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()
def main():
    transform = transforms.Compose([transforms.ToTensor()])
    model= torch.load('mnist_trans.pt')

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_acc = 0.0
    model = model.cpu()
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        outputs = model(inputs)
        # plt.imshow(inputs[0][0])
        # plt.show()
        print(torch.max(outputs, 1)[1][0])
        print(labels[0])
        for id in range(196):
            plt.subplot(14,14,id+1)
            plt.imshow(model.attn.layers[0].self_attn.attn[0][0][id].detach().numpy().reshape(14,14))
        plt.show()
        test_acc += get_accuracy(outputs, labels, BATCH_SIZE)
            
    print('Test Accuracy: %.2f'%( test_acc/i))
if __name__ == '__main__':
    main()