
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import os

# 1. download dataset
# 2. create data loader
# 3. build mode
# 4. train
# 5. save trained model

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .001

#define a model class
class FeedForwardNet(nn.Module): #model class should inherit from Module

    #constructor
    # where you define all the layers
    def __init__(self):
        super().__init__() #inherit from base class
        # store attribs for class
        # model will be an initial layer that will flatten the data (mnsit images)
        # convert to 1d array
        # some layers: input, output
        # softmax

        # layer 1: flatten 
        self.flatten = nn.Flatten() #self.flatten is arbitrary
        # multiple layers
        # will contain more than one layer
        # data will flow from one layer to the next
        self.dense_layers =  nn.Sequential(
            nn.Linear(28*28, 256),  #like "dense layer" in keras, args input features, output features
                            # input: 28*28 images in mnsist are 28 b 28 pixels , they have been flattened into a 1d array
                            # output: 256 neurons ???
            nn.ReLU(), #activation function: Applies the rectified linear unit function element-wise:
            nn.Linear(256, 10) # input: 256: bc linear output is 256 and relu is still 256
                                # output: 10: number of classes in mnsit digits [0..9]                   
        ) 
        # final layer
        # softmax, basic transformation
        # takes all values for the 10 classes and transofrms them so that the sum = 1
        # sort of normalization
        self.softmax = nn.Softmax(dim=1)

    # defines the dataflow, forward pass in network
    # next, define the forward method
    # we tell pytorch how to process the data
    # it indicates how to manipulate the data in what sequence
    def forward(self, input_data):
        flattened_data = self.flatten(input_data) #pass input data into the flatten layer
        # pass the flattened data to the dense layer ant get "logits": outputs
        logits = self.dense_layers(flattened_data)
        # pass logits to softmax to get the predictions
        predictions = self.softmax(logits)
        return predictions

def download_mnist_datasets():

    #dataset class allows us to store the labeles and sample
    #our dataset class is MNIST, already implemented and stored in pytorch
    #later in series, we will learn how to make our own dataset

    train_data = datasets.MNIST(
        root = "data", #dataset will be stored under a new folder called "/data"
                       #will be stored in the root directory
        download = True,
        train = True, #interested in the train part of the dataset
        transform = ToTensor() #allows us to apply transformation to dataset
                                # each value is normalized from [0..1]
    )

    validation_data = datasets.MNIST(
        root = "data",
        download = True,
        train = False, #interested in the nontrain part of the dataset (validation data)
        transform = ToTensor()
    )

    return train_data, validation_data

# training one epoch of the model
def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    # will loop through samples of dataset
    # at evey iteration, will return the x and y of each iteration

    for inputs, targets in data_loader:
        # must assign tensors, inputs and targets to a device
        inputs, targets = inputs.to(device), targets.to(device)

        # basic processing of a nn below
            # calc loss
            # use gradient descent to backpropogate and update the weigths
            # deep learning theory

        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)  #want predictions and targets (expected values), compute loss with this

        # backpropogate loss and use gradient descent to update weights
        # take optimizer and apply zero_grad
        # at every iter the optimzer calculated gradients
            # at each iter they get saves
            # in each batch we want to reset gradients to 0
        optimizer.zero_grad()
        # backward propogation
        loss.backward()
        # update the weights
        optimizer.step()

    print(f"Loss: {loss.item()}") # printing the loss for the last batch


# for each epoch call train_one_epoch
def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f"Epoch: {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print("-------------------------")
    print("training complete")

#create a script rq

if __name__ == "__main__":
    # 1. download MNIST dataset
    train_data, _ = download_mnist_datasets() 
    print("downloaded mnist dataset")

    # 2. create a data loader for the trainset
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

    # 3. build a model
    # must assign netowrk to a device to train
    # 1. cuda - gpu
    # 2. cpu
    # how to pick device?
    # check which cuda/cpu acceleration is available

    # if torch.cuda.is_available():
    #     device = "cuda"
    # else:
    #     device = "cpu" #me rn
    # print(f"using {device} device")
    # print("what the hell?")

    device = torch.device('mps')
    print(f"using {device} device")
    print("what the hell?")

    feed_forward_net = FeedForwardNet().to(device)

    # instantiate loss fn and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(feed_forward_net.parameters(), 
                                    lr = LEARNING_RATE) # mix of momentum and rms prop
                                    # momentum converges faster and reduces oscillations
                                    # rms prop allows denom to be smaller, since we are colecting the squares instead
                                    # https://medium.com/nerd-for-tech/optimizers-in-machine-learning-f1a9c549f8b4


    # train model
    train(feed_forward_net, train_data_loader, loss_fn, optimizer, device, EPOCHS)


    # get current directory
    model_path = os.path.dirname(__file__)
    model_path  = model_path + "/feedforwardnet.pth"
    # "feedforwardnet.pth"
    print(f"model_path: {model_path}")

    # store the model
    # .state_dict() gets all of the params (layers, ..) in a dictionary
    # feedforwardnet.pth: path we want to store the model at
    torch.save(feed_forward_net.state_dict(), model_path)
    print("model trained and stored at feedforwardnet.pth")
