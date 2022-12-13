import deeplake
from torchvision import transforms

ds = deeplake.dataset('hub://activeloop/wiki-art', access_method='local')

transformation = transforms.Compose([
    transforms.ToTensor(),
    # TODO: figure out these values
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),

    # TODO: figure out a reasonable value here
    transforms.CenterCrop(32)
])

wikiart = ds.pytorch(
    transform={'images': transformation, 'labels': None},
    tensors=["images", "labels"],
    # decode_method={'images':'pil', 'labels': 'numpy'},
    shuffle=True,
    batch_size=32)

# We expect that this should return a loader, next returns a list with two elements
# first is a tensor of 32 images
# second is a tensor of 32 integers representing the labels of that batch

# for images, labels in wikiart:
#     print("Images: ", images)
#     print("labels: ", type(labels))
#     break


# given this, this should not work vvv. We need to do this before the
# TODO: map these onto the actual values we are checking, this needs to be changed because labels is a tensor, not a number
#label_map = {4:0, 5:1, 7:2}
#class_names = ["romantic", "name2", "name3"]
# dataset = [(img, label_map[label])
#       for img, label in wikiart if label in [4,5,7] # TODO change this last list too
#       ]

# Make the network
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.conv1 = nn.Conv2d(3,n,kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n,n//2,kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(n//2,n//2,kernel_size=3, padding=1)

        self.ch = 4 * 4 * (n//2)
        self.fc1 = nn.Linear(self.ch, 32)
        self.fc2 = nn.Linear(32, 27)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = F.max_pool2d(torch.tanh(self.conv3(out)), 2)
        out = out.view(-1, self.ch)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        out = F.log_softmax(out,dim=1)
        return out


def training_function(train_loader, model, loss_fn, optimizer, device, n_epochs):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            labels = labels.flatten() # this is needed because for some reason every label is in a seperate tensor
            print(labels)
            print(imgs.shape)
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)))


cnn32 = Net(32)

import torch

import torch.optim as optim


device = (torch.device('cuda') if torch.cuda.is_available()
                      else torch.device('cpu'))

training_function(
    train_loader = wikiart,
    model = cnn32,
    loss_fn = nn.NLLLoss(),
    optimizer = optim.SGD(cnn32.parameters(), lr=1e-2),
    device=device,
    n_epochs = 2,
)
