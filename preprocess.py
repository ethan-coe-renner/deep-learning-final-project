import deeplake
from torchvision import transforms

ds = deeplake.dataset('hub://activeloop/wiki-art', access_method='local')

transformation = transforms.Compose([
    transforms.ToTensor(),
    # TODO: figure out these values
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),

    # TODO: figure out a reasonable value here
    transforms.CenterCrop(500)
])

def transform(sample):
    image, label = sample["images"], sample["labels"]
    image = transformation(image)
    return image, label

wikiart = ds.pytorch(transform=transform, tensors=["images", "labels"], decode_method={'images':'pil', 'labels': 'numpy'}, shuffle=True, batch_size=32)

# TODO: map these onto the actual values we are checking, this needs to be changed because labels is a tensor, not a number
label_map = {4:0, 5:1, 7:2}
class_names = ["romantic", "name2", "name3"]
# dataset = [(img, label_map[label])
#       for img, label in wikiart if label in [4,5,7] # TODO change this last list too
#       ]

'''
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
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = F.max_pool2d(torch.tanh(self.conv3(out)), 2)
        out = out.view(-1, self.ch)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        out = F.log_softmax(out,dim=1)
        return out


cnn32 = Net(32)

import torch
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

size = 0
for _, _ in dataset:
    size += 1

print("batches in dataset:", size)

'''
