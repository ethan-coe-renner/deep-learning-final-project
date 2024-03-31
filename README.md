# Deep Learning Project
![Starry Night by Van Gogh](imgs/van-gogh.avif)

## Dependencies
- `pip install deeplake`
- `pip install -r requirements.txt`

## Downloading Dataset
To download the dataset, run `make run` in the root of this repository. On the first run this will create a repository called `dataset` in the root which will contain the wikiart dataset from `hub://activeloop/wiki-art`.
Successive runs will just use that downloaded dataset.

# Implementation
## Project Description

The project consists of using Convolutional Neural Networks to classify artworks
by art styles from the WikiArt Dataset. The dataset consists of:

- 195 different artists
- 42129 images for training
- 10628 images for testing
- 27 labels which correspond to image genre

Due to the time constraints and lack of hardware, we decided to train our network against the following labels:

- Realism
- Expressionism
- Romanticism
- Ukio-e
- Minimalism
- Baroque

## Preprocessing Data

### DeepLake

DeepLake is a “data lake for deep learning”. It provides the WikiArt database which we use to train our model through its Python API. It even has support for PyTorch and easily allows us to convert a DeepLake dataset into a PyTorch loader. 

Here, we download the dataset. It will check to see if there is a local version downloaded and use that, otherwise it will download the dataset.

```python
ds = deeplake.dataset('hub://activeloop/wiki-art', access_method = 'local')
```

Next, we filter out all but the styles we want to train on

```python
filtered_labels = ["action_painting", "analytical_cubism", "synthetic_cubism"] #? Bottom three, for testing
@deeplake.compute
def filter_labels(sample_in, labels_list):
    return sample_in.labels.data()['text'][0] in labels_list

ds_view = ds.filter(filter_labels(filtered_labels), scheduler = 'threaded', num_workers = 0)
```

Now, we need to split the dataset into a training and a validation subset.

```python
first_eight_percent = int(len(ds_view) * 0.8)
train = ds_view[:first_eight_percent]
validate = ds_view[first_eight_percent:]
```

Also, we apply a transformation to convert the images to tensors, normalize them, and crop them.

```python
transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5322, 0.4754, 0.4146),
                         (0.2907, 0.2803, 0.2766)),

    transforms.CenterCrop(227) # different experiments used different cropping values
])
```

Finally, we use DeepLake’s `.pytorch()` function to convert the dataset into two PyTorch data loaders.

```python
wikiart_train = train.pytorch( 
    transform={'images': transformation, 'labels': None},
    tensors=["images", "labels"],
    shuffle=True,
    batch_size=32
)
wikiart_validate = validate.pytorch( 
    transform={'images': transformation, 'labels': None},
    tensors=["images", "labels"],
    shuffle=True,
    batch_size=32
)
```

### Directly from Dataset

The second way we preprocessed the Data was directly from the dataset downloaded from the official WikiArt page. This method was used in `model2.ipynb` and `model3.ipynb` .

We first defined the following transformations we wanted to apply on the data:

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop(227),
    transforms.Normalize(mean = (0.4923, 0.4398, 0.3775), std=(0.2874, 0.2797, 0.2695)) # 6 classes
])
```

- We first resized the images to 256x256 pixels, then applied a center crop of 227 pixels in order to minimize the noise and quality loss from resizing images alone.
- The mean and standard deviation values used in `Normalize()` were found using the following code:
    
    ```python
    def get_mean_and_std(loader):
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0
        for data, _ in loader:
            channels_sum += torch.mean(data, dim=[0,2,3]) # B x W x H
            channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
            num_batches += 1
        
        mean = channels_sum/num_batches
        std = (channels_squared_sum/num_batches - mean**2)**0.5
        return mean, std
    ```
    

Then  the dataset was parsed and transformed:

```python
root = "./model/wikiart" #path to Dataset directory
dataset = datasets.ImageFolder(root=root, transform=transform)
```

- The way that the data in the WikiArt directory was structured made it convenient to use `datasetes.ImageFolder()` to parse the data

The data was randomly split into a training and validation subsets in a 80%/20%, ratio reespectivelly:

```python
train_len = int(len(dataset)*0.8)
val_len = len(dataset)- train_len
train, val = data.random_split(dataset, [train_len, val_len])
```

And finally, Dataloaders with a batch size of 32 were created for each set:

```python
train_dataloader = data.DataLoader(train, batch_size=32, shuffle=True, drop_last=True)
val_dataloader = data.DataLoader(val, batch_size=32, shuffle=False, drop_last=True)
```

## Training and Validation

Below are the functions used for training and validating the models used:

```python
def training_function(train_loader, model, loss_fn, optimizer, device, n_epochs):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            labels = labels.flatten()
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs).to(device=device)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch,
                loss_train / len(train_loader)))

def validate(model, train_loader, val_loader, device):
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                # labels = labels.flatten()
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                outputs = model(imgs).to(device=device)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
        print("Accuracy {}: {:.6f}".format(name, correct / total))
```

## Simple Model

Our simple model consisted of three convolutional layers. These layers apply a max pooling and hyperbolic tangent function onto the output of each layer.

```python
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
```

### Results

The training of this model on the 6 styles mentioned for 100 epochs with a learning rate of $10^{-2}$ took 7.1 hours. This resulted in a training accuracy of 15.177567% and validation accuracy of 0.006535%. Clearly, this model is much too simple for a classification problem of this size.

This first model has a number of problems. First, it is very shallow, with only 3 convolutional layers. Additionally, this test cropped images to 32 pixels. This was done to speed up the training, but likely removed too much information from the images to do classification. We addressed many of these problems in our second, deeper model.

## Deeper Model

### Preprocessing Improvements

To improve training accuracy, we resized the images to 256x256 pixels, and applied a center crop of 227x227 pixels. 

### Model

Our model was based off of the [CaffeNet architecture](https://www.researchgate.net/figure/CaffeNet-architecture_fig1_325174710). It consists of five convolutional layers followed by three fully-connected layers, where:

- Max-pooling is applied after 1st, 2nd, and 3rd convolutional layers
- BatchNorm is applied before activation functions
- Dropout is applied after the first two fully connected layers
- ReLu activation functions are used in all layers
    - ReLu was chosen over Tanh for faster foward computations
- Log softmax is applied in output layer to compute probabilities
    - We used Negative Log Likelihood Loss function for training

```python
class Net2(nn.Module):
    def __init__(self, n_classes) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        
        # 3 fully connected layers
        # Apply dropout after first 2 fc
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, n_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1) # calculate probabilities
        return out
```

# Results

In order to test and optimize our models, we ran the deeper network in four different test cases:

## Case 1: Three styles

We trained this model on three styles (Expressionism, Romanticism, and Realism) using the following hyperparameters:

- Learning rate of 1e-4
- Stochastic Gradient Descent optimizer
- Added weight decay of 0.005 to help model generalize and avoid overfitting
- 80 epochs
- Negative Log Likelihood Loss function

The training it took around 7.5 hours, and resulted in significant improvements:

- Training accuracy of **70.4%**
- Validation accuracy of **64.3%**.
- Loss value of **0.67**

(Results of this training can be found in `model2.ipynb`)

## Case 2: Six styles

We wanted to test if the deeper model would hold its performance with a greater range of painting styles. This test case used the same hyperparameters as in Case 1, but we trained the model with all 6 styles. 

The training took around 13 hours, and it resulted in:

- Training accuracy of **64.0%**
- Validation accuracy of **57.8%**
- Loss value of **0.89**

As expected, there was a slight decrease in accuracies from adding more styles to classify. However, there is room for improvement since the loss during the training was still decreasing even at epoch 80. So we decided to test training the model with a higher learning rate to help it learn faster.

(Results of this training can be found in `model3.ipynb`)

## Case 3: Higher learning rate

For test case 3, we increased the learning rate from 1e-4 to 1e-3 and trained the same model on the same 6 painting styles.  

The training took around 10 hours, and it resulted in:

- Training accuracy: **97.6%**
- Validation accuracy: **59.1%**
- Loss value: **0.08**

There were significant changes in loss, having a decrease from 0.89 to 0.08 ****at epoch 80. As a result, we got very high training accuracy. However, there was only 1.3% increase in validation accuracy. We suspect that the increase in learning rate have made the model overfit, so for our final test case we decide to go back to a lower learning rate of 1e-4, but increase the number of epochs in the training.

(Results of this training can be found in `model3.ipynb`)

## Case 4: Low learning rate, High epoch number

In test case 4, we decreased the learning rate back to 1e-4 and increased the number of epochs to 100.

The training took around 13 hours, and it resulted in:

- Training accuracy: **67.7%**
- Validation accuracy: **58.8%**
- Loss value: **0.79**

The increase in 20 epochs from test case 2 have increased the training accuracy to 67.7% and validation accuracy by 58.8%. As predicted, having a low learning rate while increasing the number of epochs improved the accuracy numbers without overfitting the model as much as in Case 3. 

(Results of this training can be found in `model3.ipynb`)

## Analysis of results

The model with the highest training and validation accuracies for classifying the 6 painting styles was test case 3. However, if given enough time and computing power, test case 4 could potentially reach higher validation accuracies without overfitting the model as much.

We can expect accuracy values in the 50-60% range in part because in some cases classifying painting styles is more than just considering the visual elements of the artwork. Although many of the art styles in the dataset are very similar, the style into which a piece is classified has more to do with its context (time period, location, etc) rather than its visual features. For example, some paintings from Baroque (1600 AD), Romanticism (1770 AD), and Realism (1880 AD) may have similar colors and brushwork, but they are separated by the period in which the paintings were created, or the underlying message the painting is conveying. Thus, with the information contained within the dataset, it is not surprising that the model misclassified some images.

In addition, in [Fine-tuning Convolutional Neural Networks for fine art classification](https://www.sciencedirect.com/science/article/abs/pii/S0957417418304421), their CaffeNet-based implementation trained on all 27 art styles and got accuracy values around 54.2%, so the results from our model are relatively positive.