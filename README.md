# Pytorch FFN MNIST Number Prediction

## Data

Data is from [TorchVision Dataset](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html)

### Attribute 

28 x 28 pixel, 1 channel

### Data Image
![data_image](https://github.com/UncleThree0402/PyTorch_FFN_MNIST/blob/master/Photo/MNIST_numbers_Images.png)

### Data Before Normalized
![data_before](https://github.com/UncleThree0402/PyTorch_FFN_MNIST/blob/master/Photo/Data_before_normalize.png)

### Data After Normalized
![data_after](https://github.com/UncleThree0402/PyTorch_FFN_MNIST/blob/master/Photo/Data_after_normalize.png)

### Count of label
To check is dataset balanced

![count_labels](https://github.com/UncleThree0402/PyTorch_FFN_MNIST/blob/master/Photo/count_labels.png)

## Model

### Net
```python
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.input = nn.Linear(784, 64)

        self.bn1 = nn.BatchNorm1d(64)

        self.do1 = nn.Dropout(0.2)

        self.ll1 = nn.Linear(64, 32)

        self.bn2 = nn.BatchNorm1d(32)

        self.do2 = nn.Dropout(0.2)

        self.ll2 = nn.Linear(32, 32)

        self.bn3 = nn.BatchNorm1d(32)

        self.do3 = nn.Dropout(0.2)

        self.output = nn.Linear(32, 10)

    def forward(self, x):
        x = self.input(x)
        x = nn.LeakyReLU()(x)
        x = self.bn1(x)
        x = self.do1(x)
        x = self.ll1(x)
        x = nn.LeakyReLU()(x)
        x = self.bn2(x)
        x = self.do2(x)
        x = self.ll2(x)
        x = nn.LeakyReLU()(x)
        x = self.bn3(x)
        x = self.do3(x)
        x = self.output(x)
        return x
```

### Loss Function
```python
loss_fn = nn.CrossEntropyLoss()
```

### Optimizer
```python
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
```
>lr = 0.001, weight_decay = 0.02

### lr_scheduler
```python
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
```
>step_size = 5, gamma = 0.5

## Train

### Loss
![losses](https://github.com/UncleThree0402/PyTorch_FFN_MNIST/blob/master/Photo/losses.png)
> Valid lower mostly because using dropout layer

### Accuracy
![accuracies](https://github.com/UncleThree0402/PyTorch_FFN_MNIST/blob/master/Photo/accuracies.png)
> Valid higher mostly because using dropout layer

### Learning Rate
![lr_rate](https://github.com/UncleThree0402/PyTorch_FFN_MNIST/blob/master/Photo/lr_rate.png)

## Performance
![performance](https://github.com/UncleThree0402/PyTorch_FFN_MNIST/blob/master/Photo/performance.png)
> We can see that model performance is unbiased.

* Train Accuracy : 98.8%
* Validation Accuracy : 97.4%
* Test Accuracy : 97.3%

### Train
```bash
               precision    recall  f1-score   support

           0       0.99      0.99      0.99      5539
           1       0.99      0.99      0.99      6361
           2       0.99      0.99      0.99      5561
           3       0.99      0.98      0.99      5676
           4       0.99      0.99      0.99      5494
           5       0.99      0.99      0.99      5026
           6       0.99      0.99      0.99      5471
           7       0.99      0.99      0.99      5855
           8       0.99      0.98      0.99      5458
           9       0.98      0.99      0.98      5559

    accuracy                           0.99     56000
   macro avg       0.99      0.99      0.99     56000
weighted avg       0.99      0.99      0.99     56000
```

### Valid
```bash
               precision    recall  f1-score   support

           0       0.98      0.99      0.98       685
           1       0.98      0.99      0.99       780
           2       0.97      0.97      0.97       735
           3       0.97      0.97      0.97       737
           4       0.97      0.97      0.97       666
           5       0.98      0.97      0.98       611
           6       0.98      0.98      0.98       711
           7       0.97      0.98      0.98       722
           8       0.98      0.95      0.96       664
           9       0.96      0.97      0.96       689

    accuracy                           0.97      7000
   macro avg       0.97      0.97      0.97      7000
weighted avg       0.97      0.97      0.97      7000
```

### Test
```bash
               precision    recall  f1-score   support

           0       0.98      0.98      0.98       679
           1       0.97      0.99      0.98       736
           2       0.98      0.97      0.97       694
           3       0.97      0.96      0.97       728
           4       0.97      0.98      0.97       664
           5       0.97      0.96      0.97       676
           6       0.98      0.98      0.98       694
           7       0.97      0.98      0.98       716
           8       0.97      0.96      0.96       703
           9       0.96      0.96      0.96       710

    accuracy                           0.97      7000
   macro avg       0.97      0.97      0.97      7000
weighted avg       0.97      0.97      0.97      7000
```

### Confusion matrix

#### Train
![train_conf](https://github.com/UncleThree0402/PyTorch_FFN_MNIST/blob/master/Photo/tcfm.png)

#### Valid
![valid_cm](https://github.com/UncleThree0402/PyTorch_FFN_MNIST/blob/master/Photo/vcfm.png)

#### Test
![test_cm](https://github.com/UncleThree0402/PyTorch_FFN_MNIST/blob/master/Photo/ttcfm.png)
