<div class="cell markdown" id="69E4SzHiFC28">

[<img src='https://github.com/jeshraghian/snntorch/blob/master/docs/_static/img/snntorch_alpha_w.png?raw=true' width="300">](https://github.com/jeshraghian/snntorch/)
[<img src='https://github.com/neuromorphs/tonic/blob/develop/docs/_static/tonic-logo-white.png?raw=true' width="200">](https://github.com/neuromorphs/tonic/)

# Training Spiking Recurrent Neural Networks on Spiking Speech Commands

##### By: Richard Dao (<rqdao@ucsc.edu>), Annabel Truong (<anptruon@ucsc.edu>), Mira Prabhakar (<miprabha@ucsc.edu>)

<a href="https://colab.research.google.com/drive/1Ojck3Ui5KgekYF-jZujvSqa6e5iBmuVA?usp=sharing">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

</div>

<div class="cell markdown" id="B6JEqXrEF3FC">

For a comprehensive overview on how SNNs work, and what is going on
under the hood, [then you might be interested in the snnTorch tutorial
series available
here.](https://snntorch.readthedocs.io/en/latest/tutorials/index.html)
The snnTorch tutorial series is based on the following paper. If you
find these resources or code useful in your work, please consider citing
the following source:

> <cite> [Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang,
> Gregor Lenz, Girish Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and
> Wei D. Lu. "Training Spiking Neural Networks Using Lessons From Deep
> Learning". Proceedings of the IEEE, 111(9) September
> 2023.](https://ieeexplore.ieee.org/abstract/document/10242251) </cite>

</div>

<div class="cell markdown" id="pVNvTXthGiIT">

First, install the snntorch library if it is not already installed on
your machine.

</div>

<div class="cell markdown" id="USt2Zksq3mYr">

# 1. The Spiking Speech Commands (SSC) Dataset

</div>

<div class="cell markdown" id="Ojubvpc9eFvB">

**An explanation on the data:**

-   The Spiking Speech Commands (SSC) Dataset was generated using
    Lauscher, an artificial cochlea model. SSC is a spiking version of
    Google’s Speech Commands dataset and contains over 100,00 samples of
    waveform audio data.
-   The dataset contains 35 classes: Yes, No, Up, Down, Left, Right, On,
    Off, Stop, Go, Backward, Forward, Follow, Learn, Bed, Bird, Cat,
    Dog, Happy, House, Marvin, Sheila, Tree, Wow, Zero, One, Two, Three,
    Four, Five, Six, Seven, Eight, Nine

Tonic will be used to convert the dataset into a format compatible with
PyTorch/snnTorch. The documentation for Tonic can be found
[here](https://tonic.readthedocs.io/en/latest/generated/tonic.datasets.SSC.html).

For the purposes of this tutorial, we use Google Colab. If you use your
local notebook, you should edit the following cell to save the dataset
locally.

To access the data you may need to authorize the following in your
Google Drive. After the following, change the root file to your own
path.

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="bd46zFQPMt-F" outputId="e1e965c8-c7de-4299-b21a-ac947a792dfe">

``` python
from google.colab import drive
drive.mount('/content/drive')

root = '/content/drive/My Drive/'
ext = 'snnTorch Research Project' # Use for developing

path = os.path.join(root, ext)
```

</div>

<div class="cell markdown" id="SQPVQEmPMYOi">

# 2. Data Preprocessing using Tonic

</div>

<div class="cell markdown" id="wONo3PBqLNqD">

After the data is successfully mounted to your drive, we will begin the
data visualization using tonic.

The event tuples are formatted of type `(t, x, p)`:

-   `t` is time stamp
-   `x` is the audio channel
-   `p` is a boolean value that is always 0

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:333}"
id="8A3myIFz2zAs" outputId="b6ff21e5-1349-4e46-cced-096b513acc0b">

``` python
dataset = tonic.datasets.SSC(save_to=path, split='train')

events, target = dataset[2]
tonic.utils.plot_event_grid(events)
```

<div class="output display_data">

![](https://github.com/richard-dao/SRNNAC/blob/main/images/ssc_channels_vs_time.png)

</div>

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="Bqlu0r6AfGYK" outputId="f42f4841-c532-477d-9bf1-be115fa5cf12">

``` python
print(events.dtype)
```

<div class="output stream stdout">
    
    [('t', '<i8'), ('x', '<i8'), ('p', '<i8')]

</div>

</div>

<div class="cell markdown" id="pJtTdHZAwRHf">

## 2.1 Downsampling

We decided to downsample our data for efficiency purposes for this
tutorial. While this means we could potentially lose important
datapoints, it allows us to save significant computation time.

For the purposes of this tutorial, we downsampled to 375 channels, a
downsampling factor of 1/2.

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:635}"
id="vaiQjcPH3TNZ" outputId="069acb11-f8ef-4e80-84d6-35c5facf106c">

Courtesy of Rockpool.AI for their images on the SHD dataset (sister dataset of SSC)
<p style="font-size: 115%;">Downsampling allows us to turn data like this:</p>
<br>
<img src=https://rockpool.ai/_images/tutorials_rockpool-shd_4_0.png style="width:450px;">
<br>
<p style="font-size: 115%;">Into this:</p>
<br>
<img src=https://rockpool.ai/_images/tutorials_rockpool-shd_14_0.png style="width: 450px;">

</div>

<div class="cell code" id="4qkGPJ2Jf2lI">

``` python
sensor_size = datasets.SSC.sensor_size # By default is (700, 1, 1)
time_step = 12000 # The max time steps
downsample_factor = 1/2 # Change as needed

toTensorTransform = transforms.Compose([
    transforms.Downsample(spatial_factor=downsample_factor),
    transforms.ToFrame(sensor_size=(700 // int(1 / downsample_factor), 1, 1), time_window=time_step)
])


train_dataset = tonic.datasets.SSC(save_to=path, split='train', transform=toTensorTransform)
validation_dataset = tonic.datasets.SSC(save_to=path, split='valid', transform=toTensorTransform)
test_dataset = tonic.datasets.SSC(save_to=path, split='test', transform=toTensorTransform)
```

</div>

# 3. Network Architecture

We will use snnTorch and PyTorch to construct a Spiking Recurrent Neural Network (SRNN).

We use an initial LSTM layer followed by a Dropout layer into 2 hidden fully connected layers with recurrent leaky-integrate-and-fire (LIF) neurons. Finally, our classifier layers include an initial Dropout layer into a normal (LIF) output neuron

</div>

<div class="cell code" id="P2Ynod_Lltyh">

``` python
# Defining the Network Architecture
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

inputs = int(700 * downsample_factor)
hidden = 512
outputs = 35

beta = 0.95
lr = 0.0001

surrogate = snn.surrogate.atan(alpha=2)

model = nn.Sequential(
    # Define network architecture here
    nn.LSTM(input_size=inputs, hidden_size=512, batch_first=True),
    nn.Dropout(p=0.2),

    nn.Linear(in_features=512, out_features=512),
    snn.RLeaky(beta=beta, init_hidden=True),

    nn.Linear(in_features=512, out_features=512),
    snn.RLeaky(beta=beta, init_hidden=True),

    nn.Dropout(p=0.2),
    nn.Linear(in_features=512, out_features=35),
    snn.Leaky(beta=beta, init_hidden=True, output=True)
).to(device)

```

</div>

<div class="cell markdown" id="pjBaqQ8Azb8g">

## 3.1 Loss Function and Optimizer

For this tutorial, we found that Adam and Mean Squared Error Spike Count
Loss performed best.

*Mean Squared Error Spike Count Loss* obtains spikes from the correct
class a % of the time and spikes from the incorrect classes a % of the
time to encourage incorrect neurons to fire and avoid them from dying.

</div>

<div class="cell code" id="wccP37WizZWy">

``` python
optimizer = optim.Adam(model.parameters(), lr = lr, betas=(0.9, 0.999))
criterion = functional.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
```

</div>

<div class="cell markdown" id="Cev546CuPVax">

## 3.2. Define the Forward Pass

</div>

<div class="cell markdown" id="OeOVxoUe0dXf">

The standard forward pass we use for spiking neural networks, we keep
track of the total number of spikes and reset the hidden states for all
Leaky neurons in our network.

Note that data.size(0) is the number of time steps at each iteration.

</div>

<div class="cell code" id="4KhcpS5EfbAC">

``` python
def forward(net, data):
    total_spikes = [] # collect total number of spikes
    utils.reset(net) # reset hidden states of all Leaky neurons

    for i in range(data.size(0)): # loop over number of timesteps
        output_spikes, mem_out = net(data[i])
        total_spikes.append(output_spikes)

    return torch.stack(total_spikes)
```

</div>

<div class="cell markdown" id="Bo3MYMef1xZN">

For organization, we will store the loss and accuracy histories in
dictionaries.

</div>

<div class="cell code" id="_XMvH7AKxo6U">

``` python
loss_history = {
    'train':[],
    'validation':[],
    'test':[]
}
accuracy_history = {
    'train':[],
    'validation':[],
    'test':[]
}
```

</div>

<div class="cell markdown" id="nVihYMrmPb4W">

# 4. Training

Training neuromorphic data takes a large amount of computation time as
it requires seqeuentially iterating through time steps. In the case of
the SSC dataset, there are roughly 600 timesteps that will be run per
epoch.

In our own experiments, it took about 10 epochs with 600 iterations each
to crack \~55% validation accuracy.

> Warning: the following simulation will take a while. In our own
> experiments, it took about 2 hours to train 10 epochs of 600
> iterations.

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:1000,&quot;referenced_widgets&quot;:[&quot;305165b04837464096ea97f9e85fac23&quot;,&quot;8973d98108654f0d9f4f9e791c837ce1&quot;,&quot;6c619c9796fa464e9c32b1d317fc5bda&quot;,&quot;a720dcb61a8d48958e4c87e4f1e955fb&quot;,&quot;40513bdfe64f4b24b21e38bbdc4d7101&quot;,&quot;4aa613949760459fb425ef8c68b767ad&quot;,&quot;ff397268d3574c3491f85342804f09c6&quot;,&quot;373c834c87eb40d2b67186ffbf9f0271&quot;,&quot;2dd3b1d4ab814a73ab816f7983e67b39&quot;,&quot;4c29caf81aac49f882dfe8ddc83e198a&quot;,&quot;f00f9adc0ad44658be10079fc7007476&quot;]}"
collapsed="true" id="3-xVJXa6nMB7"
outputId="9dab93f7-2623-47ea-be09-a589c939c3d1">

``` python
# Training Loop
num_epochs = 10

from tqdm.autonotebook import tqdm

with tqdm(range(num_epochs), unit='Epoch', desc='Training') as pbar:
    epoch = 0
    for _ in pbar:
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            for i, (events, labels) in enumerate(dataloaders[phase]):
                events = events.squeeze()
                events, labels = events.to(device), labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    spk_rec = forward(model, events)
                    loss = criterion(spk_rec, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    if i % 25 == 0:
                        loss_history[phase].append(loss.item())
                        accuracy = functional.accuracy_rate(spk_rec, labels)
                        accuracy_history[phase].append(accuracy)
                        print(f"Epoch {epoch+1}, Iteration {i} \n{phase} loss: {loss.item():.2f}")
                        print(f"Accuracy: {accuracy * 100:.2f}%\n")

        epoch += 1
```

<div class="output stream stdout">
    
    Epoch 0, Iteration 0 
    train loss: 4.48
    Accuracy: 3.91%

    Epoch 0, Iteration 25 
    train loss: 1.87
    Accuracy: 3.91%

    Epoch 0, Iteration 50 
    train loss: 1.08
    Accuracy: 4.69%

    Epoch 0, Iteration 75 
    train loss: 0.99
    Accuracy: 3.91%

    Epoch 0, Iteration 100 
    train loss: 0.99
    Accuracy: 3.12%

    ...

    Epoch 9, Iteration 500 
    train loss: 0.66
    Accuracy: 52.34%

    Epoch 9, Iteration 525 
    train loss: 0.68
    Accuracy: 46.88%

    Epoch 9, Iteration 550 
    train loss: 0.69
    Accuracy: 45.31%

    Epoch 9, Iteration 575 
    train loss: 0.69
    Accuracy: 42.19%

    Epoch 9, Iteration 0 
    validation loss: 0.70
    Accuracy: 33.59%

    Epoch 9, Iteration 25 
    validation loss: 0.65
    Accuracy: 51.56%

</div>

</div>

<div class="cell markdown" id="SiwI8M2GPmN7">

# 5. Results

We plot and compare the accuracy and loss of the different splits of the
dataset.

</div>

<div class="cell markdown" id="4DiBDQPw5G_l">

## 5.1 Plot Train and Validation Set Accuracy

Note that we recorded accuracies and losses every **25** iterations. You
can see that our graphs, while jumpy, show an upward trend in accuracy
with little overfitting.

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:472}"
id="PVwkks77wfFo" outputId="c16127ed-2bb9-4405-94c4-f238b8737d40">

``` python
train_fig = plt.figure(facecolor="w")
plt.plot(accuracy_history['train'])
plt.title("Train Set Accuracy")
plt.xlabel("Iteration x 25")
plt.ylabel("Accuracy")
plt.show()
```

<div class="output display_data">

![](https://github.com/richard-dao/SRNNAC/blob/main/images/ssc_train_acc.png)

</div>

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:472}"
id="uLRsIb-hxQxe" outputId="5ab7137f-6dce-42aa-8510-66bbad440ce1">

``` python
val_fig = plt.figure(facecolor="w")
plt.plot(accuracy_history['validation'])
plt.title("Validation Set Accuracy")
plt.xlabel("Iteration x 25")
plt.ylabel("Accuracy")
plt.show()
```

<div class="output display_data">

![](https://github.com/richard-dao/SRNNAC/blob/main/images/ssc_val_acc.png)

</div>

</div>

<div class="cell markdown" id="i11BfsNH5i8i">

## 5.2 Testing Our Dataset

Tonic provides us a test dataset for SSC from which we can use to test
the average accuracy of our model.

</div>

<div class="cell code" id="RbXG1ALnwv0m">

``` python
model.eval()

with torch.no_grad():
    for i, (events, labels) in enumerate(dataloaders['test']):
        events = events.squeeze()
        events, labels = events.to(device), labels.to(device)

        spk_rec = forward(model, events)
        loss = criterion(spk_rec, labels)
        if i % 50 == 0:
            loss_history['test'].append(loss.item())
            accuracy = functional.accuracy_rate(spk_rec, labels)
            accuracy_history['test'].append(accuracy)
```

</div>

<div class="cell markdown" id="L823NmSu5z-U">

On average, we get around the 40% accuracy range for our test set.

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="_-cBnj0Tyl_m" outputId="69467aea-43c8-4e35-bf9c-d50ff97223cf">

``` python
print("Average Accuracy of Test Dataset: ", str(np.mean(accuracy_history['test']) * 100) + "%")
```

<div class="output stream stdout">
    
    Average Accuracy of Test Dataset:  43.8380238791423%

</div>

</div>

<div class="cell markdown" id="RYSluV7_5-40">

Finally, some comparison graphs over the entire training time.

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:927}"
id="2qv5EUjCy3z4" outputId="81b3a97b-be67-4ac0-90df-610f799614db">

``` python
loss_comparison_fig = plt.figure(facecolor="w")
plt.plot(loss_history['train'], label='train')
plt.plot(loss_history['validation'], label='validation')
plt.plot(loss_history['test'], label='test')
plt.legend(loc='best')
plt.title(f"Train vs Validation vs Test - Loss Curves , LR = {lr} Batch Size = {batch_size} Epochs = {num_epochs}")
plt.xlabel("Iteration x 25")
plt.ylabel("Loss")
plt.show()

accuracy_comparison_fig = plt.figure(facecolor="w")
plt.plot(accuracy_history['train'], label='train')
plt.plot(accuracy_history['validation'], label='validation')
plt.plot(accuracy_history['test'], label='test')
plt.legend(loc='best')
plt.title(f"Train vs Validation vs Test - Accuracy Curves , LR = {lr} Batch Size = {batch_size} Epochs = {num_epochs}")
plt.xlabel("Iteration x 25")
plt.ylabel("Accuracy")
plt.show()
```

<div class="output display_data">

![](https://github.com/richard-dao/SRNNAC/blob/main/images/ssc_train_val_test_loss.png)

</div>

<div class="output display_data">

![](https://github.com/richard-dao/SRNNAC/blob/main/images/ssc_train_val_test_acc.png)

</div>

</div>

<div class="cell markdown" id="5ZFLNk2mPuxa">

# Congrats! We're all done!

You trained a Spiking Neural Network using `snnTorch` and `Tonic` on
SSC!

</div>
