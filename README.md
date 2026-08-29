# Neural Network Mathematics
I am going to build a neural network with just maths and numpy. won't be using pytorch or tensorflow or keras. This is just me trying to understand the maths behind neural networks.

And I am going to be using the MNIST dataset which is just a bunch of handwritten numbers and we'll be training our network to read these numbers.

Might implement the same thing with pytorch later. Might even try to do it with assembly since i am also learning assembly.

And the network is intentionally small since we just want to use it to understand what the model is doing mathematicallly.

## Setup
This is just installing python and numpy, its the boring part. won't waste time writing about it. or will use an ai to write it later.

Will explain the code too after the maths so you can just paste in google collab or kaggle notebook and run step by step.

## The Maths!

The input images are 28px by 28px. Thats a total of 784px. And we need to be able to read each input pixels to map out which one is shaded and which one isn't also to know which number is which. Although we aren't using shadings in this training, we are simply going to be training with the existing labels and which one of those pixels holds values and what values they hold. I think the pixels values should be between 0-255 with 0 as white and 255 as black.

So we are going to have 3 layers: the input layer, the hidden layer and the output layer. 

The input will have 784 neurons to capture all the input pixels, the hidden layer will have just 10 neurons and this is the layer that does a lot of our processing and the output layer is also 10 neurons which represents the output 0-9. SO THAT'S ESSENTIALLY THE COMPLETE MODEL, TAKES IN 784PX AND OUTPUT 10 NUMBERS(NEURONS BUT YH TH).

```
Input Layer              Hidden Layer              Output Layer

784 neurons               10 neurons               10 neurons
(28 × 28 pixels)                                   (digits 0–9)

 i₁ ────────────────┐
 i₂ ────────────────┤
 i₃ ────────────────┤
 ⋮                  ├──────→ h₁ ──────────────┐
 i₇₈₄ ──────────────┤        h₂ ──────────────┤
                    │        h₃ ──────────────┤
                    │        ⋮                ├──→ 0
                    │        h₁₀ ─────────────┤──→ 1
                    │                         ├──→ 2
                    │                         ├──→ ⋮
                    │                         ├──→ 8
                    └─────────────────────────┴──→ 9

```

### To avoid yapping, i am going to broadly categorize each step of the mathmatics and explain each part.

1. Prepare the input tensor(matrix)
2. Forward propagation
    - Initialize with a random weight and bias
    - Run the activation function ReLU for layer $Z^{[1]}$
    - Apply the second weight and biases for the output layer and run activation function softmax this time.
3. Backward propagation
    - Calculate the difference in our prediction and the actual label
    - Calculate how much of this difference is caused by our weight and bias
    - We will also do the same calculations for our hidden layer. After undoing activation
4. We update the parameters based on the calulations and learning rate(both weights and biases)
5. We run through 2-4 again as much as we need to get our accuracy up.


## 1. Preparing the input tensor.

The training data `./data/train.csv` has each row as an image and each column is feature. The number of rows can be in thousands(as much data as we have), we will use 10,000 here to explain. And the columns are just 785: column 1 for the label and the remaining 784 for the all the 784px in each image.
 
That can be represented as a matrix with shape (10000, 784). 

We are also going to divide the data into two, we'll use half to train our model and the other half to test.

We are also going to transpose the matrix so each column represents an image and the rows represents the features. This changes our matrix shape to 784 X 10000 but it makes matrices operations easier to do.

So after transposing, we have our input neurons as 784 rows in each column and processing or training with an image requires just splicing out a column.

I think that's all for the the prepareation

## 2. Forward propagation.
### 2a. Initialize with a random weight and bias

This is where we initialize the connections between our layers with random weight and biases. Both weight and bias are just parameters we'll keep on trying to train our model to find the best value at which they give the required output.

So lets call our input matrix X

Applying our weight and bias is `==>` $Z^{[1]} = W^{[1]}X + b^{[1]}$ 

### 2b. Run the activation function ReLU for layer $Z^{[1]}$
Activation simply means running a function that adds complexity to the function of a layer, it gives room for each layer to be able to things seperately and deal with much more complicated things. Without it, we'd be able to compress all our layers into a single layer and a single equation. And that makes the whole multiple layers thing useless.

The function we will be using a is a popular one called ReLU.
```
 A[1] = ReLU(z) = {
    z if z > 0
    0 if z <= 0
 }
```

It simply drops all the negative values and keep only the relevant ones for the next layer.

### 2c. Apply the second weight and biases for the output layer.

We are repeating the same thing for the output layer, but remmeber, the input for this layer is the result of the activation from the previous one A($Z^{[1]}$)

So our $Z^{[1]} = W^{[1]}X + b^{[1]}$  becomes $Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}$. It will have its own set of weights and biases. And at this point, our weight matrices should have reduced to a simple 10X10 since both the input and out layer in this side are 10 neurons.

After getting our output matrix, we use another activation function for this, its called softmax. it essentially squeezes our result between 0 and 1 and is literally a probability distribution over 10 classes. Where all values add up to 1 and values close to zero means lower probability and close to one is higher probability.

```
 A[2] = softmax(Z[2]) = {
    e(z)/(summation of all exponentials of z's)
 }
```

## 3. Backward propagation

### 3a Calculate the differnece in our prediction and the actual label

#### 1. $\underline{dZ^{[2]}}$
The difference between our label(Y) and our prediction $A^{[2]}$ is $dZ^{[2]}$ and the equation is given by:<br>
`==>` $dZ^{[2]} = A^{[2]} - Y$<br>
it is specifically is the gradient of the loss with respect to $Z^{[2]}$


And this isn't simply subtracting the label's value from our predicted probability. Remember, we ran softmax on our output so its values are between 0 and 1, they are essentially probabilities of those values beiing the label. So to substract, we need to take our label to a one hot encoded value instead of subtracting the actual value, we'll subtract the probabibility of our label appearing(1) from the probibility of our predicted value being the one. IF that makes sense.

### 3b Calculate how much of this difference is caused by our weight and bias
So after getting  $dZ^{[2]}$, we are going to calculate just how much the weight and bias on the outer layer affected it so we can know which to change and how.

#### 2. $\underline{dW^{[2]}}$
For the weight on the output layer, the gradient loss($dW^{[2]}$) is given by <br>

$$
dW^{[2]} = \frac{1}{m}dZ^{[2]}A^{[1]T}
$$

Where m is the number of data we are trainng with and notice how the output of the hidden layer is transposed $A^{[1]T}$. We simply want to know which weights were responsible for $Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}$ 

#### 3. $\underline{db^{[2]}}$
the gradient loss of the bias is given by <br>

$$
db^{[2]} = \frac{1}{m}\sum dZ^{[2]}
$$

Pretty simple since we just aggregate the output and and average them. The same bias was added to every training example for any given output neuron

### 3c. Calculations for our hidden layer. After undoing activation
#### 4. $\underline{dZ^{[1]}}$

This is where we get how much the hidden layer contributed to the error. We go backward on the activation we did with ReLU by using the derivative of ReLU. We'll do an element wise multiplication of that to the the backward path of the $Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}$

$$
dZ^{[1]} = W^{[2]T}dZ^{[2]} * ReLU'(Z^{[1]})
$$

Where the first part of the equeation is $dA^{[1]}$ showing how much $A^{[1]}$ affected our output.
Where the derivative of ReLU is 

```
 ReLU'(z) = {
    1 if z > 0
    0 if z <= 0
 }
```

And in that, we went back on the activation of the hidden layer and found the loss gradient of the hidden layer. Now, to how much of its weight and bias affected that.

#### 5. $\underline{dW^{[1]}}$

$$
dW^{[1]} = \frac{1}{m}dZ^{[1]}X^{T}
$$

#### 6. $\underline{db^{[1]}}$

$$
db^{[1]} = \frac{1}{m}\sum dZ^{[1]}
$$


And that's all for the backward and forward propagation, here's a summary of the steps:

```
FORWARD

Z¹
↓
ReLU
↓
A¹
↓
W²
↓
Z²
↓
Softmax
↓
A²
↓
Loss


BACKWARD

dZ²
↓
dW² + db²
↓
dA¹
↓
dZ¹
↓
dW¹ + db¹
```

## 4. We update the parameters
So now, we update our parameters for the next run.

$$
W^{[1]} = W^{[1]} - \alpha dW^{[1]} \\
$$

$$
b^{[1]} = b^{[1]} - \alpha db^{[1]} \\
$$

$$
W^{[2]} = W^{[2]} - \alpha dW^{[2]} \\
$$

$$
b^{[2]} = b^{[2]} - \alpha db^{[2]} \\
$$

Where $\alpha$ is the learning rate. A fixed value

## 5. We run through the whole thing again
We keep on running trhough the whole thing till we get our weights and biases to value sufficiently accurate enough to make our model usable. Can't expect much form this though. the target is about 70-80% accuracy.


# THE CODE! 

Check the main file, i'll add comments

## Will attach the repo with pytorch version and assembly version later. IF i do .

# Links
- [Pytorch Version](./pytorch.md)
- [Assembly Version(not yet)](#)