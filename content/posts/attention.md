---
date: '2026-02-08T19:11:45-05:00'
draft: false
title: 'Atten-hut!'
---

In this blog, I want to build attention from scratch. But I don't want to jump straight into the best implementation. We're going to start with dumbest, most naive possible code, see why it fails, and then add one idea at a time until attention falls on your head like the apple fell on Newton's. 

I learned it this way and it really stuck to me. This is heavily inspired from Karpathy's video, and this concept has probably been beaten to hell. But the point of this blog is to just practice formalizing my notes.


## The task: next character prediction

The setup is: given a set of characters, how can we predict the next best character? Models like ChatGPT don't actually work at the character level, they work at something called the **token** level. This is a blog for later, but tokens are just parts of a word. So a word like "homework" will be broken into 2 tokens "home-work". For now, our tokens are characters.
The set of unique tokens is called our **vocabulary**, and it is of size $V$.

So, we take long string (more formally called a **sequence**) of text, convert it to integers, and try to train a model that  predicts the next character given the previous ones.

If the sequence of tokens is
$$
(x_1, x_2, \dots, x_T)
$$
then at position $t$, the model sees $x_1, x_2, \dots, x_t$ and is trained to predict $x_{t+1}$. This can be written as:
$$
p(x_{t+1} \mid x_1, \dots, x_t)
$$

What's important to note here is how many tokens can the model see into the past? How big should $t$ be? That's called **context**. The code below clearly visualizes that idea.
```python
block_size = 8

x = train_data[:block_size]
y = train_data[1:block_size + 1]

for t in range(block_size):
  context = x[:t + 1]
  target = y[t]
  print(f'When context is {context}, target is {target}')
```
```
When context is [18], target is 47
When context is [18, 47], target is 56
When context is [18, 47, 56], target is 57
When context is [18, 47, 56, 57], target is 58
When context is [18, 47, 56, 57, 58], target is 1
When context is [18, 47, 56, 57, 58,  1], target is 15
When context is [18, 47, 56, 57, 58,  1, 15], target is 47
When context is [18, 47, 56, 57, 58,  1, 15, 47], target is 58
```

`block_size` is how big our context is. We look 8 tokens into the past to predict the next one. `x` (our input) is the first 8 samples in our train set. `y` (our output or target) is also 8 samples long, but it's offset by 1. Why the offset? It's easier to answer that with an example. 

At timestep 0, we want to use the all tokens up to token 0 (so `x = [token_0]`) to predict `y = token_1`. 

At timestep 1, we we want to use all tokens up to token 1 (so `x = [token_0, token_1]`) to predict `y = token_2`. 

At timestep 3, we want to use all tokens up to token 2 (so `x = [token_0, token_1, token_2]`) to predict the `y = token_3`.

Extend this to some timestep `t`, we get "we want to use all tokens up to token `t` to predict the `t+1` token. And if we make `y` by an offset by one, token `t+1` is stored at index `t`. Now the code should make sense!


## The simplest possible model
Now that we understand what we are training our model to do, let's start with the easiest implementation. This would be to just ignore the context and only look at the last token when predicting the current one. This is called a bigram language model.

The way we implement this is a **lookup table**. We should be able to look up the current token and get **scores** for the next token, and we choose the token with the highest score. So this table should have $V$ spots (remember $V$ is the size of our vocabulary or the number of unique tokens). And in each spot, we should have a score for all of the tokens, which means we should have $V$ scores. This means our table will be size $V \times V$. 

We can code this out as follows.
```python
token_table = nn.Embedding(vocab_size, vocab_size)
```
This creates a matrix:
$$
W \in \R ^ {V \times V}
$$
Given an input token $x_t \in \{0, \dots, V-1 \}$, the table returns **logits**
$$
\ell _t = W[x_t] \in \R^V
$$
The logits are our scores for each token. We can convert these scores into a probability distribution, and then sample from that. That will give us our next, most likely token.

The way we do that is by passing the logits through a function called the **softmax**. This will take our raw scores and scale each value to be between 0 and 1 and make sure all the values sum up to 1. This is the requirement for a probability distribution. Finally, we sample from this distribution to get our next token.

In PyTorch, this is coded as:
```python
class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        self.token_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, current_token):
        next_token_logits = self.token_table(current_token)
        return next_token_logits
    
    def generate(self, current_token, number_of_new_tokens):
        result = []
        for _ in range(number_of_new_tokens):
            # make prediction
            next_token_logits = self.forward(current_token)
            # convert from probability distribution
            next_token_dist = F.softmax(next_token_logits, dim=-1)
            # sample from distribution
            next_token = torch.multinomial(next_token_dist, num_samples=1)
            # append next token to result
            result.append(next_token)
            # move onto next time step
            curr_token = next_token
        return result
```

Right now, this simple bigram model doesn't use any of the previous tokens to inform its predictions. Formally, we are learning
$$
p(x_{t+1} \mid x_t)
$$
but our task requires learning 
$$
p(x_{t+1} \mid x_1, \dots, x_t)
$$

## The mathematical trick in attention
How can we use the information from our context. Well, the first thing we need to do is assume our tokens contain information. Instead of simple scalars, we can say our tokens are vectors of size $C$, which stands for channels. These fancier tokens are called **token embeddings.** So suppose our scalar tokens $x_1, x_2, \dots x_T \in \R$ have been embedded into a vector space and are now
$$
x_1, x_2, \dots, x_T \in \R ^C
$$

The simplest way we can use our context is by looking at the previous tokens and **take their average**, which would represent some sort of feature vector. Note that taking the average is pretty lossy because we lose quite a bit of information (like each tokens position and we treat all tokens equally). We will see on how to recover that later. So we want
$$
\bar{x}_t = \frac{1}{t} \sum_{i=1}^t x_i
$$
This is pretty simple to code out. Assume that we have a sequence of 8 tokens and each token is in a 2D vector space:
```python
T, C = 8, 2 # time, channels
x = torch.randn(T, C) # dummy input

x_bar = torch.zeros((T, C))
for t in range(T):
    prev_context = x[:t+1] # get all tokens up to timestep t
    x_bar[t] = torch.mean(prev_context, 0) # take the average

print(x_bar[0] == x[0]) # true, because the average only consists of the first token
print(x_bar[1] == x[1]) # false, because the average now consists of the first 2 tokens
```

Finding the average from a for loop is a pretty slow operation. Let's see how to speed this up. 

We can think of the average as a **weighted sum**. For example, if we are taking the average of 4 inputs, then we are multiplying each input by 0.25 and adding them up. This is literally the summation expanded out
$$
\bar{x}_4 = \frac{1}{4} \sum_{i=1}^4 x_i \\
\bar{x}_4 = \frac{1}{4} (x_1 + x_2 + x_3 + x_4) \\
\bar{x}_4 = \frac{1}{4} x_1 + \frac{1}{4} x_2 + \frac{1}{4} x_3 + \frac{1}{4} x_4
$$

We can express this weighted sum as a **dot product** between two vectors: the inputs and their respective weights. Suppose each $x_i \in \R^C$ is a vector (not a scalar). Then the average at timestep 4 can be written as
$$
\bar{x}_4 = 
\begin{bmatrix}
\frac{1}{4} & \frac{1}{4} & \frac{1}{4} & \frac{1}{4}
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2 \\
x_3 \\
x_4
\end{bmatrix}
$$

The first vector is our weights and the second transposed vector is our input.

Now remember in each time step, we have one more token added to our context when predicting the next target token.
```
t=0: When context is [x1], target is x2
t=1: When context is [x1, x2], target is x3
t=2: When context is [x1, x2, x3], target is x4
```

At timestep $t = 0$, the context is just $[x_1]$, so the average is
$$
\bar{x}_1 = 1 \cdot x_1
$$

At timestep $t=1$, the context is $[x_1, x_2]$, so the average is
$$
\bar{x}_2 = \frac{1}{2}  x_1 + \frac{1}{2} x_2
$$

At timestep $t=2$, the context is $[x_1, x_2, x_3]$, so the average is
$$
\bar{x}_3 = \frac{1}{3} x_1 + \frac{1}{3} x_2 + \frac{1}{3} x_3
$$

And remember how we just reformulated the average into a dot product between the inputs and the weights? The weights at each timstep would clearly look like 
$$
[1]
$$
$$
[\tfrac{1}{2}, \, \tfrac{1}{2}]
$$
$$
\left[\tfrac{1}{3}, \, \tfrac{1}{3}, \, \tfrac{1}{3}\right]
$$

Now with these two sets of input and weight vectors, we can arrange them into matrices.
Let the input matrix be
$$
X =
\begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix}
\in \mathbb{R}^{3 \times C}.
$$

And let the weight matrix be
$$
W =
\begin{bmatrix}
1 & 0 & 0 \\[3px]
\frac{1}{2} & \frac{1}{2} & 0 \\[3px]
\frac{1}{3} & \frac{1}{3} & \frac{1}{3}
\end{bmatrix}
\in \mathbb{R}^{3 \times 3}.
$$

Notice that $W$ is lower triangular. That is important. It ensures that at timestep $t$, only tokens up to $t$ are used in the average If we are predicting token 5, we should only be averaging tokens 1-4 from the data. Using information from the actual token 5 in the data or tokens 6, 7, $\dots T$ is cheating! The zeros prevent any *future* token from leaking into the computation. This is called the **causal mask**.

And to perform the dot product and get the averaged out tokens, we just do a **matrix multiplication**:
$$
\bar{X} = W X \in \R^{3 \times C}
$$

The result contains
$$
\bar{X} =
\begin{bmatrix}
\bar{x}_1 \\
\bar{x}_2 \\
\bar{x}_3
\end{bmatrix}
$$

Each row of $\bar{X}$ is the average of all tokens up to that timestep.

That's the mathematical trick in attention! Instead of writing for-loops that repeatedly sum over previous tokens, the entire operation can be expressed as a single matrix multiplication that acts as a dot product between weights and the inputs.

This can be coded very easily
```python
T, C = 8, 2 # time, channels
x = torch.randn(T, C) # same dummy input

# create weights matrix for weighted sum
weights = torch.ones(T, T) # all 1s matrix
weights = torch.tril(weights) # now upper triangle is 0s and only lower triangle is 1s
weights = weights / torch.sum(weights, dim=1, keepdims=True) # have each row sum up to 1

# average out tokens via matrix multiply
xbar2 = weights @ x
```
Again, we start with the same dummy input where the sequence length is 8 and each token is in a 2D vector space. Then we instantiate the weights matrix $W$ of shape $T \times T$ where every value is 1. 
$$
W = 
\begin{bmatrix}
1 & 1 & 1 \\[3px]
1 & 1 & 1 \\[3px]
1 & 1 & 1
\end{bmatrix}
$$

We pass this matrix into `torch.tril()`, which is a nifty method returns the same matrix but the upper triangle is made all 0.
$$
W = 
\begin{bmatrix}
1 & 0 & 0 \\[3px]
1 & 1 & 0 \\[3px]
1 & 1 & 1
\end{bmatrix}
$$

Then we normalize each row to sum up to 1 by finding the total sum along each row (`dim=1`) and dividing each element in that row by the sum.
$$
W =
\begin{bmatrix}
1 & 0 & 0 \\[3px]
\frac{1}{2} & \frac{1}{2} & 0 \\[3px]
\frac{1}{3} & \frac{1}{3} & \frac{1}{3}
\end{bmatrix}
$$

Finally, we take the average by executing the matrix multiplication between the weights and the inputs.

## Adding softmax
Notice how in the last step we were normalizing the matrix so that each row sums up to 1. The keen reader will realize that we can use softmax to do that for us.

Right now, our matrix before the normalization looks like
$$
W = 
\begin{bmatrix}
1 & 0 & 0 \\[3px]
1 & 1 & 0 \\[3px]
1 & 1 & 1
\end{bmatrix}
$$
For softmax to generate our desired normalized output, we need to input something like
$$
W = 
\begin{bmatrix}
0 & -\infty & -\infty \\[3px]
0 & 0 & -\infty \\[3px]
0 & 0 & 0
\end{bmatrix}
$$

This is because softmax is defined as
$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}.
$$
So if an entry is $-\infty$, then $e^{-\infty} = 0$, which means those positions get probability 0. And if all the unmasked entries are 0, softmax makes them uniform.


In code, this would look like
```python
T, C = 8, 2 # time, channels
x = torch.randn(T, C) # same dummy input

tril = torch.tril(torch.ones(T, T))
weights = torch.zeros((T,T))
weights = weights.masked_fill(tril == 0, float('-inf'))
print(weights)
print("---")

weights = F.softmax(weights, dim=1) # normalize 
print(weights)

xbar3 = weights @ x
```
```
tensor([[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0., 0., 0., 0.]])
---
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000],
        [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
        [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.0000],
        [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]])
```
We create a `tril` matrix where the upper triangle is all zeros, and a `weights` matrix of the same shape but the entire thing is zeros. Then we lay the `tril` matrix on the `weights` one, and wherever `tril` is 0 (the upper triangle half), we change that spot in `weights` to $- \infty$. Then we apply softmax to the `weights` to normalize like before. And finally, we matrix multiply that with the input `x` to get our averaged scores of previous tokens.

## Queries, Keys, and Values
Ok, this is good and all, but a token shouldn't have to give equal weightage to all of its previous tokens. It should be able to pay special attention (hint hint!) to certain tokens that have more relevant information to offer. For example, if I am a token "dog", I should pay more attention to tokens that describe me like "fluffy" or "running" instead of other tokens that don't matter as much--it's data dependent.

How do we implement that? Imagine each token emits two vectors: a **query** and **key**. These are different metadata about itself. The query vector, roughly speaking, asks the question "What am I looking for?" The key vector says "This is what information I contain."

Formally, each token has an embedding $x_t \in \R ^C$. We want to **transform** these vectors from dimension $C$ to another $d_k$. If you remember from linear algebra, that's exactly what a matrix does: transform a vector from one dimension to a vector in another dimension by linearly projecting it. So we learn two **linear projections**
$$
W_Q \in \R^{C \times d_k}, \quad W_K \in \R^{C \times d_k}
$$

To get our query and key metadata for a token, we comput
$$
q_t = x_t W_Q, \quad k_t = x_t W_K
$$
where $q_t, k_t \in \R ^{d_k}$ (just fancy notation saying that the vectors are now in another dimension).

If we stack all token embeddings into a matrix
$$
X \in \R ^{T \times C},
$$
then we get
$$
Q = X W_Q, \quad K = X W_K \in \R ^ {T \times d_k}
$$

Implementing this in code is very easy. We can use `torch.nn.Linear()` to make the learnable projections:
```python
T, C = 8, 32 # time, bigger channels now
x = torch.randn(T, C) # dummy input shape (T, C)

head_size = 16 # this is our d_k 

# these linear layers (simple matmuls) are what we use to emit the key and query vectors
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)

k = key(x)   # shape (T, 16)
q = query(x) # shape (T, 16)
```

Now to have each token communicate with each other, we take the dot product ($Q \cdot K)$. Each token's query vector will be multiplied by every other token's key vector. If a key and query vector are aligned, then their dot product will be very high and that's how we know they are related.

Formally, we compute the following **attention score** matrix
$$
S = QK ^\top \in \R ^{T \times T},
$$
where $S_{t, i} = q_t \cdot k_i$.

Row $t$ of $S$ contains how much token $t$ is interested in every other token. And this score matrix is what replaces our initial average score matrix! Now each weight can be custom tuned to how much a token should pay attention to other tokens.

```python
# Up to this point, NO communication has happened yet between the tokens. 
# We've only computed their key and query vectors. 
# The dot product below is when the communication happens!
weights = q @ k.T # (T, 16) @ (16, T) -> (T, T)
```
From here, we take our scores matrix and apply softmax like before.
```python
tril = torch.tril(torch.ones(T, T))
weights = weights.masked_fill(tril == 0, float('-inf'))
weights = F.softmax(weights, dim=1)
```
```
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.1574, 0.8426, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2088, 0.1646, 0.6266, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5792, 0.1187, 0.1889, 0.1131, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0294, 0.1052, 0.0469, 0.0276, 0.7909, 0.0000, 0.0000, 0.0000],
        [0.0176, 0.2689, 0.0215, 0.0089, 0.6812, 0.0019, 0.0000, 0.0000],
        [0.1691, 0.4066, 0.0438, 0.0416, 0.1048, 0.2012, 0.0329, 0.0000],
        [0.0210, 0.0843, 0.0555, 0.2297, 0.0573, 0.0709, 0.2423, 0.2391]]
```
Notice how the weights are not even now across each row! Tokens are paying more attention to some tokens than others and giving them higher scores.

Finally, we don't aggregate across the raw input `x` like before. Instead, we calculate another vector from the input, called **value**. We aggregate across this. The value vector says, "If you find me interesting (aka we have a high $Q \cdot K$ dot product), this is what I can communicate to you."

Formally, we learn a third projection
$$
W_V \in \R ^ {C \times d_v},
$$
and compute
$$
v_t = x_t W_v, \quad V=X W_V \in \R ^ {T \times d_v}
$$
```python
value = nn.Linear(C, head_size, bias=False) # our d_v == d_k
v = value(x)

out = weights @ v
```

## Minor notes

A few things to note:
* Keys and queries are projected to $d_k$ and values are projected to $d_v$. In this case, $d_k = d_v$.
- Attention is simply a **communication** mechanism. You can view it as nodes in a directed graph looking at each other and aggregating information with a weighted sum from all its neighbors, where the weights are data dependent.
* Attention simply aggregates over a set of vectors--there is no notion of space or position. This is why we need to add positional encodings to the vectors.
- The `tril()` creates something called the causal mask, which prevents all tokens from seeing and communicating each other. Sometimes, this is important like in a task like sentiment classification. In this task, we need to see the entire sentence before we classify it as happy or sad. In such tasks, we can implement attention the same way, just without the mask.
* **Self-attention** just means that the keys and values come from the same source as the queries. **Cross-attention** means that the queries still get produced from input `x`, but the keys and values come from somewhere else (like an encoder block).
- Lastly, we can't just use `weights` as is. Remember, `weights` is calculated by taking the dot product between $K$ and $Q$. Sometimes, these dot products can be really high and cause numerical instability. So, we need to control the variance of the matrix by **scaling by $\frac{1}{\sqrt{d_k}}$**. This way, the softmax output will remain fairly diffused and not too saturated (think of a really peaky probability distribution versus a fairly smooth one).

## Fin
Finally, we have implemented the simplest form of attention: scaled dot product attention (SDPA for short)!
$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} \right)V
$$

In the next blog post, we will see how the attention operation fits into a bigger model architecture called the transformer. We will compare the more powerful transformer against our humble bigram model and measure the difference in performance. In the end, we will actually have a model that can decently generate text!