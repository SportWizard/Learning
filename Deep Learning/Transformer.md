# What is transformer?
Transformer is an architecture in [[Deep Learning]] widely used for text generation, translation, summarization, question answering, sentiment analysis, etc. With one of the most popular use case being ChatGPT (GPT stands for Generative Pre-trained Transformer)

# What problem is transformer used for?
Transformer is used in a lot areas of deep learning such as natural language processing (NLP), speech and audio processing, etc

# Objective/loss function
Depends on the specific task it is trained for

# How does transformer work?
Transformer uses several steps such as embedding the tokens (e.g. hi, !, combi, nation, 8, etc). Then, using an attention block to extract relationship between. Final, The information passed through a [[Fully Connected Neural Network]] for prediction. The use of an attention block and [[Fully Connected Neural Network]] can be repeated to increase the complexity of the neural network with the last [[Fully Connected Neural Network]] using [[Softmax]] to output the the probability distribution of all the tokens

Steps:
- Repeat for $n$ times
	- Embedding
	- Attention blocks
	- [[Fully Connected Neural Network]]
- [[Fully Connected Neural Network]] with [[Softmax]] to generate output that ranks tokens into a probability distribution

# Embedding
Embedding is used to convert pieces of text into vector using a redefined vocabulary or an embedding matrix contain weights (the weights are trained using data), where each vector in the matrix corresponds to a token. This is done because number are easier to understand by computers. These vectors can be thought of as vector in a high dimensional space

One thing to note is token that has similar meaning tends to point in the similar direction

![[embedded-vector.png]]

Embedding also encodes the position of the word (e.g. The King doth wake tonight and takes his rouse ... - the word "King" will have an embedded position of 2)

![[position-embedding.png]]
# Attention
Attention is responsible for figuring out which token in the sequence (or sentence) has relevant meaning to the current token and builds context-aware representation (enriching the meaning) of the current token (e.g. the word model in machine learning model is different than a fashion model)

![[attention1.png]]

##### Query, key, and value
In order to get token relevant to each other, it would need to compare every embedded tokens with every other embedded tokens, including itself. To do this, each embedded token is projected into a query, key, and value

**Query:** a vector, $\vec{q}$, that is "seeking" relevant information by comparing itself to keys. All queries can be combined into a query matrix, $Q$
$$
Q = W_Q X
$$
$$
\vec{q}_i = A \vec{x}_i, \; \forall i \in \{1, 2, \cdots, T\}
$$
$Q \in \mathbb{R}^{l \times T}$ is the query matrix
$W_Q \in \mathbb{R}^{l \times n}$ is the query projection
$X = \begin{bmatrix}\vec{x}_1 & \vec{x}_2 & \cdots & \vec{x}_T\end{bmatrix} \in \mathbb{R}^{o \times T}$ where each $\vec{x}_i$ represent a embedded token

$\vec{q} \in \mathbb{R}^l$ is a query
$\vec{x} \in \mathbb{R}^n$ is a token

$n \in \mathbb{R}$ is the number of embedded tokens
$l \in \mathbb{R}$ is the dimension of each query, $\vec{q}$, and key, $\vec{k}$
$T \in \mathbb{R}$ is the number of queries, key, and value

**Key:** a vector, $\vec{k}$, that is "offering" itself to be matched with queries. All keys can be combined into a key matrix, $K$
$$
K = W_K X
$$
$$
\vec{k}_i = A \vec{x}_i, \; \forall i \in \{1, 2, \cdots, T\}
$$
$K \in \mathbb{R}^{l \times T}$ is the query matrix
$W_K \in \mathbb{R}^{l \times n}$ is the key projection
$X = \begin{bmatrix}\vec{x}_1 & \vec{x}_2 & \cdots & \vec{x}_T\end{bmatrix} \in \mathbb{R}^{o \times T}$ where each $\vec{x}_i$ represent a embedded token

$\vec{k} \in \mathbb{R}^l$ is a key
$\vec{x} \in \mathbb{R}^n$ is a token

$n \in \mathbb{R}$ is the number of embedded tokens
$l \in \mathbb{R}$ is the dimension of each query, $\vec{q}$, and key, $\vec{k}$
$T \in \mathbb{R}$ is the number of queries, key, and value

**Value ($\vec{v}$):** a vector that is the actual content of a token. All values can be combined into a value matrix, $V$
$$
V = W_V X
$$
$$
\vec{v}_i = A \vec{x}_i, \; \forall i \in \{1, 2, \cdots, T\}
$$
$V \in \mathbb{R}^{o \times T}$ is the query matrix
$W_V \in \mathbb{R}^{o \times n}$ is the value projection
$X = \begin{bmatrix}\vec{x}_1 & \vec{x}_2 & \cdots & \vec{x}_T\end{bmatrix} \in \mathbb{R}^{o \times T}$ where each $\vec{x}_i$ represent a embedded token

$\vec{v} \in \mathbb{R}^o$ is a value
$\vec{x} \in \mathbb{R}^n$ is a token

$n \in \mathbb{R}$ is the number of embedded tokens
$o \in \mathbb{R}$ is the dimension of each value, $\vec{v}$
$T \in \mathbb{R}$ is the number of queries, key, and value

##### Query, key, and value projections
The projection is a applied to capture different important features. For example, the sequence "Fluffy brown squirrel", the projection could be applied to token "squirrel" to capture any adjective that will be relevant to this token. While another projection will be applied to the tokens "Fluffy" and "brown" to be a candidate.

Note: In the same sequence, all projections are updated using data. Furthermore, all query projections are the same, all key projections are the same, and all value projections are the same

##### Finding similarity between tokens
Using the query matrix and key matrix, we could find which query and key have the highest similarity by doing the [[Dot product]]
$$
A = \frac{Q^\intercal K}{\sqrt{d_k}}
$$
$$
A =
\begin{bmatrix}
\frac{\vec{q}_1^\intercal \vec{k}_1}{\sqrt{d_k}} & \frac{\vec{q}_1^\intercal \vec{k}_2}{\sqrt{d_k}} & \cdots & \frac{\vec{q}_1^\intercal \vec{k}_T}{\sqrt{d_k}} \\
\frac{\vec{q}_2^\intercal \vec{k}_1}{\sqrt{d_k}} & \frac{\vec{q}_2^\intercal \vec{k}_2}{\sqrt{d_k}} & \cdots & \frac{\vec{q}_2^\intercal \vec{k}_T}{\sqrt{d_k}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\vec{q}_T^\intercal \vec{k}_1}{\sqrt{d_k}} & \frac{\vec{q}_T^\intercal \vec{k}_2}{\sqrt{d_k}} & \cdots & \frac{\vec{q}_T^\intercal \vec{k}_T}{\sqrt{d_k}}
\end{bmatrix}
$$
$A \in \mathbb{R}^{T \times T}$ is the attention weight/score
$Q \in \mathbb{R}^{l \times T}$; is the query matrix
$K \in \mathbb{R}^{l \times T}$ is the query matrix
$d_k = l \in \mathbb{R}$ normalizes the output for numerical stability

$\vec{q}_i \in \mathbb{R}^l$ is a query
$\vec{k}_i \in \mathbb{R}^l$ is a key

**Why the dot product?**
The [[Dot product]] tell us whether two vector are pointing relatively in the same, orthogonal, or in opposite direction

$$
\vec{v} \cdot \vec{u} = \|\vec{v}\| \|\vec{u}\| \cos \theta
$$
The use of the cosine will be close to 1 if two vectors points relative to each other, since from looking at the unit circle, $\cos (0) = 1$, close to 0 if they are orthogonal, $\cos (\frac{\pi}{2}) = 0$, and negative if opposite of each other, $\cos (\pi) = -1$

![[Score.png]]
Note: the image is a transpose of my equation

##### Convert into probability distribution
Given $Q^\intercal \in \mathbb{R}^{T \times l}$ and $K \in \mathbb{R}^{l \times T}$, we can assume each vector in $A$ is a row vector representing a query, with elements representing an attention weight/score of the certain key
$$
A =
\begin{bmatrix}
\begin{bmatrix}a_{1 \; 1} & a_{1 \; 2} & \cdots & a_{1 \; T}\end{bmatrix} \\
\begin{bmatrix}a_{2 \; 1} & a_{2 \; 2} & \cdots & a_{2 \; T}\end{bmatrix} \\
\vdots \\
\begin{bmatrix}a_{T \; 1} & a_{T \; 2} & \cdots & a_{T \; T}\end{bmatrix}
\end{bmatrix}
$$
$A \in \mathbb{R}^{T \times T}$ is the attention weight/score
$a_{i \; j} \in \mathbb{R}$ is the attention weight

To convert into probability distribution, [[Softmax]] is applied to each vector outputting a matrix of probability distribution, where each vector sums up to 1

$$
\text{softmax}(A) =
\begin{bmatrix}
\text{softmax}(\begin{bmatrix}a_{1 \; 1} & a_{1 \; 2} & \cdots & a_{1 \; T}\end{bmatrix}) \\
\text{softmax}(\begin{bmatrix}a_{2 \; 1} & a_{2 \; 2} & \cdots & a_{2 \; T}\end{bmatrix}) \\
\vdots \\
\text{softmax}(\begin{bmatrix}a_{T \; 1} & a_{T \; 2} & \cdots & a_{T \; T}\end{bmatrix})
\end{bmatrix}
$$
$A \in \mathbb{R}^{T \times T}$ is the attention weight/score matrix
$a_{i \; j} \in \mathbb{R}$ is the attention weight/score

where for $a_{i \; j}$, $i$ is the $q_i$ and $j$ is the $k_j$. (e.g. $a_{5 \; 3}$ is the attention weight/score of the 5th query with the 3rd key)

##### Causal attention and masking
The purpose of causal attention is it is more efficient to prevent future token from influencing the current token since that gives away the answer of what comes next after the current token. For example, the sequence (or sentence) "The quick brown fox jumps over the lazy dog", you don't want the token "jumps" to influence the token "fox" and other token before "jumps", 'cause that gives away the answer

To achieve causal attention you can use a trick called masking. Masking is convert future token into $-\infty$, so that when [[Softmax]] is applied, it will be 0

![[masking.png]]
Note: the image is a transpose of my equation

##### Applying attention weight/score to value
To sum up the over contribution of all the token in the sequence (or sentence) to a token, we would multiple the corresponding probability with the values and sum it

$$
Z = \text{softmax}(A) V^\intercal
$$
$$
\text{softmax}(A) =
\begin{bmatrix}
\text{softmax}(\begin{bmatrix}a_{1 \; 1} & a_{1 \; 2} & \cdots & a_{1 \; T}\end{bmatrix}) \\
\text{softmax}(\begin{bmatrix}a_{2 \; 1} & a_{2 \; 2} & \cdots & a_{2 \; T}\end{bmatrix}) \\
\vdots \\
\text{softmax}(\begin{bmatrix}a_{T \; 1} & a_{T \; 2} & \cdots & a_{T \; T}\end{bmatrix})
\end{bmatrix}

\begin{bmatrix}
\vec{v}_1^\intercal \\ \vec{v}_2^\intercal \\ \vdots \\ \vec{v}_T^\intercal
\end{bmatrix}
$$

$$
Z =
\begin{bmatrix}\vec{z}_1 & \vec{z}_2 & \cdots & \vec{z}_T\end{bmatrix} =

\begin{bmatrix}
\begin{bmatrix}
a'_{1 \; 1} v_{1 \; 1} + a'_{1 \; 2} v_{2 \; 1}+ \cdots + a'_{1 \; T} v_{T \; 1}
\\
a'_{2 \; 1} v_{1 \; 1} + a'_{2 \; 2} v_{2 \; 1} + \cdots + a'_{2 \; T} v_{T \; 1}
\\
\vdots
\\
a'_{T \; 1} v_{1 \; 1} + a'_{T \; 2} v_{2 \; 1} + \cdots + a'_{T \; T} v_{T \; 1}
\end{bmatrix}

&

\begin{bmatrix}
a'_{1 \; 1} v_{1 \; 2} + a'_{1 \; 2} v_{2 \; 2} + \cdots + a'_{1 \; T} v_{T \; 2}
\\
a'_{2 \; 1} v_{1 \; 2} + a'_{2 \; 2} v_{2 \; 2} + \cdots + a'_{2 \; T} v_{T \; 2}
\\
\vdots
\\
a'_{T \; 1} v_{1 \; 2} + a'_{T \; 2} v_{2 \; 2} + \cdots + a'_{T \; T} v_{T \; 2}
\end{bmatrix}

&

\cdots

&

\begin{bmatrix}
a'_{1 \; 1} v_{1 \; T} + a'_{1 \; 2} v_{2 \; T} + \cdots + a'_{1 \; T} v_{T \; o}
\\
a'_{2 \; 1} v_{1 \; T} + a'_{2 \; 2} v_{2 \; T} + \cdots + a'_{2 \; T} v_{T \; o}
\\
\vdots
\\
a'_{T \; 1} v_{1 \; T} + a'_{T \; 2} v_{2 \; T} + \cdots + a'_{T \; T} v_{T \; o}
\end{bmatrix}
\end{bmatrix}
$$
$Z \in \mathbb{R}^{o \times T}$ is the attention head
$\vec{z}_i \in \mathbb{R}^o$ is the attention for the embedded token, $x_i$
$V \in \mathbb{R}^{o \times T}$ is the query matrix
$A \in \mathbb{R}^{T \times T}$ is the attention weight/score matrix

$a_{i \; j} \in \mathbb{R}$ is the attention weight/score
$a'_{i \; j} \in \mathbb{R}$ is the probability distribution of the attention weight/score
$\vec{v} \in \mathbb{R}^o$ is a value

##### Enrich the embedded token
The attention, $Z$, is then used to add the origin embedded token

$$
X' = X + Z
$$
$$
X' =

\begin{bmatrix}\vec{x}_1 + \vec{z}_1 & \vec{x}_2 + \vec{z}_2 & \cdots & \vec{x}_T + \vec{z}_T\end{bmatrix} =

\begin{bmatrix}
\begin{bmatrix}x_{1 \; 1} \\ x_{2 \; 1} \\ \vdots \\ x_{o \; 1}\end{bmatrix} +
\begin{bmatrix}
v_{1 \; 1} a_{1 \; 1} + v_{1 \; 2} a_{2 \; 1} + \cdots + v_{1 \; T} a_{T \; 1}
\\
v_{2 \; 1} a_{1 \; 1} + v_{2 \; 2} a_{2 \; 1} + \cdots + v_{2 \; T} a_{T \; 1}
\\
\vdots
\\
v_{o \; 1} a_{1 \; 1} + v_{o \; 2} a_{2 \; 1} + \cdots + v_{o \; T} a_{T \; 1}
\end{bmatrix}

&

\begin{bmatrix}x_{1 \; 2} \\ x_{2 \; 2} \\ \vdots \\ x_{o \; 2}\end{bmatrix} +
\begin{bmatrix}
v_{1 \; 1} a_{1 \; 2} + v_{1 \; 2} a_{2 \; 2} + \cdots + v_{1 \; T} a_{T \; 2}
\\
v_{2 \; 1} a_{1 \; 2} + v_{2 \; 2} a_{2 \; 2} + \cdots + v_{2 \; T} a_{T \; 2}
\\
\vdots
\\
v_{o \; 1} a_{1 \; 2} + v_{o \; 2} a_{2 \; 2} + \cdots + v_{o \; T} a_{T \; 2}
\end{bmatrix}

&

\cdots

&

\begin{bmatrix}x_{1 \; T} \\ x_{2 \; T} \\ \vdots \\ x_{o \; T}\end{bmatrix} +
\begin{bmatrix}
v_{1 \; 1} a_{1 \; T} + v_{1 \; 2} a_{2 \; T} + \cdots + v_{1 \; T} a_{T \; T}
\\
v_{2 \; 1} a_{1 \; T} + v_{2 \; 2} a_{2 \; T} + \cdots + v_{2 \; T} a_{T \; T}
\\
\vdots
\\
v_{o \; 1} a_{1 \; T} + v_{o \; 2} a_{2 \; T} + \cdots + v_{o \; T} a_{T \; T}
\end{bmatrix}
\end{bmatrix}
$$
$X' \in \mathbb{R}^{o \times T}$ is the attention
$X = \begin{bmatrix}\vec{x}_1 & \vec{x}_2 & \cdots & \vec{x}_T\end{bmatrix} \in \mathbb{R}^{o \times T}$ where each $\vec{x}_i$ represent a embedded token
$Z \in \mathbb{R}^{o \times T}$ is the attention

![[attention2.png]]

##### Self-attention
Apply attention within the same sequence (or sentence)

##### Cross-attention
Apply attention between two different sequence (or sentence) (e.g. translating a sentence to another language)

# Multi-head attention
Multi-head attention is allow for the model capture of more information. This is because each attention head uses a distinct query, key, and value projection, where each attention head captures different kinds of information. All the information will combine together and processed by the [[Fully Connected Neural Network]]

![[multi-head-attention.png]]

# Code
```python
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

class config():
  def __init__(self, batch_size=10, n_layer=12, n_head=12, n_embd=768, block_size=1024, vocab_size=50304, causal=True, device="cpu"):
    assert n_embd %  n_head == 0
    self.batch_size = batch_size
    self.n_embd = n_embd # Size of the embedding. The use of embedding is to map a word to a vector (and the vector is learned during training) and then it adds the position of the word. e.g. Embedding("cat") + Position(2) -> [0.8, -1.5, 0.7]
    self.block_size = block_size
    self.n_head = n_head # Number of multi-headed attention (multiple layer of attention). Each attention head is used to capture different information about a sequence
    self.causal = causal
    self.device = device
    self.n_layer = n_layer
    self.vocab_size = vocab_size

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)) # Mask the upper right triangle of the square matrix by making it negative infinite. When softmax is applied, it will be zero

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim

        # Query, key, and values are matrix (also can thought of as a vector containing vector representation of the embedded tokens)

        # weight_q: what is the current token is looking for. e.g. For the token "it", Q might encode the need to find a noun (like "cat") for coreference resolution
        # weight_k: how other tokens identify themselves. e.g. The key for "cat" might signal it’s a noun that can be referenced by pronouns like "it"
        # weight_v: what content the token provides. e.g. The value for "cat" encodes semantic features (e.g., animacy, noun type

        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)

        # hs: number of attention head
        # T: number of queries, keys, and values
        # nh: dimension of queries, keys, and values (column vector)

        # q: weight_q dot embedded tokens = q_i <- column vector representation of the token, where each asks the question "what token is most similar to me"
        # k: weight_k dot embedded tokens = k_i <- column vector representation of the token, where each is a candidate that will be compared with q_i using dot product
        # v: weight_v dot embedded tokens = v_i <- column vector representation of the token, where each will be mutliplied a corresponding attention weight (how similar it is) at position i and determining how much this token contributes/relevant to the result
        q = q.view(B, T, C // self.n_head, self.n_head).transpose(1, 3) # (B, hs, nh, T)
        k = k.view(B, T, C // self.n_head, self.n_head).transpose(1, 3) # (B, hs, nh, T)
        v = v.view(B, T, C // self.n_head, self.n_head).transpose(1, 3) # (B, hs, nh, T)

        # attention_weight = softmax(q^T dot k / sqrt(d_k)), where 0 <= j <= number of token in the sequence and d_k is the number of dimension to represent the query and key
        # attention_weight is a matrix (can be thought of as vector containing vectors), where each vector contains probabilities of every token's similarity to itself
        att = (q.transpose(-2, -1) @ k) * (1.0 / math.sqrt(q.size(-2))) # (B, hs, T, nh) x (B, hs, nh, T) -> # (B, hs, T, T)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float("-inf")) # Causal Mask: used to prevent future token from influencing the current prediction
        att = F.softmax(att, dim=-1) # Row-wise softmax

        # y_i = sum(v_i * attention_weight_ij) <- scalar multiplication, then vector addition for the sum, where 0 <= j <= number of token's probabilities. e.g. y_i = sum(v_i * attention_weight_ij) = [0.5, 2.3, 1.3] * 0.3 + [0.3, 0.2, 0.5] * 0.64 + ...
        # The purpose of the sum is the merge all relevant words in the sequence and that will dynamically shape the current token at position i. This is known as context-aware
        # The output, y, is a matrix, where each vector is a context-aware (a token's representation that is dynamically shaped by all relevant words in the sequence)
        y = att @ v.transpose(-2, -1) # (B, nh, T, T) x (B, hs, T, nh) -> (B, hs, T, nh)
        y = y.transpose(1, 2).transpose(2, 3).contiguous().view(B, T, C) # re-assemble all head outputs side by side - (B, T, nh, hs)
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm([config.n_embd])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm([config.n_embd])

        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm([config.n_embd]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("Number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        # Return the number of parameters in the model.
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution (num_samples=1 sample 1 character)
            idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# -----------------------------------------------------------------------------
# default config values
# I/O

# When doing multi-head attention, the vector of the embedded token is split among all the attention head

######### model size ##########
n_layer = 3 # num of layers (Transformer block - the multi-head attention and the MLP/FCNN)
n_head = 3 # num of attn heads per layer (multi-head attention) More attention head makes the model more complex (could cause overfitting)
head_size = 67 # size of each attn head (where each attention head contains the splitted embedded token's vector) - lareger size makes the model more complex (could cause overfitting)
###############################

###### training hyperparameters ######
learning_rate = 8e-6 # max learning rate
max_iters = 20e3 # total number of training iterations
batch_size = 27 # mini-batch size - Larger batch size takes longer to process per batch
block_size = 128 # block size (or number of queries, keys, and values) (sequence length - the number of tokens the model processes at once) - Higher block size could use more memory
#####################################

# adamw optimizer
beta1 = 0.9
beta2 = 0.98

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 5000 # how many steps to warm up for
lr_decay_iters = max_iters # When it starts the min_lr, should be ~= max_iters per Chinchilla
min_lr = learning_rate / 10 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
eval_interval = 1000
eval_iters = 200

# system
device = "cuda" # examples: "cpu", "cuda", "cuda:0", "cuda:1" etc., or try "mps" on macbooks
dtype = "bfloat16" # "float32", "bfloat16", or "float16", the latter will auto implement a GradScaler

torch.manual_seed(1337)

n_embd = n_head * head_size # Embedding dimension is the number of dimensions is in the vector of the embedded token

data_tr  = np.array(train_ids, dtype=np.uint16)
data_val = np.array(val_ids, dtype=np.uint16)

# poor man's data loader
def get_batch(split):
    data = data_tr if split == "train" else data_val

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# init a new model from scratch
print("Initializing a new model from scratch")
conf = config(batch_size=batch_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd,\
              block_size=block_size, vocab_size=vocab_size, device=device)
model = GPT(conf)

model = model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler("cuba", enabled=(dtype == "float16"))

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2))

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# training loop
X, Y = get_batch("train") # fetch the very first batch
t0 = time.time()
iter_num = 0

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate

    logits, loss = model(X, Y)
    X, Y = get_batch("train")

    # backward pass, with gradient scaling if training in fp16
    scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        losses = estimate_loss()
        print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} (time lapses {dt:.4f} seconds)")

    iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
```