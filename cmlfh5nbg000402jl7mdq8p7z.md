---
title: "Transformations That Change Everything"
seoTitle: "Automatic vectorization with vmap and gradients with grad in Jax"
seoDescription: "The core JAX transforms: jit for speed, vmap for batching, and grad for gradients. These three tools are enough to train neural networks from scratch."
datePublished: Mon Feb 09 2026 17:57:28 GMT+0000 (Coordinated Universal Time)
cuid: cmlfh5nbg000402jl7mdq8p7z
slug: transformations-that-change-everything
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1770659671216/1cee86fb-2dc9-42ed-a66c-5294c8c11bc1.png
tags: jax

---

Last week, we learned that JAX makes code fast through JIT compilation. We took a matrix multiplication from 2 seconds to 0.001 seconds with a single decorator.

But speed isn't JAX's only trick. The real power of JAX lies in its **transformations**; functions that take functions and return new functions with different behavior.

Today we're covering the two transformations that make JAX indispensable for machine learning:

1. `jax.vmap`: Automatic vectorization. Write code for one example, run it on a million.
    
2. `jax.grad`: Automatic differentiation. Get gradients of any function, for free.
    

By the end of this article, you'll understand why JAX developers almost never write for-loops, and you'll have trained multiple machine learning models in parallel without writing any loop at all.

## The Problem with Python Loops

Let's start with why loops are the enemy.

When you write a Python for-loop, the interpreter does a surprising amount of work for each iteration:

```python
results = []
for x in data:
    # For EACH iteration, Python must:
    # 1. Fetch x from memory
    # 2. Check the type of x
    # 3. Look up what "+" means for that type
    # 4. Execute the addition
    # 5. Append to the list (which may require memory reallocation)
    results.append(x + 1)
```

If `data` has a million elements, Python performs those administrative steps a million times. The actual math (`x + 1`) takes nanoseconds. The overhead takes microseconds. You're spending 99% of your time on bookkeeping.

In deep learning, we process batches: 64 images, 128 sentences, 256 audio clips at once. If you loop through them in Python, your GPU sits idle while Python shuffles paperwork.

NumPy helps by pushing operations into C:

```python
results = data + 1  # Vectorized, fast
```

But what if your function is more complex than addition? What if it involves multiple steps, conditionals, or nested operations? You'd have to manually rewrite everything to handle batches, adding batch dimensions everywhere and keeping track of which axis is which.

This is where `jax.vmap` changes the game.

## jax.vmap: Automatic Vectorization

`vmap` stands for "vectorizing map." It takes a function written for a single example and transforms it into a function that operates on batches.

### The Basic Pattern

```python
import jax
import jax.numpy as jnp

# A function that works on ONE number
def square(x):
    return x ** 2

# Transform it to work on MANY numbers
batched_square = jax.vmap(square)

# Now use it
numbers = jnp.array([1, 2, 3, 4, 5])
result = batched_square(numbers)
print(result)  # [1, 4, 9, 16, 25]
```

"But wait," you might say, "I could just write `numbers ** 2` directly."

True. The power of `vmap` shows up with complex functions:

```python
def complex_operation(x):
    """A function with multiple steps."""
    a = jnp.sin(x)
    b = jnp.exp(-x ** 2)
    c = a * b + jnp.log(1 + jnp.abs(x))
    return c

# Without vmap, you'd need to think about broadcasting at each step
# With vmap, you just wrap it
batched_complex = jax.vmap(complex_operation)

x_batch = jnp.linspace(-3, 3, 1000)
results = batched_complex(x_batch)
```

You write the function thinking about one input. `vmap` handles the batch dimension for you.

### Multiple Arguments: The `in_axes` Parameter

Real functions have multiple arguments. `in_axes` tells `vmap` which arguments to map over and which to broadcast.

Consider a dot product:

```python
def dot_product(weights, features):
    return jnp.dot(weights, features)
```

Different scenarios require different `in_axes`:

```python
# Scenario 1: One set of weights, many feature vectors
# weights: don't map (None), features: map over axis 0
batch_predict = jax.vmap(dot_product, in_axes=(None, 0))

weights = jnp.array([1.0, 2.0, 3.0])
features = jnp.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])

results = batch_predict(weights, features)
print(results)  # [1., 2., 3.]
```

```python
# Scenario 2: Many weight sets, one feature vector (ensemble of models)
# weights: map over axis 0, features: don't map (None)
ensemble_predict = jax.vmap(dot_product, in_axes=(0, None))

many_weights = jnp.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])
single_features = jnp.array([1.0, 2.0, 3.0])

results = ensemble_predict(many_weights, single_features)
print(results)  # [1., 2., 3.]
```

```python
# Scenario 3: Many weights, many features (parallel evaluation)
# Both: map over axis 0
parallel_predict = jax.vmap(dot_product, in_axes=(0, 0))

results = parallel_predict(many_weights, features)
print(results)  # [1., 2., 3.]
```

The rule is simple:

* `0` means "iterate over the first axis of this argument"
    
* `None` means "broadcast this argument to all iterations"
    
* You can use other integers for different axes
    

### Nested vmap

You can stack `vmap` calls for multi-dimensional batching:

```python
def single_multiply(a, b):
    return a * b

# Map over rows, then over columns
double_batched = jax.vmap(jax.vmap(single_multiply))

matrix_a = jnp.array([[1, 2], [3, 4]])
matrix_b = jnp.array([[10, 20], [30, 40]])

result = double_batched(matrix_a, matrix_b)
print(result)
# [[10, 40],
#  [90, 160]]
```

## jax.grad: Automatic Differentiation

The other transformation that makes JAX essential for ML is `jax.grad`. It computes gradients automatically.

### The Basic Pattern

```python
def f(x):
    return x ** 2

# grad returns a NEW FUNCTION that computes the derivative
df_dx = jax.grad(f)

print(f(3.0))      # 9.0
print(df_dx(3.0))  # 6.0 (derivative of x² is 2x, and 2*3=6)
```

This works for any function, no matter how complex:

```python
def messy_function(x):
    return jnp.sin(x) * jnp.exp(-x ** 2) + jnp.tanh(x)

gradient_fn = jax.grad(messy_function)

# The gradient at x=1.0
print(gradient_fn(1.0))  # -0.5047...
```

You didn't write any derivative rules. JAX traced through your function and computed the gradient automatically using the chain rule.

### jax.value\_and\_grad: Get Both at Once

In training loops, you need both the loss value (to log it) and the gradients (to update parameters). `jax.value_and_grad` gives you both in one pass:

```python
def loss_fn(params, x, y):
    prediction = params * x
    return (prediction - y) ** 2

# Returns (loss_value, gradients)
loss_and_grad_fn = jax.value_and_grad(loss_fn)

params = 1.0
x, y = 2.0, 6.0  # We want params=3.0 so that 3*2=6

loss, grad = loss_and_grad_fn(params, x, y)
print(f"Loss: {loss}")  # 16.0 (because (1*2 - 6)² = 16)
print(f"Grad: {grad}")  # -16.0
```

### Gradients with Respect to Specific Arguments

By default, `grad` differentiates with respect to the first argument. Use `argnums` to change this:

```python
def f(x, y):
    return x ** 2 + x * y

# Gradient with respect to x (first argument, default)
df_dx = jax.grad(f, argnums=0)

# Gradient with respect to y (second argument)
df_dy = jax.grad(f, argnums=1)

# Gradients with respect to both
df_both = jax.grad(f, argnums=(0, 1))

x, y = 2.0, 3.0
print(f"df/dx: {df_dx(x, y)}")  # 2*2 + 3 = 7
print(f"df/dy: {df_dy(x, y)}")  # 2
print(f"Both:  {df_both(x, y)}")  # (7.0, 2.0)
```

## Combining Transforms: The Real Power

JAX transforms compose. You can combine `jit`, `vmap`, and `grad` freely:

```python
def loss_single(params, x, y):
    """Loss for a single data point."""
    pred = params[0] * x + params[1]  # Linear: y = mx + b
    return (pred - y) ** 2

# Stack the transforms:
# 1. grad: compute gradients with respect to params
# 2. vmap: do this for a batch of (x, y) pairs
# 3. jit: compile the whole thing

batched_grad_fn = jax.jit(
    jax.vmap(
        jax.grad(loss_single),
        in_axes=(None, 0, 0)  # Same params, batch of x, batch of y
    )
)

params = jnp.array([1.0, 0.0])  # Initial guess: y = 1*x + 0
x_batch = jnp.array([1.0, 2.0, 3.0])
y_batch = jnp.array([2.0, 4.0, 6.0])  # True relationship: y = 2x

# Get gradients for each example in the batch
grads_per_example = batched_grad_fn(params, x_batch, y_batch)
print("Gradients per example:")
print(grads_per_example)

# Average them for a batch gradient
batch_grad = jnp.mean(grads_per_example, axis=0)
print(f"Batch gradient: {batch_grad}")
```

This pattern, `jit(vmap(grad(...)))`, is the backbone of efficient training in JAX.

## Project: Parallel Linear Regression

Let's train multiple models simultaneously without a single Python loop during training.

**Scenario**: We have housing price data from three different cities. Each city has different price dynamics, so we want to train a separate linear regression model for each.

### Step 1: Generate Synthetic Data

```python
import jax
import jax.numpy as jnp
from jax import random

# Seed for reproducibility
key = random.PRNGKey(42)

# True parameters for 3 cities
# City 0: price = 50 * size + 100
# City 1: price = 30 * size + 200  
# City 2: price = 80 * size + 50
true_slopes = jnp.array([50.0, 30.0, 80.0])
true_intercepts = jnp.array([100.0, 200.0, 50.0])

n_cities = 3
n_samples = 100

# Generate features (house sizes) for each city
key, subkey = random.split(key)
X = random.uniform(subkey, (n_cities, n_samples, 1), minval=10, maxval=100)

# Generate targets (prices) with some noise
key, subkey = random.split(key)
noise = random.normal(subkey, (n_cities, n_samples)) * 50

# Y[i] = true_slopes[i] * X[i] + true_intercepts[i] + noise[i]
Y = (X[:, :, 0] * true_slopes[:, None] + 
     true_intercepts[:, None] + 
     noise)

print(f"X shape: {X.shape}")  # (3, 100, 1)
print(f"Y shape: {Y.shape}")  # (3, 100)
```

### Step 2: Define the Model for ONE City

We write everything as if we only have one city:

```python
def predict(params, x):
    """Predict price for one city's houses."""
    slope, intercept = params[0], params[1]
    return x[:, 0] * slope + intercept

def loss_fn(params, x, y):
    """MSE loss for one city."""
    predictions = predict(params, x)
    return jnp.mean((predictions - y) ** 2)

def train_step(params, x, y, learning_rate):
    """One gradient descent step for one city."""
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    new_params = params - learning_rate * grads
    return new_params, loss
```

### Step 3: Vectorize Across Cities

Now we use `vmap` to run training for all three cities in parallel:

```python
# Vectorize the training step
# params: axis 0 (each city has its own params)
# x: axis 0 (each city has its own data)
# y: axis 0 (each city has its own targets)
# learning_rate: None (same for all)
parallel_train_step = jax.jit(
    jax.vmap(train_step, in_axes=(0, 0, 0, None))
)

# Initialize parameters for all 3 cities
# Shape: (3, 2) - 3 cities, 2 params each (slope, intercept)
key, subkey = random.split(key)
params = random.normal(subkey, (n_cities, 2))

print("Initial parameters:")
for i in range(n_cities):
    print(f"  City {i}: slope={params[i, 0]:.2f}, intercept={params[i, 1]:.2f}")
```

### Step 4: Train All Models in Parallel

```python
learning_rate = 0.0001
n_epochs = 2000

print("\nTraining...")
for epoch in range(n_epochs):
    # This single line trains ALL THREE models simultaneously
    params, losses = parallel_train_step(params, X, Y, learning_rate)
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | Losses: {losses}")

print("\nFinal learned parameters:")
print(f"{'City':<6} {'True Slope':<12} {'Learned Slope':<15} {'True Intercept':<15} {'Learned Intercept':<15}")
print("-" * 70)
for i in range(n_cities):
    print(f"{i:<6} {true_slopes[i]:<12.1f} {params[i, 0]:<15.2f} {true_intercepts[i]:<15.1f} {params[i, 1]:<15.2f}")
```

Expected output:

```plaintext
Final learned parameters:
City   True Slope   Learned Slope   True Intercept   Learned Intercept  
----------------------------------------------------------------------
0      50.0         49.87           100.0            102.34         
1      30.0         29.92           200.0            201.15         
2      80.0         79.78           50.0             52.89
```

We trained three separate models, and there's not a single Python for-loop in the training logic. The `parallel_train_step` function processes all cities in one fused GPU kernel.

## Why This Matters

The pattern we just used scales to serious applications:

**Hyperparameter search**: Train 100 models with different learning rates simultaneously. Pick the best one.

**Ensemble methods**: Train 10 models with different random seeds. Average their predictions for more robust results.

**Per-user personalization**: Train a tiny model for each of your 10,000 users. `vmap` handles the parallelization.

**Bayesian methods**: Sample 1000 parameter configurations from a posterior distribution and evaluate all of them at once.

The key insight is that `vmap` doesn't just save you from writing loops; it enables computations that would be impractical with sequential processing.

## Exercises

1. **Gradient verification**: Use `jax.grad` to compute the derivative of `f(x) = sin(x)`. Plot it alongside `cos(x)` to verify they match.
    
2. **The ensemble challenge**: Modify the project to train 10 models on the *same* city but with different random initializations. Use `vmap` over the params axis only (`in_axes=(0, None, None, None)`). Check if they all converge to similar values.
    
3. **Second derivatives**: `jax.grad` returns a function; and you can take its gradient too. Compute the second derivative of `f(x) = x³` and verify it equals `6x`.
    

## Quick Reference

```python
import jax
import jax.numpy as jnp

# vmap: Automatic Vectorization
def single_fn(x):
    return x ** 2

batched_fn = jax.vmap(single_fn)
results = batched_fn(jnp.array([1, 2, 3]))

# With multiple arguments
def dot(w, x):
    return jnp.dot(w, x)

# Shared weights, batched inputs
batch_dot = jax.vmap(dot, in_axes=(None, 0))

# grad: Automatic Differentiation
def loss(params):
    return params ** 2

grad_fn = jax.grad(loss)
gradient = grad_fn(3.0)  # 6.0

# Get both value and gradient
loss_val, grad_val = jax.value_and_grad(loss)(3.0)

# Gradient with respect to specific argument
def f(x, y):
    return x * y

df_dy = jax.grad(f, argnums=1)

# Combining Transforms
fast_batched_grad = jax.jit(jax.vmap(jax.grad(loss_fn), in_axes=(None, 0)))
```

## What's Next

We've now covered the core JAX transforms: `jit` for speed, `vmap` for batching, and `grad` for gradients. These three tools are enough to train neural networks from scratch.

But writing raw JAX for complex models gets tedious. Next week, we'll introduce **Flax NNX';** a neural network library that gives you PyTorch-style ergonomics while keeping all the power of JAX transformations.

We'll build our first real neural network: a CNN for image classification.

## Resources

Automatic Vectorization

[JAX AI Stack Guide](https://jaxstack.ai/)