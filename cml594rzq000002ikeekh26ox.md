---
title: "Why JAX? The NumPy You Know, But Faster"
seoTitle: "Why JAX? The NumPy You Know, But Faster"
seoDescription: "How JAX uses XLA to strap a rocket engine onto your math operations."
datePublished: Mon Feb 02 2026 14:15:09 GMT+0000 (Coordinated Universal Time)
cuid: cml594rzq000002ikeekh26ox
slug: why-jax-the-numpy-you-know-but-faster
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1770040892040/a8adacb0-31ca-4dd9-bc27-d29584712241.png
tags: numpy, jax, xla

---

If you've been doing machine learning in Python for any length of time, you've written code like this:

```python
import numpy as np

x = np.random.randn(1000, 1000)
y = np.random.randn(1000, 1000)
result = np.dot(x, y)
```

NumPy is comfortable. It's the first thing we reach for when we need to do math. But here's the uncomfortable truth: for the kind of work we're about to do: training neural networks, processing millions of samples, running on GPUs and TPUs, NumPy is holding us back.

Not because NumPy is bad. It's genuinely excellent at what it was designed for. The problem is that NumPy was designed in 2005, before GPUs became the workhorses of machine learning, before TPUs existed, and before we needed to compute gradients of functions with millions of parameters.

JAX is what NumPy would look like if we designed it today, knowing what we know now.

By the end of this article, we'll have:

1. Understood *why* JAX is faster (not just that it is)
    
2. Written our first JAX code and seen the speedup ourselves
    
3. Learned the one mental shift that trips up everyone coming from NumPy or PyTorch
    
4. Built a working benchmark that proves the difference
    

Let's get into it.

## The Problem with "Normal" Python

When you run Python code, the interpreter reads your instructions one line at a time, translates each one to machine code, executes it, then moves to the next line. This is called **interpretation**.

```plaintext
Line 1 → Translate → Execute
Line 2 → Translate → Execute
Line 3 → Translate → Execute
...
```

For a script that processes a CSV file or serves a web page, this is fine. The overhead of interpretation is negligible compared to the actual work being done.

But matrix multiplication? That's different. When we multiply two 5000×5000 matrices, we're doing 125 billion floating-point operations. The "translate → execute" overhead for each operation adds up fast.

NumPy helps by pushing the heavy lifting into compiled C code. When you call [`np.dot`](http://np.dot)`()`, Python hands off the work to optimized BLAS libraries that run at near-hardware speed. That's why NumPy is fast*er* than pure Python.

But there's still a problem: **Python is still orchestrating the operations**. Every time you chain NumPy calls together: [`np.dot`](http://np.dot)`()`, then `np.sum()`, then `np.exp()`, Python has to:

1. Call into C
    
2. Wait for the result
    
3. Copy the result back to Python
    
4. Call into C again for the next operation
    

Each of those handoffs has overhead. And when you're doing this millions of times in a training loop, it adds up.

## How JAX Fixes This: XLA Compilation

JAX takes a different approach. Instead of executing operations one at a time, JAX can **compile your entire function** into a single optimized program using XLA (Accelerated Linear Algebra).

Here's what that means in practice:

```plaintext
Read entire function → Analyze → Optimize → Fuse operations → Execute once
```

When JAX compiles a function, it:

* **Fuses operations**: Instead of computing `a + b`, storing the result, then computing `result * c`, XLA fuses these into a single kernel that does `(a + b) * c` without intermediate storage.
    
* **Eliminates dead code**: If you compute something but never use it, XLA removes it entirely.
    
* **Optimizes memory access**: XLA reorders operations to minimize cache misses and memory transfers.
    
* **Targets your hardware**: The same JAX code compiles to optimized instructions for CPU, GPU, or TPU without you changing anything.
    

The result? Code that runs 10x, 100x, sometimes 1000x faster than the NumPy equivalent.

## Setting Up

Let's stop talking and start coding. We'll use Google Colab for this, it's free and gives us access to GPUs.

```bash
# Install the JAX AI stack (includes JAX, Flax, Optax, and friends)
!pip install -q jax-ai-stack
```

Now let's verify our setup:

```python
import jax
import jax.numpy as jnp
import numpy as np
import time

print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")
```

If you're on a GPU runtime, you should see something like:

```plaintext
JAX version: 0.8.0
Available devices: [CudaDevice(id=0)]
```

If you see `CpuDevice`, that's fine too; JAX still provides speedups on CPU through XLA compilation.

## The First Mental Shift: Immutability

Before we benchmark anything, we need to talk about the one thing that trips up *everyone* coming from NumPy or PyTorch.

**In JAX, arrays are immutable. You cannot modify them in place.**

In NumPy, this is perfectly normal:

```python
# NumPy: Mutable arrays
arr = np.zeros(5)
arr[0] = 42  # Modify in place
print(arr)   # [42. 0. 0. 0. 0.]
```

In JAX, this will raise an error:

```python
# JAX: This will FAIL
arr = jnp.zeros(5)
arr[0] = 42  #TypeError: JAX arrays are immutable
```

Why? Because immutability is what makes JAX's optimizations possible. If arrays can be modified from anywhere in your code, the compiler can't safely reorder operations or run them in parallel. By guaranteeing that arrays never change, JAX can aggressively optimize your code.

So how do we update arrays? We use the `.at[].set()` syntax, which **returns a new array** with the modification:

```python
# JAX: The correct way
arr = jnp.zeros(5)
new_arr = arr.at[0].set(42)

print(arr)      # [0. 0. 0. 0. 0.] — Original unchanged
print(new_arr)  # [42. 0. 0. 0. 0.] — New array with the update
```

This feels wasteful at first, are we really copying the entire array just to change one element? In practice, no. JAX and XLA are smart enough to optimize this. But conceptually, you should think of it as creating a new array.

Here's the full set of `.at[]` operations:

```python
x = jnp.array([1, 2, 3, 4, 5])

# Set a value
x.at[0].set(10)         # [10, 2, 3, 4, 5]

# Add to a value
x.at[0].add(10)         # [11, 2, 3, 4, 5]

# Multiply a value
x.at[0].multiply(10)    # [10, 2, 3, 4, 5]

# Works with slices too
x.at[1:3].set(99)       # [1, 99, 99, 4, 5]
```

Commit this to memory. You'll use it constantly.

## The Benchmark: NumPy vs JAX vs JAX+JIT

Now let's prove that JAX is actually faster. We'll multiply two large matrices; this is the core operation in neural networks (every linear layer is a matrix multiplication).

### Step 1: Create the Data

```python
# Matrix size
size = 3000

# Create random matrices with NumPy
x_np = np.random.normal(size=(size, size)).astype(np.float32)
y_np = np.random.normal(size=(size, size)).astype(np.float32)

# Convert to JAX arrays
x_jax = jnp.array(x_np)
y_jax = jnp.array(y_np)

print(f"Matrix shape: {x_np.shape}")
print(f"Total elements per matrix: {size * size:,}")
print(f"Operations for multiplication: {size ** 3:,}")
```

Output:

```plaintext
Matrix shape: (3000, 3000)
Total elements per matrix: 9,000,000
Operations for multiplication: 27,000,000,000
```

That's 27 billion operations. Let's see who can do it fastest.

### Step 2: Benchmark NumPy

```python
def matmul_numpy(x, y):
    return np.dot(x, y)

# Warmup
_ = matmul_numpy(x_np, y_np)

# Timed run
start = time.perf_counter()
result_np = matmul_numpy(x_np, y_np)
numpy_time = time.perf_counter() - start

print(f"NumPy time: {numpy_time:.4f} seconds")
```

Result:

```plaintext
NumPy time: 0.6294 seconds
```

### Step 3: Benchmark JAX (without JIT)

```python
def matmul_jax(x, y):
    return jnp.dot(x, y)

# Warmup
_ = matmul_jax(x_jax, y_jax).block_until_ready()

# Timed run
start = time.perf_counter()
result_jax = matmul_jax(x_jax, y_jax).block_until_ready()
jax_time = time.perf_counter() - start

print(f"JAX time (no JIT): {jax_time:.4f} seconds")
```

**Important**: We call `.block_until_ready()` because JAX operations are **asynchronous**. When you call [`jnp.dot`](http://jnp.dot)`()`, JAX immediately returns a "future" and continues executing Python code while the GPU works in the background. Without `block_until_ready()`, we'd be timing how fast JAX can *dispatch* the operation, not how fast it actually *runs*.

Result:

```plaintext
JAX time (no JIT): 0.0094 seconds
```

### Step 4: Benchmark JAX with JIT Compilation

Here's where JAX shows its true power. We add a single decorator:

```python
@jax.jit
def matmul_jax_jit(x, y):
    return jnp.dot(x, y)

# First call: JAX traces and compiles the function
# This takes a moment, so we don't include it in the benchmark
print("Compiling...")
_ = matmul_jax_jit(x_jax, y_jax).block_until_ready()
print("Done.")

# Timed run (using the compiled version)
start = time.perf_counter()
result_jit = matmul_jax_jit(x_jax, y_jax).block_until_ready()
jit_time = time.perf_counter() - start

print(f"JAX time (with JIT): {jit_time:.4f} seconds")
```

The first call to a `@jax.jit` function is slow because JAX is **tracing** your function, figuring out what operations it contains, and then **compiling** it with XLA. Subsequent calls use the compiled version and are extremely fast.

### Step 5: Compare Results

```python
print("RESULTS")
print(f"NumPy:          {numpy_time:.4f} seconds")
print(f"JAX (no JIT):   {jax_time:.4f} seconds")
print(f"JAX (with JIT): {jit_time:.4f} seconds")
print(f"\nSpeedup (JIT vs NumPy): {numpy_time / jit_time:.1f}x")
```

Typical results on a Colab GPU:

```plaintext
RESULTS
NumPy:          0.6294 seconds
JAX (no JIT):   0.0094 seconds
JAX (with JIT): 0.0198 seconds

Speedup (JAX vs NumPy):     67.2x
Speedup (JIT vs NumPy):     31.8x
Speedup (JIT vs JAX):       0.5x
```

## What Just Happened?

Let's break down why the JIT version is so much faster:

1. **Hardware acceleration**: JAX moved the computation to the GPU, which has thousands of cores optimized for parallel math.
    
2. **XLA compilation**: Even on GPU, the JIT version is faster than raw JAX because XLA fuses operations and optimizes memory access patterns.
    
3. **No Python overhead**: Once compiled, the function runs entirely in native code. Python is only involved in dispatching the call.
    

The key insight is that `@jax.jit` doesn't just run your code on a GPU; it fundamentally changes *how* your code runs.

## The Randomness Trap (Bonus Lesson)

There's one more gotcha that catches everyone early on. Try this:

```python
@jax.jit
def broken_random():
    return np.random.randn(5)  # Using NumPy's random

result1 = broken_random()
result2 = broken_random()
print(f"First call:  {result1}")
print(f"Second call: {result2}")
```

You'll notice that `result1` and `result2` are **identical**. The random numbers got "baked in" during compilation.

JAX requires **explicit random state** management. Here's the correct way:

```python
from jax import random

@jax.jit
def correct_random(key):
    return random.normal(key, shape=(5,))

# Create a PRNG key
key = random.PRNGKey(42)

# Split the key for each use
key, subkey1 = random.split(key)
result1 = correct_random(subkey1)

key, subkey2 = random.split(key)
result2 = correct_random(subkey2)

print(f"First call:  {result1}")
print(f"Second call: {result2}")
```

Now you get different random numbers each time. We'll cover this pattern in depth when we build neural networks, but for now, just remember: **never use** `np.random` inside JIT-compiled functions.

## Exercises

Before moving on, try these:

1. **Break the rules**: Try to modify a JAX array in place (`x[0] = 1`). Read the error message carefully: JAX errors are verbose but informative.
    
2. **Vary the size**: Run the benchmark with different matrix sizes (1000, 2000, 5000). How does the speedup change?
    
3. **Chain operations**: Write a function that does multiple operations ([`jnp.dot`](http://jnp.dot), then `jnp.sum`, then `jnp.exp`). Compare JIT vs non-JIT. The speedup should be even larger because XLA fuses the operations.
    

## What's Next

We've established *why* JAX is fast and seen the proof. But speed is only half the story. Next week, we'll explore **transformations**; the features that make JAX genuinely different from NumPy, not just faster.

Specifically, we'll cover:

* `jax.vmap`: Automatic vectorization that eliminates for-loops
    
* `jax.grad`: Automatic differentiation that makes backpropagation trivial
    

These two functions are why JAX has become the framework of choice for machine learning research. Once you understand them, you'll never look at NumPy the same way again.

## Quick Reference

```python
import jax
import jax.numpy as jnp
from jax import random

# Basic array operations (same as NumPy)
x = jnp.array([1, 2, 3])
y = jnp.zeros((3, 3))
z = jnp.dot(a, b)

# Updating arrays (immutable style)
new_x = x.at[0].set(99)
new_x = x.at[1:].add(10)

# JIT compilation
@jax.jit
def fast_function(x):
    return jnp.dot(x, x.T)

# Explicit randomness
key = random.PRNGKey(0)
key, subkey = random.split(key)
samples = random.normal(subkey, shape=(100,))

# Block for accurate timing
result = fast_function(x).block_until_ready()
```

**Next week**: *Transformations That Change Everything—Automatic Vectorization with vmap and Gradients with grad*

### **Resources:**

[JAX Documentation](https://jax.readthedocs.io)

[JAX AI Stack](https://jaxstack.ai)

[Notebook](https://colab.research.google.com/drive/1f-qg4vlfBHSQSdEdhfgDdRA4k1dPtAqW?usp=sharing)