# NumPy for AI: Complete Deep Dive

NumPy is the backbone of every AI/ML library. PyTorch tensors, Scikit-learn internals, and even image processing all rely on NumPy-style operations. This guide covers everything you need.

---

## 📌 1. Array Fundamentals

### 1.1 Creating Arrays
```python
import numpy as np

# From lists
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Special arrays
zeros = np.zeros((3, 4))          # 3x4 matrix of zeros
ones = np.ones((2, 3))            # 2x3 matrix of ones
identity = np.eye(4)              # 4x4 identity matrix
random_arr = np.random.randn(3, 3) # 3x3 random normal distribution
linspace = np.linspace(0, 1, 50)  # 50 evenly spaced values from 0 to 1
arange = np.arange(0, 10, 0.5)   # Step-based range
```

### 1.2 Array Properties
```python
arr = np.random.randn(3, 4, 5)

arr.shape      # (3, 4, 5) — dimensions
arr.ndim       # 3 — number of dimensions
arr.size       # 60 — total elements
arr.dtype      # float64 — data type
arr.itemsize   # 8 — bytes per element
arr.nbytes     # 480 — total memory used
```

### 1.3 Data Types (Critical for AI)
```python
# float32 is standard for ML models (saves GPU memory)
arr_f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)

# float16 for mixed-precision training
arr_f16 = arr_f32.astype(np.float16)

# int8 for model quantization
arr_int8 = np.array([1, 2, 3], dtype=np.int8)

# bool for masks
mask = np.array([True, False, True], dtype=np.bool_)
```

---

## 📌 2. Indexing & Slicing (AI-Critical)

### 2.1 Basic Indexing
```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

matrix[0, 1]      # 2 — single element
matrix[1, :]      # [4, 5, 6] — entire row
matrix[:, 2]      # [3, 6, 9] — entire column
matrix[0:2, 1:3]  # [[2, 3], [5, 6]] — sub-matrix
```

### 2.2 Boolean Indexing (Used Everywhere in ML)
```python
data = np.array([10, 25, 30, 5, 15, 40])

# Filter values
data[data > 20]              # [25, 30, 40]
data[(data > 10) & (data < 35)]  # [25, 30, 15]

# Replace values (like handling outliers)
data[data > 30] = 30         # Clip outliers
```

### 2.3 Fancy Indexing
```python
arr = np.array([10, 20, 30, 40, 50])

# Select specific indices
indices = [0, 2, 4]
arr[indices]    # [10, 30, 50]

# Useful for selecting specific features or samples
features = np.random.randn(100, 10)  # 100 samples, 10 features
selected = features[:, [0, 3, 7]]    # Pick features 0, 3, 7
```

---

## 📌 3. Reshaping & Dimension Manipulation

### 3.1 Reshape (Most Used in Deep Learning)
```python
arr = np.arange(12)

# Reshape to matrix
matrix = arr.reshape(3, 4)     # 3 rows, 4 columns
matrix = arr.reshape(3, -1)    # -1 means "auto-calculate" → same as (3, 4)

# Flatten back
flat = matrix.ravel()          # [0, 1, 2, ..., 11]
flat = matrix.flatten()        # Same but returns a copy
```

### 3.2 Adding/Removing Dimensions
```python
arr = np.array([1, 2, 3])     # Shape: (3,)

# Add dimension (critical for model input)
row = arr[np.newaxis, :]       # Shape: (1, 3) — row vector
col = arr[:, np.newaxis]       # Shape: (3, 1) — column vector
expanded = np.expand_dims(arr, axis=0)  # Same as newaxis

# Remove dimension
squeezed = np.squeeze(expanded)  # Back to (3,)
```

### 3.3 Transpose & Swapaxes
```python
matrix = np.random.randn(3, 4)
matrix.T                      # Shape: (4, 3)

# For 3D+ arrays (e.g., image batch: batch x height x width x channels)
images = np.random.randn(32, 224, 224, 3)  # Batch of 32 images
# Convert to channels-first (PyTorch format)
images_chf = np.transpose(images, (0, 3, 1, 2))  # (32, 3, 224, 224)
```

---

## 📌 4. Broadcasting (Core AI Concept)

Broadcasting lets you operate on arrays of different shapes without copying data.

### 4.1 Rules
```
Rule 1: If arrays differ in ndim, prepend 1s to the smaller shape.
Rule 2: Arrays with size 1 along a dimension act as if they had the size of the larger array.
Rule 3: If shapes don't match and neither is 1, error.
```

### 4.2 Examples
```python
# Scalar broadcast
arr = np.array([1, 2, 3])
arr * 2                       # [2, 4, 6]

# Vector + Matrix
matrix = np.ones((3, 4))
row = np.array([1, 2, 3, 4])
matrix + row                  # Each row gets [1,2,3,4] added

# Normalizing features (subtract mean per column)
data = np.random.randn(100, 5)
mean = data.mean(axis=0)      # Shape: (5,)
normalized = data - mean      # Broadcasting: (100, 5) - (5,) → (100, 5)

# Standardization (Z-score)
std = data.std(axis=0)
z_scores = (data - mean) / std
```

---

## 📌 5. Linear Algebra (Foundation for ML)

### 5.1 Matrix Operations
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Element-wise
A * B                         # [[5, 12], [21, 32]]

# Matrix multiplication (THE most important operation in ML)
A @ B                         # [[19, 22], [43, 50]]
np.dot(A, B)                  # Same
np.matmul(A, B)               # Same

# Transpose
A.T                           # [[1, 3], [2, 4]]
```

### 5.2 Key Linear Algebra Operations
```python
# Determinant
np.linalg.det(A)              # -2.0

# Inverse (used in linear regression: (X^T X)^-1 X^T y)
A_inv = np.linalg.inv(A)

# Eigenvalues & Eigenvectors (used in PCA)
eigenvalues, eigenvectors = np.linalg.eig(A)

# Singular Value Decomposition (used in recommendation systems)
U, S, Vt = np.linalg.svd(A)

# Solving linear equations: Ax = b
b = np.array([1, 2])
x = np.linalg.solve(A, b)

# Norms (used in regularization)
np.linalg.norm(A)             # Frobenius norm
np.linalg.norm(A, ord=1)      # L1 norm
np.linalg.norm(A, ord=2)      # L2 norm (spectral)
```

---

## 📌 6. Statistical Operations (Data Analysis for ML)

```python
data = np.random.randn(1000, 5)

# Descriptive stats
data.mean(axis=0)       # Mean per feature
data.std(axis=0)        # Standard deviation per feature
data.var(axis=0)        # Variance per feature
np.median(data, axis=0) # Median per feature

# Percentiles (for outlier detection)
np.percentile(data, [25, 50, 75], axis=0)  # Q1, Q2, Q3

# Correlation matrix (feature relationships)
correlation = np.corrcoef(data.T)  # 5x5 correlation matrix

# Cumulative operations
np.cumsum(data[:, 0])   # Running total
np.cumprod(data[:, 0])  # Running product

# Argmax / Argmin (used for predictions)
predictions = np.array([0.1, 0.7, 0.2])
predicted_class = np.argmax(predictions)  # 1
```

---

## 📌 7. Random Number Generation (Critical for ML)

```python
rng = np.random.default_rng(seed=42)  # Modern API with seed

# Distributions commonly used in AI
rng.normal(0, 1, size=(3, 3))        # Gaussian — weight initialization
rng.uniform(0, 1, size=(3, 3))       # Uniform — random sampling
rng.integers(0, 10, size=(5,))       # Integer — index sampling
rng.choice([1, 2, 3, 4], size=2, replace=False)  # Without replacement

# Shuffling (for training data)
indices = np.arange(1000)
rng.shuffle(indices)
train_idx = indices[:800]
test_idx = indices[800:]

# Setting seeds for reproducibility
np.random.seed(42)  # Legacy but widely used
```

---

## 📌 8. Performance & Vectorization

### 8.1 Why Vectorization Matters
```python
import time

size = 1_000_000
a = np.random.randn(size)
b = np.random.randn(size)

# ❌ Slow: Python loop
start = time.time()
result = [a[i] * b[i] for i in range(size)]
print(f"Loop: {time.time() - start:.4f}s")

# ✅ Fast: Vectorized
start = time.time()
result = a * b
print(f"Vectorized: {time.time() - start:.4f}s")
# Typically 50-100x faster!
```

### 8.2 Common Vectorized Patterns in AI
```python
# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Softmax function (output layer of classification models)
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

# ReLU activation
def relu(x):
    return np.maximum(0, x)

# Euclidean distance between all pairs (KNN, clustering)
def pairwise_distance(X):
    sq = np.sum(X**2, axis=1, keepdims=True)
    return np.sqrt(sq + sq.T - 2 * X @ X.T)

# Cosine similarity (NLP embeddings)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

---

## 📌 9. Memory Management

```python
# Views vs Copies
arr = np.array([1, 2, 3, 4, 5])
view = arr[1:4]        # View — shares memory, changes affect original
copy = arr[1:4].copy() # Copy — independent memory

# Check if it's a view
view.base is arr       # True
copy.base is arr       # False

# Memory-efficient operations
# Use out= parameter to avoid creating new arrays
result = np.empty_like(arr)
np.add(arr, arr, out=result)  # In-place addition

# Use np.memmap for arrays that don't fit in RAM
large_data = np.memmap('data.bin', dtype='float32', mode='w+', shape=(10000, 1000))
```

---

## 📌 10. Real-World AI Exercises

### Exercise 1: Implement Linear Regression from Scratch
```python
# Generate data
X = np.random.randn(100, 3)
true_weights = np.array([2, -1, 0.5])
y = X @ true_weights + np.random.randn(100) * 0.1

# Normal equation: w = (X^T X)^-1 X^T y
w = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"Learned weights: {w}")  # Should be close to [2, -1, 0.5]
```

### Exercise 2: Implement K-Means Clustering
```python
def kmeans(X, k, max_iters=100):
    centroids = X[np.random.choice(len(X), k, replace=False)]
    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids
```

### Exercise 3: Image Processing Basics
```python
# Simulate a grayscale image
image = np.random.randint(0, 256, (28, 28), dtype=np.uint8)

# Normalize to [0, 1] — standard preprocessing
normalized = image / 255.0

# Flatten for ML model input
flat = normalized.reshape(1, -1)  # Shape: (1, 784)

# Add batch dimension for CNN
batched = normalized[np.newaxis, np.newaxis, :, :]  # (1, 1, 28, 28)
```

---

## 🎤 Interview Q&A (Top 10 NumPy Questions)

### Q1: What is the difference between `np.dot()`, `@`, and `*` in NumPy?
> **How to answer:** Show you understand element-wise vs matrix multiplication.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

A * B       # Element-wise: [[5, 12], [21, 32]]
A @ B       # Matrix multiply: [[19, 22], [43, 50]]
np.dot(A,B) # Same as @ for 2D arrays

# Key difference: For 1D arrays, np.dot gives inner product (scalar)
# For 2D+, np.dot and @ both do matrix multiplication
```
**Why it matters:** Matrix multiplication is the core operation in neural networks (every layer is a matrix multiply).

---

### Q2: Explain broadcasting. Can you give an example where it fails?
> **How to answer:** State the 3 rules, give a working and failing example.

```python
# ✅ Works: (3, 4) + (4,) → row added to each row
np.ones((3, 4)) + np.array([1, 2, 3, 4])

# ✅ Works: (3, 1) + (1, 4) → expands both
np.ones((3, 1)) + np.ones((1, 4))  # Result: (3, 4)

# ❌ Fails: (3,) + (4,) → shapes don't align
np.array([1, 2, 3]) + np.array([1, 2, 3, 4])  # ValueError!
```
**Interviewer expects:** You mention broadcasting is how NumPy avoids loops and memory copies for element-wise operations on mismatched shapes.

---

### Q3: What is the difference between `reshape()`, `ravel()`, and `flatten()`?
> **How to answer:** Focus on memory behavior (view vs copy).

```python
arr = np.array([[1, 2], [3, 4]])

reshaped = arr.reshape(4)     # View — shares memory with original
raveled = arr.ravel()         # View — shares memory (usually)
flattened = arr.flatten()     # Copy — independent memory

# Proof:
raveled[0] = 99
print(arr[0, 0])  # 99 — original changed!
```
**Key insight:** In production ML, prefer `ravel()` over `flatten()` to save memory when you don't need a copy.

---

### Q4: Implement softmax from scratch. Why do we subtract the max?
> **How to answer:** This is asked in almost every ML coding interview.

```python
def softmax(x):
    # Subtract max for numerical stability (prevents overflow in exp)
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

# Without subtracting max:
x = np.array([1000, 1001, 1002])
# np.exp(1000) = inf → NaN results!

# With subtracting max:
# np.exp([0, 1, 2]) → works perfectly
```
**Why it matters:** Shows you understand numerical stability — critical for training deep learning models.

---

### Q5: How would you normalize a dataset using NumPy?
> **How to answer:** Show both Z-score and Min-Max normalization.

```python
data = np.random.randn(100, 5)  # 100 samples, 5 features

# Z-Score Normalization (mean=0, std=1)
z_normalized = (data - data.mean(axis=0)) / data.std(axis=0)

# Min-Max Normalization (scale to [0, 1])
min_max = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

# Verify
print(z_normalized.mean(axis=0))  # ~[0, 0, 0, 0, 0]
print(z_normalized.std(axis=0))   # ~[1, 1, 1, 1, 1]
```
**When to use which:** Z-score for Gaussian data (linear regression, SVM). Min-Max for bounded ranges (images, neural networks).

---

### Q6: What is the difference between a view and a copy? Why does it matter?
> **How to answer:** Explain memory implications for large ML datasets.

```python
large_data = np.random.randn(1_000_000, 100)  # ~800 MB

# View: 0 extra memory
subset_view = large_data[:1000]  # Still points to same memory
subset_view.base is large_data   # True

# Copy: doubles memory usage
subset_copy = large_data[:1000].copy()  # Creates new 800 KB array
subset_copy.base is large_data  # False

# Danger: Modifying a view modifies the original!
subset_view[0, 0] = 999
print(large_data[0, 0])  # 999!
```
**Production tip:** When preprocessing training data, use views for read-only operations and copies when you need to modify data independently.

---

### Q7: How would you implement cosine similarity between two vectors?
> **How to answer:** Common in NLP/embedding interviews.

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Batch cosine similarity (all pairs)
def batch_cosine_similarity(X):
    """Compute cosine similarity matrix for all row pairs."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    normalized = X / norms
    return normalized @ normalized.T

# Example: Compare word embeddings
embedding_cat = np.array([0.2, 0.8, 0.1])
embedding_dog = np.array([0.3, 0.7, 0.2])
embedding_car = np.array([0.9, 0.1, 0.8])

print(cosine_similarity(embedding_cat, embedding_dog))  # High (~0.98)
print(cosine_similarity(embedding_cat, embedding_car))  # Low (~0.40)
```

---

### Q8: Why is float32 preferred over float64 in deep learning?
> **How to answer:** Show practical understanding of GPU memory constraints.

```python
data_64 = np.random.randn(10000, 1000).astype(np.float64)
data_32 = data_64.astype(np.float32)

print(f"float64: {data_64.nbytes / 1e6:.1f} MB")  # 80.0 MB
print(f"float32: {data_32.nbytes / 1e6:.1f} MB")   # 40.0 MB — 50% less!

# In deep learning:
# - GPU memory is limited (8-80 GB)
# - float32 gives sufficient precision for gradients
# - float16 (mixed precision) saves even more → 2x faster training
# - int8 used for model quantization (inference optimization)
```

---

### Q9: How would you find the top-K predictions from a model output?
> **How to answer:** Shows practical numpy usage in inference.

```python
# Model outputs probabilities for 1000 classes
predictions = np.random.rand(1000)

# Method 1: Sort (O(n log n))
top_k_indices = np.argsort(predictions)[-5:][::-1]

# Method 2: argpartition (O(n)) — faster for large arrays!
top_k_indices = np.argpartition(predictions, -5)[-5:]
top_k_indices = top_k_indices[np.argsort(predictions[top_k_indices])[::-1]]

top_k_values = predictions[top_k_indices]
print(f"Top 5 classes: {top_k_indices}")
print(f"Top 5 scores: {top_k_values}")
```
**Key insight:** `argpartition` is O(n) vs `argsort`'s O(n log n) — matters when scoring millions of items (recommendation systems).

---

### Q10: Implement a simple gradient descent using NumPy.
> **How to answer:** The ultimate "can you code ML from scratch" question.

```python
def gradient_descent(X, y, lr=0.01, epochs=1000):
    """Linear regression with gradient descent."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for epoch in range(epochs):
        # Forward pass
        y_pred = X @ weights + bias

        # Compute gradients
        dw = (1/n_samples) * (X.T @ (y_pred - y))
        db = (1/n_samples) * np.sum(y_pred - y)

        # Update parameters
        weights -= lr * dw
        bias -= lr * db

        # Log loss every 100 epochs
        if epoch % 100 == 0:
            loss = np.mean((y_pred - y) ** 2)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return weights, bias

# Test
X = np.random.randn(200, 3)
true_w = np.array([2.0, -1.0, 0.5])
y = X @ true_w + 3.0 + np.random.randn(200) * 0.1

w, b = gradient_descent(X, y)
print(f"Learned: w={w}, b={b:.2f}")  # Close to [2, -1, 0.5] and b≈3
```

---

## 🎯 Mastery Checklist

- [ ] Create and manipulate multi-dimensional arrays
- [ ] Use boolean indexing for data filtering
- [ ] Reshape arrays for model input/output
- [ ] Understand and apply broadcasting
- [ ] Perform matrix multiplication and linear algebra
- [ ] Calculate statistics for feature analysis
- [ ] Write vectorized code (no Python loops)
- [ ] Manage memory with views vs copies
- [ ] Implement ML algorithms from scratch using NumPy
- [ ] Answer all 10 interview questions confidently
