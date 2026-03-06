# Python for AI: Deep Dive Curriculum

Moving from Data Engineering to AI Engineering requires a shift from "moving data" to "building intelligent systems". This guide covers the advanced Python concepts you need to master.

## 1. Memory Efficiency & Large Data
*   **Generators & Iterators**: Processing multi-gigabyte datasets without crashing RAM.
*   **`itertools` & `functools`**: High-performance data manipulation.
*   **Memory Profiling**: Using `memory_profiler` and `objgraph` to find leaks in long-running models.

## 2. Advanced Modular Code
*   **Decorators**: Creating timing wrappers for model inference, logging, and validation.
*   **Context Managers (`with` blocks)**: Managing GPU memory and database connections.
*   **Metaclasses**: Understanding how library like PyTorch use them for class registration.

## 3. Concurrency & Parallelism
*   **`asyncio`**: Building high-throughput AI APIs (FastAPI integration).
*   **Multiprocessing**: Bypassing the GIL for CPU-bound tasks like image augmentation or training.
*   **Threading**: Efficiently handling I/O bound tasks like streaming data from Kafka.

## 4. Robust AI System Design
*   **Type Hinting & Pydantic**: Building self-documenting, type-safe data schemas for AI models.
*   **Advanced OOP**: Singleton (for loading heavy models once), Factory (for swapping model versions).
*   **Metaprogramming**: Introspection and dynamic attribute access.

## 5. Performance Optimization
*   **Vectorization**: Moving beyond loops to NumPy/Tensor operations.
*   **Cython & C-Extensions**: Understanding how Python interfaces with high-performance C++ kernels in AI libraries.
*   **Numba**: JIT (Just-In-Time) compilation for specialized math functions.

## 6. Testing AI Systems
*   **Pytest & Mocking**: Mocking expensive model calls during unit tests.
*   **Property-Based Testing**: Using `hypothesis` to find edge cases in data processing.

---

### 💡 Recommendation
Start with **Generators** and **Concurrency**, as these bridge the gap between Data Engineering pipelines and AI model serving.

---

## 🎤 Interview Q&A (Top 10 Advanced Python Questions)

### Q1: What are decorators? Write one that times a function.
> **How to answer:** Show practical usage, not just theory.

```python
import time
from functools import wraps

def timer(func):
    @wraps(func)  # Preserves function name and docstring
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timer
def train_model(epochs):
    """Train a model for N epochs."""
    time.sleep(0.5)  # Simulate training
    return "model_trained"

train_model(10)  # "train_model took 0.5001s"

# Decorator with arguments (advanced):
def retry(max_attempts=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {attempt+1} failed: {e}")
            raise Exception(f"Failed after {max_attempts} attempts")
        return wrapper
    return decorator

@retry(max_attempts=3)
def call_api():
    # Simulate flaky API call
    pass
```
**AI use case:** Timing model inference, retrying API calls, logging predictions.

---

### Q2: Explain generators. When would you use one in ML?
> **How to answer:** Focus on memory efficiency with large datasets.

```python
# ❌ List: Loads everything into memory
def load_all_data(filepath):
    with open(filepath) as f:
        return [process(line) for line in f]  # 10GB file = 10GB RAM!

# ✅ Generator: Processes one item at a time
def data_generator(filepath, batch_size=32):
    """Yield batches of data — like PyTorch DataLoader."""
    batch = []
    with open(filepath) as f:
        for line in f:
            batch.append(process(line))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

# Usage: Iterate without loading everything
for batch in data_generator('huge_dataset.csv', batch_size=64):
    model.train(batch)  # Process 64 samples at a time

# Generator expressions (one-liners)
squares = (x**2 for x in range(1_000_000))  # 0 memory until iterated
```
**Key point:** PyTorch's `DataLoader` and TensorFlow's `tf.data` are built on generator patterns.

---

### Q3: What is the GIL? How does it affect ML workloads?
> **How to answer:** Show you understand when it matters and when it doesn't.

```python
# GIL = Global Interpreter Lock
# Only ONE thread can execute Python bytecode at a time

# ❌ GIL hurts: CPU-bound Python code
import threading
# Multiple threads doing math → NO speedup (GIL blocks parallelism)

# ✅ GIL doesn't matter for ML because:
# 1. NumPy/PyTorch release GIL during C operations
# 2. Heavy computation runs in C/CUDA, not Python
# 3. I/O operations (file reading, API calls) release GIL

# Workarounds:
# 1. multiprocessing — separate processes, each has its own GIL
from multiprocessing import Pool
with Pool(4) as p:
    results = p.map(heavy_computation, data_chunks)

# 2. asyncio — for I/O bound (API calls, database queries)
# 3. C extensions — NumPy, PyTorch already do this
```

---

### Q4: What's the difference between `__init__`, `__new__`, and `__call__`?
> **How to answer:** Shows deep OOP understanding — asked in senior roles.

```python
class Model:
    def __new__(cls, *args, **kwargs):
        """Controls object creation (before __init__)."""
        print("Creating instance")
        instance = super().__new__(cls)
        return instance

    def __init__(self, name):
        """Initializes the object (after __new__)."""
        print("Initializing instance")
        self.name = name

    def __call__(self, x):
        """Makes instance callable like a function."""
        print(f"Predicting with {self.name}")
        return x * 2

model = Model("GPT")     # "Creating instance" → "Initializing instance"
result = model(42)         # "Predicting with GPT" → 84

# AI use case: PyTorch nn.Module uses __call__ — that's why you do model(x)
```

---

### Q5: Implement the Singleton pattern for a model loader.
> **How to answer:** Prevents loading a 5GB model multiple times.

```python
class ModelLoader:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, model_path):
        if self._model is None:
            print(f"Loading model from {model_path}...")
            self._model = f"model_at_{model_path}"  # Simulate heavy load
        return self._model

# Usage:
loader1 = ModelLoader()
loader2 = ModelLoader()
print(loader1 is loader2)  # True — same instance!

loader1.load("gpt-4.bin")   # "Loading model..."
loader2.load("gpt-4.bin")   # No print — already loaded!
```

---

### Q6: How does `asyncio` work? Write an async API call handler.
> **How to answer:** Critical for building AI APIs with FastAPI.

```python
import asyncio

async def call_model_api(user_id, query):
    """Simulate async model inference."""
    print(f"Processing {user_id}...")
    await asyncio.sleep(1)  # Simulate model inference time
    return {"user_id": user_id, "response": f"Answer to: {query}"}

async def handle_batch(requests):
    """Process multiple requests concurrently."""
    tasks = [call_model_api(r['user_id'], r['query']) for r in requests]
    results = await asyncio.gather(*tasks)
    return results

# Simulate 5 concurrent API calls
requests = [{"user_id": i, "query": f"Question {i}"} for i in range(5)]
results = asyncio.run(handle_batch(requests))
# All 5 complete in ~1 second (not 5 seconds!)
```
**Key insight:** FastAPI is async by default — every AI endpoint benefits from this pattern.

---

### Q7: What is a context manager? Write one for GPU memory.
> **How to answer:** Beyond just `with open()`.

```python
from contextlib import contextmanager

@contextmanager
def gpu_memory_scope(device_id=0):
    """Allocate GPU memory and ensure cleanup."""
    print(f"Allocating GPU {device_id}")
    # In real code: torch.cuda.set_device(device_id)
    try:
        yield device_id
    finally:
        print(f"Releasing GPU {device_id}")
        # In real code: torch.cuda.empty_cache()

# Usage:
with gpu_memory_scope(0) as gpu:
    print(f"Training on GPU {gpu}")
    # model.train()
# GPU memory automatically freed after this block

# Class-based context manager:
class Timer:
    def __enter__(self):
        import time
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        print(f"Elapsed: {self.elapsed:.4f}s")

with Timer() as t:
    # Heavy computation
    pass
```

---

### Q8: Explain `*args` and `**kwargs`. How are they used in ML frameworks?
> **How to answer:** Show flexibility patterns from real frameworks.

```python
def flexible_train(*args, **kwargs):
    """Accept any combination of arguments."""
    print(f"Positional args: {args}")
    print(f"Keyword args: {kwargs}")

flexible_train(100, 0.01, epochs=50, batch_size=32)

# Real-world: PyTorch forward() uses this pattern
class MyModel:
    def forward(self, x, *args, **kwargs):
        # Handles varying inputs gracefully
        temperature = kwargs.get('temperature', 1.0)
        return x / temperature

# Decorator forwarding all arguments:
def log_calls(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper
```

---

### Q9: What is Pydantic and why is it important for AI APIs?
> **How to answer:** Shows production ML system knowledge.

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional

class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    model_name: str = Field(default="gpt-4")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=100, le=4096)

    @validator('model_name')
    def validate_model(cls, v):
        valid_models = ['gpt-4', 'gpt-3.5', 'claude']
        if v not in valid_models:
            raise ValueError(f"Model must be one of {valid_models}")
        return v

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    tokens_used: int

# FastAPI uses Pydantic for automatic validation:
# @app.post("/predict", response_model=PredictionResponse)
# async def predict(request: PredictionRequest):
#     return PredictionResponse(...)
```
**Why it matters:** Every production AI API uses Pydantic for input validation — prevents bad data from crashing models.

---

### Q10: What's the difference between shallow copy and deep copy?
> **How to answer:** Important when working with nested model configs.

```python
import copy

# Shallow copy: copies outer object, but inner objects are shared
original = {"model": {"layers": [64, 128, 256]}, "lr": 0.01}
shallow = copy.copy(original)
shallow["model"]["layers"].append(512)
print(original["model"]["layers"])  # [64, 128, 256, 512] — MODIFIED!

# Deep copy: copies everything recursively
original = {"model": {"layers": [64, 128, 256]}, "lr": 0.01}
deep = copy.deepcopy(original)
deep["model"]["layers"].append(512)
print(original["model"]["layers"])  # [64, 128, 256] — SAFE!

# When it matters in ML:
# - Copying model configs for hyperparameter experiments
# - Cloning model state dictionaries
# - Duplicating nested data structures for A/B testing
```

---

## 🎯 Mastery Checklist

- [ ] Write custom decorators (timing, retry, logging)
- [ ] Use generators for memory-efficient data loading
- [ ] Understand the GIL and when to use multiprocessing vs threading
- [ ] Implement Singleton pattern for model loading
- [ ] Build async handlers with asyncio for AI APIs
- [ ] Write context managers for resource management
- [ ] Use Pydantic for API input/output validation
- [ ] Answer all 10 interview questions confidently

