# Data Structures & Algorithms for AI Engineers

As an AI Engineer, DSA isn't about competitive programming — it's about **writing efficient ML systems**, **optimizing data pipelines**, and **acing technical interviews**. This guide focuses on the DSA topics that matter most for AI roles.

---

## 📌 1. Arrays & Lists (Foundation)

### 1.1 Key Concepts
```python
# Dynamic arrays (Python list)
arr = [1, 2, 3, 4, 5]

# Time Complexities:
# Access:    O(1)   — arr[i]
# Search:    O(n)   — linear scan
# Insert:    O(n)   — shift elements
# Append:    O(1)   — amortized
# Delete:    O(n)   — shift elements
```

### 1.2 Two Pointer Technique (Very Common in Interviews)
```python
def two_sum_sorted(arr, target):
    """Find two numbers that sum to target in sorted array."""
    left, right = 0, len(arr) - 1
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return [-1, -1]
```

### 1.3 Sliding Window (Used in Time-Series & NLP)
```python
def max_subarray_sum(arr, k):
    """Maximum sum of subarray of size k — like a rolling window."""
    window_sum = sum(arr[:k])
    max_sum = window_sum
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)
    return max_sum

# AI Application: Computing rolling features for time-series models
# Similar logic is used in 1D convolution in CNNs
```

---

## 📌 2. Hash Maps & Sets (Most Used in AI)

### 2.1 Dictionary Patterns
```python
# Counting (like word frequency in NLP)
from collections import Counter

text = "the cat sat on the mat the cat"
word_freq = Counter(text.split())
# Counter({'the': 3, 'cat': 2, 'sat': 1, 'on': 1, 'mat': 1})

# Top-k frequent elements
top_k = word_freq.most_common(3)

# GroupBy pattern (manual)
from collections import defaultdict
data = [('IT', 'Alice'), ('HR', 'Bob'), ('IT', 'Charlie'), ('HR', 'Diana')]
groups = defaultdict(list)
for dept, name in data:
    groups[dept].append(name)
# {'IT': ['Alice', 'Charlie'], 'HR': ['Bob', 'Diana']}
```

### 2.2 AI-Specific Applications
```python
# Vocabulary mapping (NLP tokenization)
vocab = {}
for idx, word in enumerate(set(text.split())):
    vocab[word] = idx
# {'the': 0, 'cat': 1, 'sat': 2, ...}

# Inverse vocabulary
inv_vocab = {v: k for k, v in vocab.items()}

# Feature hashing (for high-cardinality features)
def feature_hash(feature, num_buckets=1000):
    return hash(feature) % num_buckets

# Caching predictions (memoization)
from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive_prediction(input_tuple):
    # Simulate model inference
    return sum(input_tuple) / len(input_tuple)
```

---

## 📌 3. Stacks & Queues

### 3.1 Stack (LIFO)
```python
# Stack — used in expression parsing, backtracking
stack = []
stack.append(1)   # Push
stack.append(2)
top = stack.pop()  # Pop → 2

# AI Application: Parsing nested model configurations (JSON/YAML)
def is_valid_brackets(s):
    """Validate nested brackets — like checking model config syntax."""
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    for char in s:
        if char in '([{':
            stack.append(char)
        elif char in ')]}':
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()
    return len(stack) == 0
```

### 3.2 Queue & Deque
```python
from collections import deque

# Queue — used in BFS, message processing (like your Kafka experience)
queue = deque()
queue.append(1)      # Enqueue (right)
queue.popleft()      # Dequeue (left) — O(1)

# Priority Queue — used in beam search (text generation), A* search
import heapq

min_heap = []
heapq.heappush(min_heap, (0.9, 'prediction_A'))  # (priority, value)
heapq.heappush(min_heap, (0.3, 'prediction_B'))
heapq.heappush(min_heap, (0.7, 'prediction_C'))
best = heapq.heappop(min_heap)  # (0.3, 'prediction_B') — lowest first

# Top-K predictions (used in model output)
scores = [0.1, 0.9, 0.3, 0.7, 0.5, 0.8]
top_3 = heapq.nlargest(3, enumerate(scores), key=lambda x: x[1])
# [(1, 0.9), (5, 0.8), (3, 0.7)]
```

---

## 📌 4. Trees (Critical for ML)

### 4.1 Binary Tree Basics
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Traversals (understand these — Decision Trees use them)
def inorder(node):
    """Left → Root → Right (sorted for BST)."""
    if not node:
        return []
    return inorder(node.left) + [node.val] + inorder(node.right)

def preorder(node):
    """Root → Left → Right (serialize tree)."""
    if not node:
        return []
    return [node.val] + preorder(node.left) + preorder(node.right)

def bfs_level_order(root):
    """Level-by-level (used in BFS search)."""
    if not root:
        return []
    result, queue = [], deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(level)
    return result
```

### 4.2 Why Trees Matter in AI
```python
# Decision Trees — the foundation of XGBoost, Random Forest, LightGBM
# These are the most powerful models for tabular data

# Key concepts to understand:
# 1. Information Gain / Gini Impurity — how splits are chosen
# 2. Tree depth — controls overfitting
# 3. Pruning — removing unnecessary branches

# Trie — used for autocomplete, prefix search in NLP
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

---

## 📌 5. Graphs (AI System Design)

### 5.1 Graph Representations
```python
# Adjacency List (most common)
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'D'],
    'D': ['B', 'C']
}

# Adjacency Matrix (used in GNNs — Graph Neural Networks)
import numpy as np
adj_matrix = np.array([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
])
```

### 5.2 BFS & DFS
```python
def bfs(graph, start):
    """Breadth-First Search — used in knowledge graphs, web crawling."""
    visited = set()
    queue = deque([start])
    order = []
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            order.append(node)
            queue.extend(n for n in graph[node] if n not in visited)
    return order

def dfs(graph, start, visited=None):
    """Depth-First Search — used in dependency resolution, topological sort."""
    if visited is None:
        visited = set()
    visited.add(start)
    result = [start]
    for neighbor in graph[start]:
        if neighbor not in visited:
            result.extend(dfs(graph, neighbor, visited))
    return result
```

### 5.3 AI Applications of Graphs
```python
# Topological Sort — DAG scheduling (like Airflow, Spark DAGs)
def topological_sort(graph):
    """Order tasks so dependencies come first — ML pipeline scheduling."""
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    queue = deque([n for n in in_degree if in_degree[n] == 0])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return order

# Example: ML Pipeline DAG
pipeline = {
    'data_ingestion': ['data_cleaning'],
    'data_cleaning': ['feature_engineering'],
    'feature_engineering': ['model_training', 'validation_split'],
    'validation_split': ['model_evaluation'],
    'model_training': ['model_evaluation'],
    'model_evaluation': ['model_deployment'],
    'model_deployment': []
}
print(topological_sort(pipeline))
# ['data_ingestion', 'data_cleaning', 'feature_engineering', ...]
```

---

## 📌 6. Sorting & Searching

### 6.1 Sorting Algorithms to Know
```python
# Python's built-in: Timsort — O(n log n)
arr = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_arr = sorted(arr)               # Returns new list
arr.sort()                              # In-place

# Custom sorting (used for ranking model predictions)
predictions = [
    {'label': 'cat', 'confidence': 0.9},
    {'label': 'dog', 'confidence': 0.7},
    {'label': 'bird', 'confidence': 0.85}
]
ranked = sorted(predictions, key=lambda x: x['confidence'], reverse=True)

# Merge Sort — O(n log n) — basis for external sorting (large datasets)
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Quick Select — O(n) average — find k-th smallest without full sort
# Used in: Top-K predictions, finding median efficiently
```

### 6.2 Binary Search
```python
def binary_search(arr, target):
    """O(log n) search — used for threshold tuning in ML."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# AI Application: Finding optimal threshold for classification
def find_optimal_threshold(thresholds, scores, target_precision):
    """Binary search for threshold that gives target precision."""
    left, right = 0, len(thresholds) - 1
    best = -1
    while left <= right:
        mid = (left + right) // 2
        precision = calculate_precision(scores, thresholds[mid])
        if precision >= target_precision:
            best = mid
            left = mid + 1   # Try higher threshold
        else:
            right = mid - 1
    return thresholds[best] if best != -1 else None
```

---

## 📌 7. Dynamic Programming (Interview Essential)

### 7.1 Core Patterns
```python
# Pattern 1: Fibonacci (Memoization)
def fib(n, memo={}):
    if n <= 1: return n
    if n not in memo:
        memo[n] = fib(n-1) + fib(n-2)
    return memo[n]

# Pattern 2: Longest Common Subsequence (used in diff tools, DNA sequencing)
def lcs(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

# Pattern 3: Knapsack (Resource allocation — GPU memory, batch sizing)
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i-1][w]
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w-weights[i-1]] + values[i-1])
    return dp[n][capacity]
```

### 7.2 AI Applications of DP
```python
# Viterbi Algorithm — used in HMM (Hidden Markov Models) for sequence labeling
# Edit Distance — used in spell correction, NLP
def edit_distance(word1, word2):
    """Minimum edits to transform word1 → word2 (NLP spell check)."""
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

# Beam Search — used in text generation (GPT, translation)
# This is essentially a BFS with limited width (top-K at each step)
```

---

## 📌 8. Complexity Analysis Quick Reference

| Operation | Array | Hash Map | BST | Heap |
| --- | --- | --- | --- | --- |
| Access | O(1) | O(1) avg | O(log n) | O(1) min/max |
| Search | O(n) | O(1) avg | O(log n) | O(n) |
| Insert | O(n) | O(1) avg | O(log n) | O(log n) |
| Delete | O(n) | O(1) avg | O(log n) | O(log n) |

### Common AI Algorithm Complexities
| Algorithm | Training | Prediction |
| --- | --- | --- |
| Linear Regression | O(n·d²) | O(d) |
| KNN | O(1) | O(n·d) |
| Decision Tree | O(n·d·log n) | O(log n) |
| K-Means | O(n·k·d·i) | O(k·d) |
| Neural Network | O(epochs·n·layers) | O(layers) |

*n = samples, d = features, k = clusters, i = iterations*

---

## 📌 9. Interview Problems by Category

### Must-Practice Problems (Top 20 for AI Roles)

**Arrays & Strings:**
1. Two Sum
2. Best Time to Buy and Sell Stock
3. Maximum Subarray (Kadane's Algorithm)
4. Merge Intervals
5. Product of Array Except Self

**Hash Maps:**
6. Group Anagrams
7. Top K Frequent Elements
8. Longest Substring Without Repeating Characters

**Trees:**
9. Maximum Depth of Binary Tree
10. Validate Binary Search Tree
11. Serialize and Deserialize Binary Tree

**Graphs:**
12. Number of Islands
13. Course Schedule (Topological Sort)
14. Word Ladder

**Dynamic Programming:**
15. Longest Common Subsequence
16. Edit Distance
17. Coin Change

**Sorting & Searching:**
18. Merge K Sorted Lists
19. Search in Rotated Sorted Array
20. Kth Largest Element

---

## 🎤 Interview Q&A (Top 10 DSA Questions for AI Roles)

### Q1: Two Sum — the most common interview opener
> **How to answer:** Show O(n) hash map solution, not O(n²) brute force.

```python
def two_sum(nums, target):
    """Return indices of two numbers that sum to target."""
    seen = {}  # value → index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# Example
print(two_sum([2, 7, 11, 15], 9))  # [0, 1]
```
**Time:** O(n) | **Space:** O(n)
**AI connection:** Similar pattern used in finding duplicate embeddings in vector databases.

---

### Q2: Maximum Subarray (Kadane's Algorithm)
> **How to answer:** Classic DP problem — asked at Google, Amazon, Meta.

```python
def max_subarray(nums):
    """Find contiguous subarray with maximum sum."""
    max_sum = current_sum = nums[0]
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum

print(max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))  # 6 → [4,-1,2,1]
```
**AI connection:** Used in finding the most impactful sequence of features or the highest-confidence prediction window.

---

### Q3: Top K Frequent Elements
> **How to answer:** Show heap-based O(n log k) solution.

```python
from collections import Counter
import heapq

def top_k_frequent(nums, k):
    """Return k most frequent elements."""
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)

print(top_k_frequent([1,1,1,2,2,3], 2))  # [1, 2]

# Alternative: Bucket Sort — O(n)
def top_k_bucket(nums, k):
    count = Counter(nums)
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, freq in count.items():
        buckets[freq].append(num)
    result = []
    for i in range(len(buckets) - 1, 0, -1):
        result.extend(buckets[i])
        if len(result) >= k:
            return result[:k]
```
**AI connection:** Directly used in NLP for finding top-K words, top-K model predictions, and beam search.

---

### Q4: LRU Cache — system design meets DSA
> **How to answer:** Must implement with O(1) get and put.

```python
from collections import OrderedDict

class LRUCache:
    """Least Recently Used Cache — used for model prediction caching."""
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)  # Mark as recently used
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # Remove least recent

# Usage:
cache = LRUCache(3)
cache.put("user_123", 0.95)  # Cache prediction
cache.get("user_123")         # 0.95 — O(1)
```
**AI connection:** Production ML systems cache expensive model predictions to avoid re-computation.

---

### Q5: Number of Islands — BFS/DFS on grid
> **How to answer:** Shows graph traversal skill on matrix data.

```python
def num_islands(grid):
    """Count connected components (1s) in a grid."""
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
            return
        grid[r][c] = '0'  # Mark visited
        dfs(r+1, c); dfs(r-1, c)
        dfs(r, c+1); dfs(r, c-1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)
    return count
```
**AI connection:** Connected component analysis is used in image segmentation and clustering visualization.

---

### Q6: Merge Intervals — data preprocessing classic
> **How to answer:** Sort first, then merge.

```python
def merge_intervals(intervals):
    """Merge overlapping intervals."""
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged

print(merge_intervals([[1,3],[2,6],[8,10],[15,18]]))
# [[1,6],[8,10],[15,18]]
```
**AI connection:** Used in merging time windows for event detection, combining overlapping bounding boxes in object detection (NMS).

---

### Q7: Course Schedule — Topological Sort (graph + BFS)
> **How to answer:** "Can you complete all courses?" = cycle detection in DAG.

```python
from collections import deque, defaultdict

def can_finish(num_courses, prerequisites):
    """Detect if all courses can be completed (no circular dependencies)."""
    graph = defaultdict(list)
    in_degree = [0] * num_courses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    completed = 0

    while queue:
        course = queue.popleft()
        completed += 1
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)

    return completed == num_courses

print(can_finish(4, [[1,0],[2,1],[3,2]]))  # True
print(can_finish(2, [[0,1],[1,0]]))         # False (cycle!)
```
**AI connection:** ML pipelines are DAGs — this validates if your pipeline has circular dependencies.

---

### Q8: Edit Distance — the NLP interview question
> **How to answer:** Classic DP, directly used in spell correction.

```python
def min_distance(word1, word2):
    """Minimum operations (insert/delete/replace) to convert word1 → word2."""
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],     # Delete
                    dp[i][j-1],     # Insert
                    dp[i-1][j-1]    # Replace
                )
    return dp[m][n]

print(min_distance("kitten", "sitting"))  # 3
```
**AI connection:** Used in fuzzy matching, autocorrect, and evaluating text generation quality (WER).

---

### Q9: Implement a Min-Heap from scratch
> **How to answer:** Shows deep understanding of priority queues.

```python
class MinHeap:
    def __init__(self):
        self.heap = []

    def push(self, val):
        self.heap.append(val)
        self._sift_up(len(self.heap) - 1)

    def pop(self):
        if len(self.heap) == 1:
            return self.heap.pop()
        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._sift_down(0)
        return root

    def _sift_up(self, i):
        parent = (i - 1) // 2
        while i > 0 and self.heap[i] < self.heap[parent]:
            self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]
            i = parent
            parent = (i - 1) // 2

    def _sift_down(self, i):
        n = len(self.heap)
        while True:
            smallest = i
            left, right = 2*i + 1, 2*i + 2
            if left < n and self.heap[left] < self.heap[smallest]:
                smallest = left
            if right < n and self.heap[right] < self.heap[smallest]:
                smallest = right
            if smallest == i:
                break
            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
            i = smallest
```
**AI connection:** Priority queues power beam search in text generation and Dijkstra's algorithm in graph-based models.

---

### Q10: Design a system to find similar items — combining DSA + ML
> **How to answer:** The ultimate question that tests both DSA and AI knowledge.

```python
import numpy as np
from collections import defaultdict

class SimilaritySearch:
    """Simple approximate nearest neighbor search using locality-sensitive hashing."""
    def __init__(self, num_planes=10, dim=128):
        self.planes = np.random.randn(num_planes, dim)
        self.buckets = defaultdict(list)

    def _hash(self, vector):
        """Create hash by checking which side of each hyperplane the vector falls."""
        projections = self.planes @ vector
        return tuple((projections > 0).astype(int))

    def index(self, item_id, vector):
        h = self._hash(vector)
        self.buckets[h].append((item_id, vector))

    def query(self, vector, top_k=5):
        h = self._hash(vector)
        candidates = self.buckets[h]
        if not candidates:
            return []
        # Rank candidates by cosine similarity
        scores = []
        for item_id, v in candidates:
            sim = np.dot(vector, v) / (np.linalg.norm(vector) * np.linalg.norm(v))
            scores.append((item_id, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
```
**Why this impresses:** Shows you understand LSH (used in real vector databases like Pinecone/FAISS), combining algorithmic thinking with ML knowledge.

---

## 🎯 Mastery Checklist

- [ ] Understand time & space complexity for all data structures
- [ ] Implement Two Pointer and Sliding Window patterns
- [ ] Use Hash Maps for counting, grouping, and caching
- [ ] Traverse trees (DFS, BFS) and understand Decision Tree connection
- [ ] Implement BFS/DFS on graphs and Topological Sort
- [ ] Write Binary Search for threshold optimization
- [ ] Solve basic DP problems (Fibonacci, LCS, Knapsack, Edit Distance)
- [ ] Know AI algorithm complexities for system design interviews
- [ ] Practice top 20 LeetCode problems for AI interviews
- [ ] Answer all 10 interview questions with code confidently
