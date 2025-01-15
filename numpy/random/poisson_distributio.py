"""Poisson Distribution is a Discrete Distribution.

It estimates how many times an event can happen in a specified time. e.g. If someone eats twice a day what is the probability he will eat thrice?

It has two parameters:

lam - rate or known number of occurrences e.g. 2 for above problem.

size - The shape of the returned array.
    """
    
"""Let’s break it down step by step to help you understand!

---

### **Code**
```python
from numpy import random

x = random.poisson(lam=2, size=10)

print(x)
```

---

### **Explanation**

#### **1. `random.poisson(lam, size)`**
This function generates random numbers from a **Poisson distribution**.

#### **Poisson Distribution Overview**:
- The Poisson distribution models the number of times an event occurs within a fixed interval of time or space.
- Events must:
  - Happen independently of each other.
  - Occur at a constant average rate.
- Parameters:
  - **`lam`** (lambda): The average number of events (rate of occurrence) in the interval.
  - **`size`**: The number of random values to generate.

---

### **In This Code**:
- **`lam=2`**: On average, 2 events are expected in the given interval.
- **`size=10`**: Generate 10 random numbers from this Poisson distribution.

---

#### **What It Means**:
Suppose you're modeling the number of emails you receive in an hour, and you expect to receive 2 emails on average (`lam=2`). The Poisson distribution can tell you how likely it is to receive 0, 1, 2, 3, or more emails in an hour.

Each number in the output array (`x`) represents the number of events (e.g., emails received) in a single interval.

---

### **Example Output**
```python
[1, 3, 0, 2, 4, 1, 2, 2, 3, 0]
```
Here:
- The first value (`1`) means that in the first interval, 1 event occurred.
- The second value (`3`) means that in the second interval, 3 events occurred.
- And so on.

---

### **Visualizing Poisson Distribution**
If you want to visualize the distribution, you can use this code:
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Generate Poisson distributed data
data = random.poisson(lam=2, size=1000)

# Plot the distribution
sns.histplot(data, kde=False, bins=20)
plt.title("Poisson Distribution with λ=2")
plt.xlabel("Number of Events")
plt.ylabel("Frequency")
plt.show()
```

---

### **Real-World Examples**
The Poisson distribution is used in scenarios where events occur randomly and independently:
1. **Emails or Messages**:
   - Number of emails received per hour.
2. **Traffic**:
   - Number of cars passing through a toll booth in an hour.
3. **Defects in Manufacturing**:
   - Number of defects per batch in a production line.
4. **Customer Arrivals**:
   - Number of customers arriving at a store in a given time.

---

### **Key Takeaways**
- **`lam` (λ)** controls the average number of events.
- A Poisson-distributed random variable tells you how many events are likely to occur in a fixed interval.
- Output represents the number of occurrences per interval for multiple intervals (`size`).

Let me know if you'd like further clarification or examples!"""
    
    
    
# from numpy import random

# x = random.poisson(lam=2, size=10)

# print(x)

from numpy import random
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

# Generate Poisson distributed data
data = random.poisson(lam=2, size=5)

# Plot the distribution
sns.histplot(data, kde=False, bins=20)
plt.title("Poisson Distribution with λ=2")
plt.xlabel("Number of Events")
plt.ylabel("Frequency")
print(data)
plt.savefig("poison_distribution.png")
