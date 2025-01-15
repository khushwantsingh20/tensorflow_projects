"""Binomial Distribution
Binomial Distribution is a Discrete Distribution.

It describes the outcome of binary scenarios, e.g. toss of a coin, it will either be head or tails.

It has three parameters:

n - number of trials.

p - probability of occurence of each trial (e.g. for toss of a coin 0.5 each).

size - The shape of the returned array.

The code and its functionality revolve around visualizing the **binomial distribution** and understanding how data behaves. Here's a detailed explanation of its **use cases**:

---    

### **1. Understanding Distributions**
The code helps to visualize the **binomial distribution** of data:
- **Binomial distribution**: It is a discrete distribution that describes the outcome of binary scenarios, e.g. toss of a coin, it will either be head or tails.

**Use Case**:
- Identify whether your dataset follows a binomial distribution.
- Spot deviations, such as skewness or outliers.

---

### **2. Financial Analysis**
- **Binomial Distribution**: It is used to model the probability of success in a series of Bernoulli trials, where each trial has a fixed probability of success.

**Use Case**:
- Financial Analysis: Used to model the probability of success in a series of Bernoulli trials, where each trial has a fixed probability of success.

---

### **3. Machine Learning**
- **Data Preprocessing**:
    - Helps visualize and ensure data is normally distributed (a common requirement for many machine learning algorithms).
    - Identify the need for transformations (e.g., logarithmic, square root) to make data normal.

Example:
- Regression models often assume that residuals (errors) follow a normal distribution.

---     

### **4. Statistical Testing**
- Helps in visualizing data before performing statistical tests like:
    - **t-tests**: Assume data is normally distributed.
    - **ANOVA**: Relies on the normality of data.

---    

### **5. Comparing Datasets**
You can compare different datasets by overlaying multiple KDE plots. This is useful in:
- Hypothesis testing.
- Analyzing trends across different groups or conditions.

---

### **6. Data Simulation**
- **Generate synthetic data**: Useful for simulations when real-world data isn’t available.

---    

In summary, **this code is useful for visualizing and understanding the shape of data distributions, making it a powerful tool in data analysis, machine learning, and statistical modeling.**  """

# import matplotlib.pyplot as plt
# from numpy import random
# import numpy as np
# import seaborn as sns

# # Use kdeplot instead of distplot
# sns.kdeplot(random.binomial(n=10, p=0.5, size=1000), fill=True)  # `fill=True` adds a shaded area under the curve

# plt.savefig("binomial_distribution_plot.png")

from numpy import random

x = random.binomial(n=10, p=0.5, size=10)

print(x)