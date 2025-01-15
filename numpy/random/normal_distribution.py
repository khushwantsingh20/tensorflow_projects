# from numpy import random

# x = random.normal(size=(2, 3))

# print(x)



# from numpy import random
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Use kdeplot instead of distplot
# sns.kdeplot(random.normal(loc=1,scale=2, size=(2,3)), fill=True)  # `fill=True` adds a shaded area under the curve

# plt.savefig("normal_distribution_plot.png")



from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

weights = random.normal(loc=70, scale=10, size=1000)  # Mean weight = 70kg, Std Dev = 10kg
sns.kdeplot(weights, fill=True)
plt.title("Weight Distribution")
plt.savefig("normal_distribution_plot1.png")
