import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Set up the parameters
n = 100  # number of bulbs in each batch
p = 0.05  # probability of a defect (0.5%)

# Create an array of possible numbers of defective bulbs
k = np.arange(0, 200)  # We'll calculate for 0 to 19 defective bulbs

# Calculate the PMF for each number of defective bulbs
pmf = stats.binom.pmf(k, n, p)

# Plot the results
plt.figure(figsize=(10, 6))
plt.bar(k, pmf)
plt.title("Probability of Defective Bulbs in Batches of 1000")
plt.xlabel("Number of Defective Bulbs")
plt.ylabel("Probability")
plt.xticks(k)
plt.show()

# Calculate the cumulative probability of having 10 or fewer defective bulbs
cumulative_prob = stats.binom.cdf(10, n, p)
print(f"Probability of 10 or fewer defective bulbs: {cumulative_prob:.4f}")

# Calculate the expected number of defective bulbs
expected_defects = n * p
print(f"Expected number of defective bulbs: {expected_defects:.2f}")


coin_flip = stats.binom(p=0.5, n=10)

cf_expected_value = coin_flip.expect()
print(f"Expected value of a coin flip: {cf_expected_value:.4f}")

print(f"CDF: {coin_flip.cdf(5):.4f}")
print(f"PMF: {coin_flip.pmf(5):.4f}")
print(f"PPF: {coin_flip.ppf(0.5):.4f}")

# The sum of PMF values is equal to CDF
assert coin_flip.cdf(5) == sum([coin_flip.pmf(i) for i in range(5)])
