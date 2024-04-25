import matplotlib as plt
from encoders import PCAEncoder
from pathlib import Path
import numpy as np



def savefig(fname, fig=None, verbose=True):
    path = Path("..", "figs", fname)
    (plt if fig is None else fig).savefig(path, bbox_inches="tight", pad_inches=0)
    if verbose:
        print(f"Figure saved as '{path}'")



# np.random.seed(3164)  # make sure you keep this seed
# j1, j2 = np.random.choice(d, 2, replace=False)  # choose 2 random features
# random_is = np.random.choice(n, 15, replace=False)  # choose random examples

fig, ax = plt.subplots()
# ax.scatter(X_train_standardized[:, j1], X_train_standardized[:, j2])
encoder = PCAEncoder(2)
encoder.fit(X_train_standardized)
Pcs = encoder.W # PCs are dimension (k x d) ## 2 x 85
Z = encoder.encode(X_train_standardized) # encoded space Z is dimension (n x k) (k is number of modes) ## 50 x 85
plt.scatter(Z[:, 0], Z[:, 1])
plt.xlabel("Z_1")
plt.ylabel("Z_2")
plt.title("Encoded space")
random_is = np.random.choice(n, 30, replace=False)  # choose random features
for i in random_is:
    xy = Z[i]
    # arrowProps ={'arrowstyle':'simple'}
    ax.annotate(animal_names[i], xy=xy)
savefig("PC_plot.png", fig)
plt.close(fig)
pc1_most_influential = np.argmax(np.abs(Pcs[0]))
pc2_most_influential = np.argmax(np.abs(Pcs[1]))
print(f"Trait with largest influence on PC1: {trait_names[pc1_most_influential]}")
print(f"Trait with largest influence on PC2: {trait_names[pc2_most_influential]}")

