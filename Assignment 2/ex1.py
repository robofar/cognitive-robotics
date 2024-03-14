import numpy as np

def gaussian_distribution(mean, std, x):
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

def softmax(x):
    # Compute softmax values for each sets of scores in x
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()