def compute_utility(
    learning_gain,
    representation_gain=0.0,
    cost=0.0,
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
):
    return alpha * learning_gain + beta * representation_gain - gamma * cost
