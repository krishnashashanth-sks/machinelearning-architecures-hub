import numpy as np

def update_deep_esn_states(u, x1_prev, x2_prev, W_in_1, W_res_1, W_in_2, W_res_2):
    """
    Updates the states of a two-layer deep ESN for a given input.

    Args:
        u (np.ndarray): Current input signal (input_dim, 1).
        x1_prev (np.ndarray): Previous state of Layer 1 reservoir (reservoir_size, 1).
        x2_prev (np.ndarray): Previous state of Layer 2 reservoir (reservoir_size_2, 1).
        W_in_1 (np.ndarray): Input weight matrix for Layer 1.
        W_res_1 (np.ndarray): Recurrent weight matrix for Layer 1.
        W_in_2 (np.ndarray): Input weight matrix for Layer 2.
        W_res_2 (np.ndarray): Recurrent weight matrix for Layer 2.

    Returns:
        tuple: A tuple containing the updated states (x1_curr, x2_curr).
    """
    # Layer 1 update
    # x_1(t) = tanh(W_in_1 * u(t) + W_res_1 * x_1(t-1))
    x1_curr = np.tanh(np.dot(W_in_1, u) + np.dot(W_res_1, x1_prev))

    # Layer 2 update
    # Input to Layer 2: [u(t); x_1(t)]
    # u is (input_dim, 1), x1_curr is (reservoir_size, 1)
    # Concatenate u and x1_curr along the first axis (rows)
    input_to_layer2 = np.vstack((u, x1_curr))

    # x_2(t) = tanh(W_in_2 * [u(t); x_1(t)] + W_res_2 * x_2(t-1))
    x2_curr = np.tanh(np.dot(W_in_2, input_to_layer2) + np.dot(W_res_2, x2_prev))

    return x1_curr, x2_curr