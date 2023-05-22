import numpy as np


def generate_fake_price_data(
    N,
    seed: int,
    num_points=100,
    amplitude_range=(0.5, 1.5),
    frequency_range=(2, 2),
    noise_std=0.2,
    phase_shift=5,
    autocorrelation_coefficient=0.99,
):
    """Generate N approximately sinusoidal data sets with noise.

    Each data set is a shift of the previous data set plus some noise, and a phase shift of the sinusoids.

    Args:
        N (int): Number of data sets to generate.
        num_points (int): Number of data points in each data set.
        amplitude_range (tuple): Range of amplitudes of the sinusoids.
        frequency_range (tuple): Range of frequencies of the sinusoids.
        noise_std (float): Standard deviation of the Gaussian noise.
        phase_shift (float): Phase shift of the sinusoids between data sets.

    Returns:
        A list of N+1 arrays, where each array contains the generated data set.
    """

    np.random.seed(seed)

    # Generate the first data set
    x = np.linspace(0, 2 * np.pi, num_points)
    y = np.random.normal(loc=0, scale=noise_std, size=num_points)

    amplitude = np.random.uniform(*amplitude_range)
    frequency = np.random.uniform(*frequency_range)
    autocorr = np.random.normal(loc=0, scale=amplitude / 4, size=num_points)

    for j in range(1, num_points):
        autocorr[j] += autocorrelation_coefficient * autocorr[j - 1]

    y += amplitude * np.sin(frequency * x) + autocorr

    data_sets = [y]

    # Generate the remaining data sets
    for i in range(1, N):
        y = np.random.normal(loc=0, scale=noise_std, size=num_points)
        phase = phase_shift * (i + 1)

        amplitude = np.random.uniform(*amplitude_range)
        frequency = np.random.uniform(*frequency_range)
        autocorr = np.random.normal(loc=0, scale=amplitude / 4, size=num_points)

        for j in range(1, num_points):
            autocorr[j] += autocorrelation_coefficient * autocorr[j - 1]

        y += amplitude * np.sin(frequency * x + phase) + autocorr

        data_sets.append(y)

    new = []
    for i in data_sets:
        i = np.interp(i, (np.min(i), np.max(i)), (-100, 15000))
        i = np.clip(i, -100, 15000)
        new.append(i)

    return new


# Generate data with seed=42 and N=5
# data = generate_data(2, 6943, 2)
# data = generate_fake_price_data(
#     N=5,
#     seed=100,
#     num_points=100,
#     noise_std=0.3,
#     phase_shift=0.2,
#     autocorrelation_coefficient=0.98,
# )
# combined_data = np.concatenate(data)

# # Plot each set of data
# fig, axs = plt.subplots(nrows=1, ncols=len(data), figsize=(16, 4))
# for i in range(len(data)):
#     axs[i].plot(data[i])
#     axs[i].set_title(f"Set {i+1}")
# plt.show()

# Plot all sets of data on the same plot
# fig, ax = plt.subplots(figsize=(8, 4))
# for i in range(len(data)):
#     ax.plot(data[i], label=f"Set {i+1}")
# ax.legend()
# plt.show()

# # Plot the concatenated data
# fig, ax = plt.subplots(figsize=(8, 4))
# ax.plot(combined_data, label="Concatenated")
# ax.legend()
# plt.show()
