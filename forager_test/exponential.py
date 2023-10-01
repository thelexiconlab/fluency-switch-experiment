import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def exponential_curve(x, c, m):
    return c * (1 - np.exp(-m * x))

def fit_and_plot_curves(reaction_times, labels):
    x = np.arange(1, len(reaction_times)+1)
    # Adjusted initial parameter guesses
    c_guess = max(reaction_times)  # Set initial guess for c to the maximum data value
    m_guess = 0.01

    # Parameter bounds
    parameter_bounds = ([0, 0], [2 * max(reaction_times), 5])  # Constrain c to [0, 2 * max(y)] and m to [0, 5]

    # Fit the curve
    popt, _ = curve_fit(exponential_curve, x, reaction_times, p0=[c_guess, m_guess], bounds=parameter_bounds, maxfev=10000)

    # Fit the exponential curve to the data
    fitted_curve = exponential_curve(x, *popt)

    # Calculate slope differences and classifications
    classifications = np.zeros(len(x))
    for i in range(len(x)):
        if i == 0:
            classifications[i] = 2
        else:
            fitted_slope_difference = fitted_curve[i] - fitted_curve[i - 1]
            raw_slope_difference = reaction_times[i] - reaction_times[i - 1]
            if fitted_slope_difference > raw_slope_difference:
                # fitted RT is higher than raw RT
                classifications[i] = 0
            else:
                # fitted RT is lower than raw RT
                classifications[i] = 1
        
    print("Classifications: ", classifications)

    # Plot the raw data and the fitted curve
    # Create a custom colormap with blue, red, and green colors
    custom_cmap = ListedColormap(['blue', 'red', 'green'])

    # Plot the raw data (colored dots)
    plt.figure(figsize=(10, 6))
    plt.scatter(x, reaction_times, label='Raw Data', c=classifications, cmap=custom_cmap, s=50,edgecolors='face')

    # Plot the fitted curve
    plt.plot(x, fitted_curve, label='Fitted Curve', color='black', linewidth=2)
    
    # Add labels to the points
    for i, label in enumerate(labels):
        plt.annotate(label, (x[i], reaction_times[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    
    plt.xlabel('Response Number')
    plt.ylabel('Reaction Time')
    plt.title('Raw Data and Fitted Curve')
    plt.legend()
    plt.colorbar(label='Classification')
    plt.show()

# Your reaction times and labels
reaction_times = [3.841, 5.541, 8.441, 18.041, 25.641, 31.441, 33.741, 35.341, 42.341, 47.341,
                  59.041, 65.941, 72.141, 77.041, 87.141, 90.741, 104.141, 124.841, 133.441, 142.741, 159.341, 162.541, 175.041]
labels = ['lawyer', 'doctor', 'researcher', 'social worker', 'politician', 'writer', 'singer', 'musician', 'chef',
          'secretary', 'nurse', 'audiologist', 'speech pathologist', 'teacher', 'seamstress', 'actor', 'cashier',
          'professor', 'construction worker', 'manager', 'firefighter', 'police', 'surgeon']

fit_and_plot_curves(reaction_times, labels)
