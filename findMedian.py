import numpy as np

# Data
state_to_median_value = {
    'Alabama': 267100,
    'Alaska': 350000,
    'Arizona': 435300,
    'Arkansas': 246000,
    'California': 793600,
    'Colorado': 586100,
    'Connecticut': 424900,
    'Delaware': 375000,
    'District of Columbia': 640000,
    'Florida': 405000,
    'Georgia': 389500,
    'Hawaii': 714100,
    'Idaho': 539000,
    'Illinois': 266800,
    'Indiana': 242500,
    'Iowa': 289900,
    'Kansas': 263700,
    'Kentucky': 246700,
    'Louisiana': 243300,
    'Maine': 370000,
    'Maryland': 395000,
    'Massachusetts': 595700,
    'Michigan': 238800,
    'Minnesota': 330500,
    'Mississippi': 232800,
    'Missouri': 243500,
    'Montana': 609900,
    'Nebraska': 280400,
    'Nevada': 479299,
    'New Hampshire': 451400,
    'New Jersey': 485900,
    'New Mexico': 358600,
    'New York': 649000,
    'North Carolina': 362200,
    'North Dakota': 334075,
    'Ohio': 228000,
    'Oklahoma': 233900,
    'Oregon': 490200,
    'Pennsylvania': 268100,
    'Rhode Island': 455500,
    'South Carolina': 360800,
    'South Dakota': 300200,
    'Tennessee': 418900,
    'Texas': 336400,
    'Utah': 548900,
    'Vermont': 395800,
    'Virginia': 415600,
    'Washington': 605400,
    'West Virginia': 284000,
    'Wisconsin': 329000,
    'Wyoming': 317000
}

# Get median value
median_price = np.median(list(state_to_median_value.values()))

# Calculate absolute differences from the median
differences = {state: abs(price - median_price) for state, price in state_to_median_value.items()}

# Sort states based on absolute differences
sorted_states = sorted(differences.items(), key=lambda x: x[1])

# Output the sorted list of states closest to the median
print("States closest to the median housing price:")
for state, difference in sorted_states:
    print(state, "($", state_to_median_value[state], ") - Difference:", difference)
