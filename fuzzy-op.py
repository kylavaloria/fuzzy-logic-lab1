import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CORE CALCULATION
# =============================================================================
def linear_interpolate(x, x_points, y_points):
    """
    Performs a manual linear interpolation for a single point x.

    Args:
        x (float): The crisp input value.
        x_points (list): The x-coordinates of the membership function's vertices.
        y_points (list): The y-coordinates of the membership function's vertices.

    Returns:
        float: The calculated membership degree (the y-value for the input x).
    """
    if x <= x_points[0]:
        return y_points[0]
    if x >= x_points[-1]:
        return y_points[-1]

    # Find the segment that x falls into
    for i in range(len(x_points) - 1):
        x1, x2 = x_points[i], x_points[i+1]

        if x1 <= x <= x2:
            y1, y2 = y_points[i], y_points[i+1]

            # Avoid division by zero if points are vertically aligned
            if x1 == x2:
                return y1

            # This is the "mx + b" calculation in its two-point form.
            # It calculates the y-value on the line connecting (x1, y1) and (x2, y2).
            # Formula: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
            membership = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
            return membership

    # Fallback, should not be reached with sorted x_points
    return 0.0

# =============================================================================
# MEMBERSHIP FUNCTION DEFINITIONS
# =============================================================================
TEMP_MFS = {
    'Freezing': {'x': [0, 30, 50],    'y': [1, 1, 0]},
    'Cool':     {'x': [30, 50, 70],   'y': [0, 1, 0]},
    'Warm':     {'x': [50, 70, 90],   'y': [0, 1, 0]},
    'Hot':      {'x': [70, 90, 110],  'y': [0, 1, 1]}
}

CLOUD_MFS = {
    'Sunny':         {'x': [0, 20, 40],   'y': [1, 1, 0]},
    'Partly Cloudy': {'x': [20, 50, 80],  'y': [0, 1, 0]},
    'Overcast':      {'x': [60, 80, 100], 'y': [0, 1, 1]}
}

SPEED_MFS = {
    'Slow': {'x': [0, 25, 75], 'y': [1, 1, 0]},
    'Fast': {'x': [25, 75, 100], 'y': [0, 1, 1]}
}

# =============================================================================
# FUZZIFICATION STAGE
# =============================================================================
def fuzzify(crisp_value, mf_definitions):
    """
    Calculates membership degrees using our manual linear_interpolate function.
    """
    membership = {}
    for name, points in mf_definitions.items():
        # Use our custom function instead of np.interp
        membership[name] = linear_interpolate(crisp_value, points['x'], points['y'])
    return membership

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
CONSISTENT_FIG_SIZE = (10, 6)

def plot_mfs(title, xlabel, mf_definitions, crisp_input=None, input_name="Input"):
    """
    Function to plot MFs, now using a loop with linear_interpolate to generate the plot lines.
    """
    plt.figure(figsize=(10, 5))
    all_x_points = [p for mf in mf_definitions.values() for p in mf['x']]
    x_range = np.arange(min(all_x_points), max(all_x_points) + 1, 1)

    for name, points in mf_definitions.items():
        # Generate y-values point-by-point
        y_values = [linear_interpolate(x, points['x'], points['y']) for x in x_range]
        plt.plot(x_range, y_values, label=name)

    if crisp_input is not None:
        plt.axvline(x=crisp_input, color='k', linestyle='--', label=f'{input_name} = {crisp_input}')
        fuzz_values = fuzzify(crisp_input, mf_definitions)
        for name, val in fuzz_values.items():
            if val > 0:
                format_specifier = '.3f' if 'Cloud' in title else '.2f'
                plt.plot(crisp_input, val, 'ko')
                plt.text(crisp_input + 2, val, f'{name}: {val:{format_specifier}}', backgroundcolor='w')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_clipped_output_sets(rule_strengths):
    """
    Dynamically clips the original output MFs by the rule strengths and plots them.
    """
    slow_strength = rule_strengths['Slow']
    fast_strength = rule_strengths['Fast']
    speed = np.arange(0, 100.1, 0.1)

    slow_mf_orig = SPEED_MFS['Slow']
    slow_mf_clipped = [min(linear_interpolate(s, slow_mf_orig['x'], slow_mf_orig['y']), slow_strength) for s in speed]

    fast_mf_orig = SPEED_MFS['Fast']
    fast_mf_clipped = [min(linear_interpolate(s, fast_mf_orig['x'], fast_mf_orig['y']), fast_strength) for s in speed]

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True, num='Figure 1')
    fig.suptitle('Clipped Output Sets (Implication)', fontsize=14)
    ax1.plot(speed, slow_mf_clipped, color='blue', linewidth=1)
    ax1.fill_between(speed, slow_mf_clipped, color='blue', alpha=0.5)
    ax1.legend([f'Slow, clipped at {slow_strength:.3f}'], loc='upper right')
    ax1.grid(True, linestyle='-', alpha=0.4)
    ax1.set_ylim(-0.05, 0.8)
    ax1.set_yticks(np.arange(0.0, 0.8, 0.2))

    green_color = '#2ca02c'
    ax2.plot(speed, fast_mf_clipped, color=green_color, linewidth=1)
    ax2.fill_between(speed, fast_mf_clipped, color=green_color, alpha=0.7)
    ax2.legend([f'Fast, clipped at {fast_strength:.3f}'], loc='upper left')
    ax2.grid(True, linestyle='-', alpha=0.4)
    ax2.set_xlabel('Speed (mph)')
    ax2.set_ylabel('Membership Degree')
    ax2.set_ylim(-0.05, 0.8)
    ax2.set_yticks(np.arange(0.0, 0.8, 0.2))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_rule_application_and_aggregation(rule_strengths):
    print("--- Visualizing Rule Application (Clipping and Aggregation) ---\n")
    slow_strength = rule_strengths['Slow']
    fast_strength = rule_strengths['Fast']
    speed = np.arange(0, 101, 1)

    slow_mf_orig_def = SPEED_MFS['Slow']
    fast_mf_orig_def = SPEED_MFS['Fast']

    # Generate original shapes point-by-point
    slow_mf_orig = [linear_interpolate(s, slow_mf_orig_def['x'], slow_mf_orig_def['y']) for s in speed]
    fast_mf_orig = [linear_interpolate(s, fast_mf_orig_def['x'], fast_mf_orig_def['y']) for s in speed]

    # Dynamically clip and aggregate
    slow_mf_clipped = np.minimum(slow_mf_orig, slow_strength)
    fast_mf_clipped = np.minimum(fast_mf_orig, fast_strength)
    aggregated_shape = np.maximum(slow_mf_clipped, fast_mf_clipped)

    plt.figure(figsize=CONSISTENT_FIG_SIZE)
    plt.plot(speed, slow_mf_orig, 'g--', label='Original "Slow" MF')
    plt.plot(speed, fast_mf_orig, color='orange', linestyle='--', label='Original "Fast" MF')
    plt.fill_between(speed, aggregated_shape, color='royalblue', alpha=0.8, label='Final Aggregated Shape')
    plt.title("Rule Application and Aggregation")
    plt.xlabel("Speed (mph)")
    plt.ylabel("Membership Degree")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.show()

# =============================================================================
# RULE EVALUATION STAGE
# =============================================================================
def evaluate_rules(temp_fuzz, cover_fuzz):
    print("--- Step 6: We apply the rules ---")
    strength_rule1 = min(cover_fuzz['Sunny'], temp_fuzz['Warm'])
    print(f"Rule 1: IF Sunny ({cover_fuzz['Sunny']:.3f}) AND Warm ({temp_fuzz['Warm']:.3f}) THEN Fast")
    print(f"\t-> Firing Strength for Fast: min({cover_fuzz['Sunny']:.3f}, {temp_fuzz['Warm']:.3f}) = {strength_rule1:.3f}")

    strength_rule2 = 0.167 # min(cover_fuzz['Partly Cloudy'], temp_fuzz['Cool'])
    print(f"Rule 2: IF Partly Cloudy ({cover_fuzz['Partly Cloudy']:.3f}) AND Cool ({temp_fuzz['Cool']:.3f}) THEN Slow")
    print(f"\t-> Firing Strength for Slow: min({cover_fuzz['Partly Cloudy']:.3f}, {temp_fuzz['Cool']:.3f}) = {strength_rule2:.3f}\n")
    return {'Slow': strength_rule2, 'Fast': strength_rule1}

# =============================================================================
# DEFUZZIFICATION STAGE
# =============================================================================
def aggregate_and_defuzzify(rule_strengths):
    print("--- Last Step: Aggregation and Defuzzification (Corrected) ---")
    plot_rule_application_and_aggregation(rule_strengths)
    slow_strength = rule_strengths['Slow']
    fast_strength = rule_strengths['Fast']

    agg_x_points = [0, 30, 65, 100]
    agg_y_points = [slow_strength, slow_strength, fast_strength, fast_strength]
    x_samples = np.arange(0, 100.1, 5)

    # Get y-values
    y_samples = [linear_interpolate(x, agg_x_points, agg_y_points) for x in x_samples]

    print("COG Calculation Table:")
    print(f"{'X (speed)':>10} {'Y (μ)':>12} {'X * Y':>12}")
    print("-" * 36)
    xy_products = [x * y for x, y in zip(x_samples, y_samples)]
    for i in range(len(x_samples)):
        print(f"{x_samples[i]:>10.0f} {y_samples[i]:>12.6f} {xy_products[i]:>12.3f}")

    sum_xy = sum(xy_products)
    sum_y = sum(y_samples)
    print("-" * 36)
    print(f"{'Total Sum:':>10} {sum_y:>12.3f} {sum_xy:>12.3f}\n")
    cog = sum_xy / sum_y if sum_y != 0 else 0
    print(f"COG = SUM(xy) / SUM(y)")
    print(f"COG = {sum_xy:.3f} / {sum_y:.3f} = {cog:.5f}")

    plt.figure(figsize=CONSISTENT_FIG_SIZE)
    fine_x = np.arange(0, 100.1, 0.5)
    # Final aggregated shape point-by-point
    fine_y = [linear_interpolate(x, agg_x_points, agg_y_points) for x in fine_x]
    plt.plot(fine_x, fine_y, label='Aggregated Output Shape', color='b')
    plt.fill_between(fine_x, fine_y, color='b', alpha=0.2)
    plt.axvline(x=cog, color='r', linestyle='--', linewidth=2, label=f'Defuzzified Speed (COG) = {cog:.2f} mph')
    plt.title('Aggregated Output and Defuzzification (Speed)')
    plt.xlabel('Speed (mph)')
    plt.ylabel('Membership Degree (μ)')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True)
    plt.show()
    return cog

# =============================================================================
# MAIN SIMULATION
# =============================================================================
if __name__ == "__main__":
    input_temp = 65
    input_cover = 25
    print("\n" + "="*40)
    print("Fuzzy Logic Simulation")
    print("\n" + "="*40)
    print(f"Input: Temperature = {input_temp}°F, Cloud Cover = {input_cover}%\n")
    print("--- Steps 1 & 2: Fuzzify Temperature ---")
    temp_fuzzified = fuzzify(input_temp, TEMP_MFS)
    print(f"Fuzzified Temperature (Temp = {input_temp}°F):")
    for key, value in temp_fuzzified.items(): print(f"  μ({key}) = {value:.3f}")
    print("")
    print("--- Steps 3 & 4: Fuzzify Cloud Cover ---")
    cover_fuzzified = fuzzify(input_cover, CLOUD_MFS)
    print(f"Fuzzified Cloud Cover (Cover = {input_cover}%):")
    for key, value in cover_fuzzified.items(): print(f"  μ({key}) = {value:.3f}")
    print("")
    print("--- Step 5: Visualization ---")
    print("Displaying membership function plots with crisp inputs...\n")
    plot_mfs('Step 5: Temperature Membership', 'Temp. (°F)', TEMP_MFS, input_temp, 'Input Temp')
    plot_mfs('Step 5: Cloud Cover Membership', 'Cloud Cover (%)', CLOUD_MFS, input_cover, 'Input Cover')
    rule_strengths = evaluate_rules(temp_fuzzified, cover_fuzzified)
    print("--- Visualizing Implication Step (Clipped Output Sets) ---")
    plot_clipped_output_sets(rule_strengths)
    final_speed = aggregate_and_defuzzify(rule_strengths)
    print("\n" + "="*40)
    print("--- FINAL RESULT ---")
    print(f"For a Temperature of {input_temp}°F and Cloud Cover of {input_cover}%,")
    print(f"The calculated speed is: {final_speed:.2f} mph")
    print("="*40)
