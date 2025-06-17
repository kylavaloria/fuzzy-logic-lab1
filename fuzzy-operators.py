import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class FuzzyOperationsSimulator:
    def __init__(self):
        # Temperature membership function definitions from PDF
        self.temp_memberships = {
            'Freezing': [(0, 1), (30, 1), (50, 0), (110, 0)],
            'Cool': [(0, 0), (30, 0), (50, 1), (70, 0), (110, 0)],
            'Warm': [(0, 0), (50, 0), (70, 1), (90, 0), (110, 0)],
            'Hot': [(0, 0), (70, 0), (90, 1), (110, 1)]
        }

        # Cloud cover membership function definitions from PDF
        self.cover_memberships = {
            'Sunny': [(0, 1), (20, 1), (40, 0), (100, 0)],
            'Partly Cloudy': [(0, 0), (20, 0), (50, 1), (80, 0), (100, 0)],
            'Overcast': [(0, 0), (60, 0), (80, 1), (100, 1)]
        }

        # Rules from PDF
        self.rules = [
            ('Sunny', 'Warm', 'Fast'),
            ('Partly Cloudy', 'Cool', 'Slow')
        ]

    def calculate_linear_params(self, x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
        """Calculate slope (m) and intercept (b) for linear equation y = mx + b"""
        if x2 == x1:
            return 0, y1
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return m, b

    def get_membership_value(self, x: float, membership_points: List[Tuple[float, float]]) -> float:
        """Calculate membership value for given input using piecewise linear interpolation"""
        # Sort points by x coordinate
        points = sorted(membership_points, key=lambda p: p[0])

        # If x is outside the range, return 0
        if x <= points[0][0]:
            return points[0][1]
        if x >= points[-1][0]:
            return points[-1][1]

        # Find the segment containing x
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]

            if x1 <= x <= x2:
                if x1 == x2:
                    return y1
                # Linear interpolation
                m, b = self.calculate_linear_params(x1, y1, x2, y2)
                return m * x + b

        return 0

    def evaluate_temperature_membership(self, temp: float) -> Dict[str, float]:
        """Step 2: Calculate fuzzy membership for temperature"""
        print(f"Step 2: Evaluating temperature membership for {temp}°F")
        memberships = {}

        for category, points in self.temp_memberships.items():
            value = self.get_membership_value(temp, points)
            memberships[category] = value
            print(f"{category} = {value}")

        return memberships

    def evaluate_cover_membership(self, cover: float) -> Dict[str, float]:
        """Step 4: Calculate fuzzy membership for cloud cover"""
        print(f"\nStep 4: Evaluating cloud cover membership for {cover}%")
        memberships = {}

        for category, points in self.cover_memberships.items():
            value = self.get_membership_value(cover, points)
            memberships[category] = value
            print(f"{category} = {value}")

        return memberships

    def apply_rules(self, temp_memberships: Dict[str, float], cover_memberships: Dict[str, float]) -> List[Tuple[str, str, str, float]]:
        """Step 6: Apply fuzzy rules using minimum operation"""
        print("\nStep 6: Applying rules")
        rule_results = []

        for cover_condition, temp_condition, output in self.rules:
            cover_value = cover_memberships[cover_condition]
            temp_value = temp_memberships[temp_condition]
            # Use minimum for AND operation
            rule_strength = min(cover_value, temp_value)
            rule_results.append((cover_condition, temp_condition, output, rule_strength))
            print(f"{cover_condition} and {temp_condition} -> {output}: min({cover_value}, {temp_value}) = {rule_strength}")

        return rule_results

    def calculate_centroid(self, rule_results: List[Tuple[str, str, str, float]]) -> float:
        """Calculate centroid (COG) based on rule results"""
        print("\nCalculating Centroid of Gravity (COG)")

        # From PDF: We get 30 from slow and 65 from fast (plateau points)
        # Slow: 0 to 30 with membership 0.167
        # Fast: 65 to 100 with membership 0.75
        # Transition: 30 to 65 with linear interpolation

        slow_membership = 0
        fast_membership = 0

        for _, _, output, strength in rule_results:
            if output == 'Slow':
                slow_membership = strength
            elif output == 'Fast':
                fast_membership = strength

        print(f"Slow membership: {slow_membership}")
        print(f"Fast membership: {fast_membership}")

        # Calculate slope and intercept for transition region (30 to 65)
        # From PDF: m = 0.017, b = -0.333
        m = (fast_membership - slow_membership) / (65 - 30)
        b = slow_membership - m * 30

        print(f"Transition slope (m): {m}")
        print(f"Transition intercept (b): {b}")

        # Calculate COG using discrete approximation
        x_values = list(range(0, 101))  # 0 to 100
        y_values = []
        xy_values = []

        for x in x_values:
            if x <= 30:
                y = slow_membership
            elif x >= 65:
                y = fast_membership
            else:
                y = m * x + b

            y_values.append(y)
            xy_values.append(x * y)

        sum_y = sum(y_values)
        sum_xy = sum(xy_values)

        print(f"Sum of Y: {sum_y}")
        print(f"Sum of XY: {sum_xy}")

        cog = sum_xy / sum_y if sum_y != 0 else 0
        print(f"COG = {cog}")

        return cog, x_values, y_values

    def visualize_membership_functions(self):
        """Create visualizations for membership functions"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Temperature membership functions
        temp_range = np.linspace(0, 110, 1000)
        ax1.set_title('Temperature Membership Functions')
        ax1.set_xlabel('Temperature (°F)')
        ax1.set_ylabel('Membership')

        colors = ['blue', 'green', 'orange', 'red']
        for i, (category, points) in enumerate(self.temp_memberships.items()):
            memberships = [self.get_membership_value(t, points) for t in temp_range]
            ax1.plot(temp_range, memberships, label=category, color=colors[i], linewidth=2)

        ax1.axvline(x=65, color='black', linestyle='--', alpha=0.7, label='Input: 65°F')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Cloud cover membership functions
        cover_range = np.linspace(0, 100, 1000)
        ax2.set_title('Cloud Cover Membership Functions')
        ax2.set_xlabel('Cloud Cover (%)')
        ax2.set_ylabel('Membership')

        colors = ['yellow', 'gray', 'darkgray']
        for i, (category, points) in enumerate(self.cover_memberships.items()):
            memberships = [self.get_membership_value(c, points) for c in cover_range]
            ax2.plot(cover_range, memberships, label=category, color=colors[i], linewidth=2)

        ax2.axvline(x=25, color='black', linestyle='--', alpha=0.7, label='Input: 25%')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Rule evaluation visualization
        temp_memberships = self.evaluate_temperature_membership(65)
        cover_memberships = self.evaluate_cover_membership(25)
        rule_results = self.apply_rules(temp_memberships, cover_memberships)

        ax3.set_title('Rule Evaluation Results')
        categories = ['Sunny∩Warm', 'Partly∩Cool']
        values = [result[3] for result in rule_results]
        colors = ['lightblue', 'lightcoral']

        bars = ax3.bar(categories, values, color=colors, alpha=0.7)
        ax3.set_ylabel('Rule Strength')
        ax3.set_ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        ax3.grid(True, alpha=0.3)

        # Output membership function and COG
        cog, x_values, y_values = self.calculate_centroid(rule_results)

        ax4.set_title('Output Speed Membership and COG')
        ax4.plot(x_values, y_values, 'b-', linewidth=2, label='Combined Output')
        ax4.axvline(x=cog, color='red', linestyle='--', linewidth=2, label=f'COG = {cog:.2f}')
        ax4.fill_between(x_values, y_values, alpha=0.3, color='blue')
        ax4.set_xlabel('Speed')
        ax4.set_ylabel('Membership')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def run_simulation(self, temp: float, cloud_cover: float):
        """Run complete fuzzy operations simulation"""
        print("="*60)
        print("FUZZY OPERATIONS SIMULATION")
        print("="*60)
        print(f"Input: Temperature = {temp}°F, Cloud Cover = {cloud_cover}%")
        print()

        # Step 1: Define membership functions (already done in __init__)
        print("Step 1: Membership functions defined")

        # Step 2: Evaluate temperature membership
        temp_memberships = self.evaluate_temperature_membership(temp)

        # Step 3: Define cloud cover membership functions (already done)
        print("\nStep 3: Cloud cover membership functions defined")

        # Step 4: Evaluate cloud cover membership
        cover_memberships = self.evaluate_cover_membership(cloud_cover)

        # Step 5: Visualization summary
        print(f"\nStep 5: Visualization Summary")
        print(f"{temp}°F falls under: {[k for k, v in temp_memberships.items() if v > 0]}")
        print(f"{cloud_cover}% cloud cover falls under: {[k for k, v in cover_memberships.items() if v > 0]}")

        # Step 6: Apply rules
        rule_results = self.apply_rules(temp_memberships, cover_memberships)

        # Calculate final output
        cog, _, _ = self.calculate_centroid(rule_results)

        print("\n" + "="*60)
        print(f"FINAL RESULT: Speed = {cog:.5f}")
        print("="*60)

        # Create visualization
        fig = self.visualize_membership_functions()
        plt.show()

        return cog, temp_memberships, cover_memberships, rule_results

# Example usage matching the PDF
if __name__ == "__main__":
    simulator = FuzzyOperationsSimulator()

    # Run simulation with values from PDF: Temp = 65°F, Cloud Cover = 25%
    final_speed, temp_results, cover_results, rules = simulator.run_simulation(65, 25)

    print("\nDetailed Results:")
    print(f"Temperature Memberships: {temp_results}")
    print(f"Cloud Cover Memberships: {cover_results}")
    print(f"Rule Results: {rules}")
