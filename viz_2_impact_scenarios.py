# Simulates and plots the impact of artificial recharge interventions on aquifer sustainability across scenarios.
"""
VISUALIZATION 2: IMPACT SCENARIO ANALYSIS
Shows: What happens with vs without AI-driven intervention
"""

print("="*60)
print("Starting Impact Scenarios Visualization...")
print("="*60)

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("✓ Libraries imported successfully")

# Simulation data (replace with your actual model predictions)
days = np.arange(0, 365)

print("✓ Generating scenario data...")

# Scenario 1: Critical Region (e.g., Rajasthan)
# Without intervention
baseline_depletion_rate = 0.015  # m/day
gwl_no_intervention = 4.5 + days * baseline_depletion_rate + \
                      0.3 * np.sin(2*np.pi*days/365)  # Seasonal variation

# With AI-recommended intervention
intervention_start = 90
gwl_with_intervention = gwl_no_intervention.copy()
# Intervention reduces depletion by 60%
gwl_with_intervention[intervention_start:] = \
    4.5 + intervention_start * baseline_depletion_rate + \
    (days[intervention_start:] - intervention_start) * baseline_depletion_rate * 0.4 + \
    0.3 * np.sin(2*np.pi*days[intervention_start:]/365)

# Scenario 2: Semi-Critical Region (e.g., Maharashtra)
gwl_semi_no = 8.0 + days * 0.008 + 0.5 * np.sin(2*np.pi*days/365)
gwl_semi_with = gwl_semi_no.copy()
gwl_semi_with[intervention_start:] = \
    8.0 + intervention_start * 0.008 + \
    (days[intervention_start:] - intervention_start) * 0.003 + \
    0.5 * np.sin(2*np.pi*days[intervention_start:]/365)

print("✓ Data generation complete")
print("✓ Creating visualizations...")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('AI-Driven Groundwater Management: Impact Scenarios\nDemonstrating Model Utility in Resource Planning', 
             fontsize=16, fontweight='bold')

# Plot 1: Critical Region Scenario
ax1 = axes[0, 0]
ax1.plot(days, gwl_no_intervention, 'r--', linewidth=2.5, label='Without AI Intervention', alpha=0.7)
ax1.plot(days, gwl_with_intervention, 'g-', linewidth=2.5, label='With AI-Recommended Actions')
ax1.axhline(y=10, color='darkred', linestyle=':', linewidth=2, label='Critical Threshold')
ax1.axvline(x=intervention_start, color='blue', linestyle='--', alpha=0.5, linewidth=1.5)

# Highlight saved water level
ax1.fill_between(days[intervention_start:], 
                 gwl_no_intervention[intervention_start:], 
                 gwl_with_intervention[intervention_start:], 
                 alpha=0.3, color='green', label='Water Level Saved')

ax1.set_xlabel('Days from Present', fontsize=11, fontweight='bold')
ax1.set_ylabel('Groundwater Depth (m below ground)', fontsize=11, fontweight='bold')
ax1.set_title('Critical Region (Rajasthan-Type)\nPreventing Aquifer Collapse', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.invert_yaxis()

# Add annotations
ax1.annotate('AI Alert Triggers\nIntervention', xy=(intervention_start, gwl_no_intervention[intervention_start]),
            xytext=(intervention_start+50, 3), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

water_saved = (gwl_no_intervention[-1] - gwl_with_intervention[-1])
ax1.text(250, 8, f'Water Level\nPreserved:\n{water_saved:.1f}m', 
         fontsize=11, fontweight='bold', 
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# Plot 2: Semi-Critical Region Scenario
ax2 = axes[0, 1]
ax2.plot(days, gwl_semi_no, 'orange', linewidth=2.5, linestyle='--', label='Without Intervention', alpha=0.7)
ax2.plot(days, gwl_semi_with, 'darkgreen', linewidth=2.5, label='With AI Monitoring')
ax2.axhline(y=10, color='darkred', linestyle=':', linewidth=2, label='Critical Threshold')
ax2.axvline(x=intervention_start, color='blue', linestyle='--', alpha=0.5, linewidth=1.5)

ax2.fill_between(days[intervention_start:], 
                 gwl_semi_no[intervention_start:], 
                 gwl_semi_with[intervention_start:], 
                 alpha=0.3, color='yellow')

ax2.set_xlabel('Days from Present', fontsize=11, fontweight='bold')
ax2.set_ylabel('Groundwater Depth (m below ground)', fontsize=11, fontweight='bold')
ax2.set_title('Semi-Critical Region (Maharashtra-Type)\nMaintaining Safe Levels', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.invert_yaxis()

water_saved_semi = (gwl_semi_no[-1] - gwl_semi_with[-1])
ax2.text(250, 11, f'Stabilized at\nSafe Level\n({water_saved_semi:.1f}m saved)', 
         fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Plot 3: Cost-Benefit Analysis
ax3 = axes[1, 0]

interventions = ['AI System\n(Annual)', 'Recharge\nStructures', 'Awareness\nCampaigns', 
                'Total\nIntervention\nCost', 'Cost of\nCrisis\n(No Action)']
costs = [2, 25, 8, 35, 150]  # Lakhs INR
colors_cost = ['steelblue', 'skyblue', 'lightblue', 'orange', 'red']

bars = ax3.bar(interventions, costs, color=colors_cost, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Cost (Lakhs INR)', fontsize=11, fontweight='bold')
ax3.set_title('Economic Impact Analysis\nROI of AI-Based Early Warning', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for bar, cost in zip(bars, costs):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'Rs.{cost}L', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Highlight ROI
ax3.annotate('', xy=(3.5, 35), xytext=(4.5, 150),
            arrowprops=dict(arrowstyle='<->', color='green', lw=3))
ax3.text(4, 92, 'ROI: 4.3x', fontsize=12, fontweight='bold', color='green',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Plot 4: Multi-Region Comparison
ax4 = axes[1, 1]

regions_comp = ['Punjab', 'Maharashtra', 'Bihar', 'Tamil Nadu', 'Rajasthan', 'UP']
current_status = [2.5, 3.5, 4.8, 5.2, 6.1, 3.8]  # Depletion risk score (0-10)
forecasted_1yr = [2.2, 3.0, 4.5, 4.5, 4.8, 3.2]  # With AI intervention

x = np.arange(len(regions_comp))
width = 0.35

bars1 = ax4.bar(x - width/2, current_status, width, label='Current Risk Level',
               color='#e74c3c', alpha=0.8, edgecolor='black')
bars2 = ax4.bar(x + width/2, forecasted_1yr, width, label='Projected (1 Year)\nwith AI Management',
               color='#2ecc71', alpha=0.8, edgecolor='black')

ax4.set_ylabel('Depletion Risk Score (0-10)', fontsize=11, fontweight='bold')
ax4.set_title('Regional Impact Projection\nRisk Reduction via AI Models', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(regions_comp, fontsize=9, rotation=45, ha='right')
ax4.legend(fontsize=9, loc='upper right')
ax4.grid(axis='y', alpha=0.3)

# Add improvement percentages
for i, (curr, proj) in enumerate(zip(current_status, forecasted_1yr)):
    improvement = ((curr - proj) / curr) * 100
    ax4.text(i, max(curr, proj) + 0.3, f'↓{improvement:.0f}%', 
            ha='center', fontsize=9, fontweight='bold', color='green')

print("✓ All plots created")

plt.tight_layout()

# Add info box
fig.text(0.5, 0.02, 
         'Models Used: ARIMA/SARIMA (90-day forecast) + ANN (recharge estimation) + Random Forest (risk classification)\n'
         'Intervention Strategy: Early alerts → Targeted recharge structures → Usage regulation → Community awareness',
         ha='center', fontsize=8, style='italic', color='gray',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

print("✓ Saving figure...")
plt.savefig('impact_scenarios.png', dpi=300, bbox_inches='tight')
print("="*60)
print("✓✓✓ SUCCESS! Saved: impact_scenarios.png")
print("="*60)
plt.close()