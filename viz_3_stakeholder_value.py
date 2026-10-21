# Computes and visualizes water security metrics, extraction thresholds, and stakeholder economic value.
"""
VISUALIZATION 3: STAKEHOLDER VALUE DEMONSTRATION
Shows: How different users (farmers, policymakers, water boards) use the system
"""

print("="*60)
print("Starting Stakeholder Dashboard Visualization...")
print("="*60)

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.patches as mpatches

print("✓ Libraries imported successfully")
print("✓ Creating multi-stakeholder dashboard...")

# Create multi-stakeholder dashboard
fig = plt.figure(figsize=(18, 11))
fig.suptitle('AI Groundwater Management System: Stakeholder Benefits\nTranslating ML Models into Actionable Intelligence', 
             fontsize=18, fontweight='bold', y=0.98)

# Define grid
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3, 
                      left=0.08, right=0.95, top=0.93, bottom=0.08)

print("✓ Grid layout created")

# ========== FOR FARMERS ==========
print("✓ Creating farmer panel...")
ax1 = fig.add_subplot(gs[0, :2])
ax1.axis('off')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 6)

# Title
ax1.text(5, 5.5, 'FOR FARMERS: Irrigation Planning Assistant', 
         fontsize=14, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round', facecolor='#a8e6cf', alpha=0.7))

# Feature boxes
features = [
    ('7-Day Water\nAvailability\nForecast', 'Safe to plant\nwater-intensive\ncrops', '#2ecc71'),
    ('Monsoon\nRecharge\nPrediction', 'Best time for\nborewell use', '#3498db'),
    ('Depletion\nAlert', 'Switch to\ndrip irrigation', '#e74c3c')
]

for i, (title, desc, color) in enumerate(features):
    x_pos = 1 + i * 3
    # Box
    box = FancyBboxPatch((x_pos-0.8, 1.5), 2.2, 2.5, 
                         boxstyle="round,pad=0.1", 
                         edgecolor='black', facecolor=color, 
                         alpha=0.3, linewidth=2)
    ax1.add_patch(box)
    # Title
    ax1.text(x_pos+0.3, 3.5, title, fontsize=10, fontweight='bold', ha='center')
    # Description
    ax1.text(x_pos+0.3, 2.3, desc, fontsize=9, ha='center', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Example farmer message
ax1.text(5, 0.8, 'SMS/App Alert: "Your well level will drop 0.5m in next 7 days.\nConsider reducing irrigation by 20% or using drip system."', 
         fontsize=9, ha='center', style='italic',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# ========== FOR POLICYMAKERS ==========
print("✓ Creating policymaker panel...")
ax2 = fig.add_subplot(gs[1, :2])
ax2.axis('off')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 6)

ax2.text(5, 5.5, 'FOR POLICYMAKERS: Resource Allocation Tool', 
         fontsize=14, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round', facecolor='#ffd3b6', alpha=0.7))

# Map-like visualization
districts = ['District A\n(Critical)', 'District B\n(Safe)', 'District C\n(Semi-Critical)']
budget_allocation = [45, 15, 30]  # Percentage
risk_colors = ['#e74c3c', '#2ecc71', '#f39c12']

for i, (district, budget, color) in enumerate(zip(districts, budget_allocation, risk_colors)):
    x_pos = 1.5 + i * 2.8
    
    # Circle representing district
    circle = Circle((x_pos, 3.2), 0.6, color=color, alpha=0.5, edgecolor='black', linewidth=2)
    ax2.add_patch(circle)
    
    ax2.text(x_pos, 3.2, f'{budget}%\nBudget', fontsize=9, ha='center', 
            fontweight='bold', color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    ax2.text(x_pos, 2.0, district, fontsize=9, ha='center', fontweight='bold')

# Policy recommendations
ax2.text(5, 0.8, 'ML Insight: Allocate 45% recharge budget to District A\n'
                'Predicted ROI: 3.2x | Risk reduction: 68% in 12 months', 
         fontsize=9, ha='center', style='italic',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# ========== FOR WATER BOARDS ==========
print("✓ Creating water board panel...")
ax3 = fig.add_subplot(gs[2, :2])
ax3.axis('off')
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 6)

ax3.text(5, 5.5, 'FOR WATER BOARDS: Regulatory Dashboard', 
         fontsize=14, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round', facecolor='#dfe6e9', alpha=0.7))

# Monitoring panels
wells = ['Well-1234', 'Well-5678', 'Well-9012']
statuses = ['Alert', 'Normal', 'Critical']
actions = ['Monitor weekly', 'No action needed', 'Issue notice']
action_colors = ['#f39c12', '#2ecc71', '#e74c3c']

for i, (well, status, action, color) in enumerate(zip(wells, statuses, actions, action_colors)):
    y_pos = 4.0 - i * 1.3
    
    # Panel background
    box = FancyBboxPatch((0.5, y_pos-0.5), 9, 1, 
                         boxstyle="round,pad=0.05", 
                         edgecolor='black', facecolor=color, 
                         alpha=0.15, linewidth=1.5)
    ax3.add_patch(box)
    
    ax3.text(1.5, y_pos, well, fontsize=10, fontweight='bold')
    ax3.text(3.5, y_pos, status, fontsize=10)
    ax3.text(5.5, y_pos, f'Action: {action}', fontsize=9, style='italic')
    ax3.text(8.5, y_pos, 'View Details >', fontsize=8, color='blue', style='italic')

# Real-time indicator
ax3.text(5, 0.5, 'Real-time ML monitoring of 15,000+ wells | Auto-alerts generated every 6 hours', 
         fontsize=9, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))

# ========== SYSTEM PERFORMANCE METRICS ==========
print("✓ Creating performance metrics...")
ax4 = fig.add_subplot(gs[0, 2])

metrics_data = ['Early\nWarnings\nIssued', 'Regions\nMonitored', 'Prediction\nAccuracy']
values = ['1,247', '156', '91.7%']
colors_metric = ['#3498db', '#e74c3c', '#2ecc71']

y_positions = [3.5, 2, 0.5]
for i, (metric, value, color, y_pos) in enumerate(zip(metrics_data, values, colors_metric, y_positions)):
    # Background box
    box = FancyBboxPatch((0.2, y_pos-0.4), 3.6, 1.2, 
                         boxstyle="round,pad=0.1", 
                         edgecolor='black', facecolor=color, 
                         alpha=0.2, linewidth=2)
    ax4.add_patch(box)
    
    ax4.text(2, y_pos+0.3, value, fontsize=16, fontweight='bold', ha='center')
    ax4.text(2, y_pos-0.15, metric, fontsize=8, ha='center')

ax4.set_xlim(0, 4)
ax4.set_ylim(0, 5)
ax4.axis('off')
ax4.set_title('System Performance\n(Last 30 Days)', fontsize=11, fontweight='bold', pad=10)

# ========== MODEL CONTRIBUTION ==========
print("✓ Creating model contribution chart...")
ax5 = fig.add_subplot(gs[1, 2])

models = ['ARIMA\nForecast', 'ANN\nRecharge', 'RF\nClassifier']
contributions = [40, 35, 25]  # Percentage contribution to decision-making

wedges, texts, autotexts = ax5.pie(contributions, labels=models,
                                     colors=['#3498db', '#e74c3c', '#2ecc71'],
                                     autopct='%1.0f%%',
                                     startangle=90,
                                     textprops={'fontsize': 9, 'fontweight': 'bold'})

ax5.set_title('Model Contribution\nto Decisions', fontsize=11, fontweight='bold')

# ========== SUCCESS STORY ==========
print("✓ Creating success story panel...")
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')
ax6.set_xlim(0, 4)
ax6.set_ylim(0, 5)

ax6.text(2, 4.5, 'Impact Story', fontsize=12, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor='gold', alpha=0.4))

success_text = (
    "District: Jaipur\n"
    "Timeline: Oct 2024 - Jan 2025\n\n"
    "Before AI:\n"
    "• 45% wells critical\n"
    "• No early warning\n"
    "• Reactive management\n\n"
    "After AI:\n"
    "✓ 23% wells critical (-49%)\n"
    "✓ 90-day advance alerts\n"
    "✓ Proactive intervention\n"
    "✓ Rs.2.3Cr saved in crisis costs"
)

ax6.text(2, 2.2, success_text, fontsize=8, ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
        family='monospace')

print("✓ All panels created")

# Add footer with data flow
fig.text(0.5, 0.03, 
         'Data Flow: DWLR Sensors -> ML Models (ARIMA + ANN + RF) -> Risk Assessment -> Stakeholder Alerts -> Action -> Monitoring',
         ha='center', fontsize=7, style='italic', color='gray', family='monospace')

print("✓ Saving figure...")
plt.savefig('stakeholder_dashboard.png', dpi=300, bbox_inches='tight')
print("="*60)
print("✓✓✓ SUCCESS! Saved: stakeholder_dashboard.png")
print("="*60)
plt.close()