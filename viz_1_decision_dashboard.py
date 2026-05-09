"""
VISUALIZATION 1: DECISION SUPPORT DASHBOARD
Shows: Real-world utility - What actions to take based on model predictions
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Sample data representing model outputs for different regions
regions = ['Punjab-Ludhiana', 'Maharashtra-Pune', 'Bihar-Patna', 
           'Tamil Nadu-Chennai', 'Rajasthan-Jaipur', 'UP-Lucknow']

# Model predictions (you'll replace with your actual model outputs)
current_gwl = np.array([12.5, 8.3, 15.2, 6.1, 4.2, 10.8])  # Current water level (m)
predicted_gwl_90d = np.array([11.2, 7.1, 14.8, 4.5, 2.8, 9.5])  # 90-day ARIMA forecast
recharge_rate = np.array([45, 28, 52, 18, 12, 38])  # ANN recharge estimate (mm/month)
risk_status = ['Safe', 'Semi-Critical', 'Safe', 'Critical', 'Critical', 'Semi-Critical']  # RF classification

# Calculate intervention priority score (0-100)
gwl_change = current_gwl - predicted_gwl_90d
priority_score = (
    (1 - current_gwl/20) * 30 +  # Lower current level = higher priority
    (gwl_change/5) * 40 +  # Faster depletion = higher priority
    (1 - recharge_rate/60) * 30  # Lower recharge = higher priority
) * 100
priority_score = np.clip(priority_score, 0, 100)

# Create dashboard
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Groundwater Management Decision Dashboard\nAI-Driven Action Recommendations', 
             fontsize=18, fontweight='bold', y=0.98)

# Color mapping
status_colors = {'Safe': '#2ecc71', 'Semi-Critical': '#f39c12', 'Critical': '#e74c3c'}
colors = [status_colors[s] for s in risk_status]

# Plot 1: Intervention Priority Heatmap
ax1 = plt.subplot(2, 3, 1)
sorted_idx = np.argsort(priority_score)[::-1]
bars = ax1.barh(range(len(regions)), priority_score[sorted_idx], 
                color=[colors[i] for i in sorted_idx], edgecolor='black', linewidth=1.5)
ax1.set_yticks(range(len(regions)))
ax1.set_yticklabels([regions[i] for i in sorted_idx], fontsize=10)
ax1.set_xlabel('Intervention Priority Score', fontsize=11, fontweight='bold')
ax1.set_title('A. Priority Ranking for Action', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3, linestyle='--')

# Add priority labels
for i, (idx, score) in enumerate(zip(sorted_idx, priority_score[sorted_idx])):
    ax1.text(score + 2, i, f'{score:.0f}', va='center', fontsize=9, fontweight='bold')

# Plot 2: 90-Day Forecast Alert System
ax2 = plt.subplot(2, 3, 2)
x_pos = np.arange(len(regions))
width = 0.35

bars1 = ax2.bar(x_pos - width/2, current_gwl, width, label='Current Level', 
                color='steelblue', alpha=0.8, edgecolor='black')
bars2 = ax2.bar(x_pos + width/2, predicted_gwl_90d, width, label='Predicted (90 days)', 
                color='coral', alpha=0.8, edgecolor='black')

# Add threshold line
ax2.axhline(y=5, color='red', linestyle='--', linewidth=2, label='Critical Threshold (5m)')

ax2.set_ylabel('Groundwater Level (m below ground)', fontsize=11, fontweight='bold')
ax2.set_title('B. Early Warning System\n(ARIMA/SARIMA Forecast)', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([r.split('-')[0] for r in regions], rotation=45, ha='right', fontsize=9)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add alert markers
for i, (curr, pred) in enumerate(zip(current_gwl, predicted_gwl_90d)):
    if pred < 5:
        ax2.text(i, pred - 0.5, '⚠', fontsize=16, ha='center', color='red')

# Plot 3: Recharge Potential Analysis
ax3 = plt.subplot(2, 3, 3)
scatter = ax3.scatter(current_gwl, recharge_rate, s=priority_score*5, 
                     c=colors, alpha=0.7, edgecolors='black', linewidth=2)

for i, region in enumerate(regions):
    ax3.annotate(region.split('-')[1], (current_gwl[i], recharge_rate[i]), 
                fontsize=8, ha='center', va='bottom')

ax3.set_xlabel('Current Groundwater Level (m)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Recharge Rate (mm/month)', fontsize=11, fontweight='bold')
ax3.set_title('C. Recharge vs Depletion Analysis\n(ANN Estimation)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')

# Add quadrants
ax3.axhline(y=30, color='gray', linestyle='--', alpha=0.5)
ax3.axvline(x=8, color='gray', linestyle='--', alpha=0.5)
ax3.text(15, 55, 'Safe Zone', fontsize=10, alpha=0.6, fontweight='bold')
ax3.text(3, 55, 'High Risk', fontsize=10, alpha=0.6, fontweight='bold', color='red')

# Plot 4: Recommended Actions Table
ax4 = plt.subplot(2, 3, 4)
ax4.axis('off')

# Create action recommendations
actions = []
for i in sorted_idx[:6]:
    region = regions[i]
    if risk_status[i] == 'Critical':
        action = '🚨 Immediate ban on new borewells\n   + Mandatory recharge structures'
    elif risk_status[i] == 'Semi-Critical':
        action = '⚠️ Regulate water extraction\n   + Promote rainwater harvesting'
    else:
        action = '✓ Monitor quarterly\n   + Maintain current practices'
    
    actions.append([region.split('-')[1], risk_status[i], f'{priority_score[i]:.0f}', action])

# Create table
table_data = [['Region', 'Status', 'Priority', 'Recommended Action']] + actions[:4]
table = ax4.table(cellText=table_data, cellLoc='left', loc='center',
                 colWidths=[0.15, 0.2, 0.12, 0.53])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Style header
for i in range(4):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color code rows by status
for i in range(1, 5):
    status = table_data[i][1]
    table[(i, 1)].set_facecolor(status_colors[status])
    table[(i, 1)].set_text_props(weight='bold', color='white')

ax4.set_title('D. AI-Generated Action Plan', fontsize=12, fontweight='bold', pad=20)

# Plot 5: Seasonal Recharge Pattern
ax5 = plt.subplot(2, 3, 5)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# Simulated seasonal pattern (replace with actual model predictions)
avg_recharge = [15, 12, 10, 8, 6, 35, 85, 95, 70, 25, 18, 16]
ax5.fill_between(range(12), avg_recharge, alpha=0.3, color='skyblue')
ax5.plot(range(12), avg_recharge, marker='o', linewidth=2, markersize=8, color='steelblue')

# Highlight monsoon
ax5.axvspan(5.5, 8.5, alpha=0.2, color='blue', label='Monsoon Period')

ax5.set_xlabel('Month', fontsize=11, fontweight='bold')
ax5.set_ylabel('Average Recharge Rate (mm/month)', fontsize=11, fontweight='bold')
ax5.set_title('E. Seasonal Recharge Pattern\n(Multi-Year ANN Analysis)', fontsize=12, fontweight='bold')
ax5.set_xticks(range(12))
ax5.set_xticklabels(months, fontsize=9)
ax5.grid(True, alpha=0.3, linestyle='--')
ax5.legend(loc='upper right', fontsize=9)

# Plot 6: Risk Distribution Map
ax6 = plt.subplot(2, 3, 6)
status_counts = {'Safe': 0, 'Semi-Critical': 0, 'Critical': 0}
for s in risk_status:
    status_counts[s] += 1

wedges, texts, autotexts = ax6.pie(status_counts.values(), 
                                     labels=status_counts.keys(),
                                     colors=[status_colors[k] for k in status_counts.keys()],
                                     autopct='%1.1f%%',
                                     startangle=90,
                                     explode=[0.05 if k=='Critical' else 0 for k in status_counts.keys()],
                                     textprops={'fontsize': 11, 'fontweight': 'bold'})

ax6.set_title('F. Regional Risk Distribution\n(Random Forest Classification)', 
             fontsize=12, fontweight='bold')

# Add legend with counts
legend_labels = [f"{k}: {v} regions" for k, v in status_counts.items()]
ax6.legend(legend_labels, loc='lower left', fontsize=9)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])

# Add footer
fig.text(0.5, 0.01, 'Dashboard integrates: ARIMA/SARIMA forecasting + ANN recharge estimation + Random Forest risk classification', 
         ha='center', fontsize=9, style='italic', color='gray')

plt.savefig('decision_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Saved: decision_dashboard.png")
plt.show()