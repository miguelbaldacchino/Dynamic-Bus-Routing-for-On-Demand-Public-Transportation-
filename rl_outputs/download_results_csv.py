import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# All 45 policies with category assignment
frontier_set = {
    'greedy+alns', 'rl+ga (rl_v6)', 'rl+ts (rl_v6)',
    'rl+ts (rl_v5)', 'rl+alns (rl_v4)', 'rl+ts (rl_v4)', 'rl+ts (rl_v3)'
}

rl11_set = {
    'rl (rl_v3)', 'rl (rl_v3ant)',
    'rl+sa (rl_v3)', 'rl+sa (rl_v3ant)',
    'rl+ts (rl_v3ant)',
    'rl+ga (rl_v3)', 'rl+ga (rl_v3ant)',
    'rl+alns (rl_v3)', 'rl+alns (rl_v3ant)'
}

# Frontier colours matching parallel coordinates
frontier_colours = {
    'rl+ts (rl_v4)':    '#2171b5',  # RL2.0 blue
    'greedy+alns':      '#238b45',  # green
    'rl+ts (rl_v3)':    '#d7301f',  # red
    'rl+alns (rl_v4)':  '#6a51a3',  # purple
    'rl+ts (rl_v5)':    '#c2185b',  # pink
    'rl+ts (rl_v6)':    '#b45309',  # orange
    'rl+ga (rl_v6)':    '#636363',  # gray
}

all_policies = [
    ('greedy', 75.6, 15.01),
    ('greedy+sa', 77.4, 14.93),
    ('greedy+ts', 78.9, 13.77),
    ('greedy+ga', 78.7, 13.81),
    ('greedy+alns', 79.9, 13.67),
    ('rl (rl_base)', 74.9, 15.65),
    ('rl (rl_v3)', 63.3, 11.73),
    ('rl (rl_v3ant)', 75.3, 16.33),
    ('rl (rl_v4)', 71.2, 14.55),
    ('rl (rl_v5)', 73.8, 15.86),
    ('rl (rl_v4ant)', 76.0, 16.79),
    ('rl (rl_v5ant)', 75.6, 16.54),
    ('rl (rl_v6)', 74.8, 17.13),
    ('rl+sa (rl_base)', 76.2, 14.93),
    ('rl+sa (rl_v3)', 64.4, 10.78),
    ('rl+sa (rl_v3ant)', 78.4, 15.49),
    ('rl+sa (rl_v4)', 73.9, 14.12),
    ('rl+sa (rl_v5)', 76.7, 15.58),
    ('rl+sa (rl_v4ant)', 77.6, 15.65),
    ('rl+sa (rl_v5ant)', 78.0, 15.52),
    ('rl+sa (rl_v6)', 76.7, 15.37),
    ('rl+ts (rl_base)', 77.9, 13.45),
    ('rl+ts (rl_v3)', 68.0, 8.71),
    ('rl+ts (rl_v3ant)', 78.7, 13.73),
    ('rl+ts (rl_v4)', 76.2, 12.09),
    ('rl+ts (rl_v5)', 77.8, 13.35),
    ('rl+ts (rl_v4ant)', 78.9, 14.20),
    ('rl+ts (rl_v5ant)', 78.7, 13.85),
    ('rl+ts (rl_v6)', 78.1, 13.40),
    ('rl+ga (rl_base)', 78.1, 13.45),
    ('rl+ga (rl_v3)', 67.4, 9.05),
    ('rl+ga (rl_v3ant)', 78.5, 13.91),
    ('rl+ga (rl_v4)', 75.9, 12.23),
    ('rl+ga (rl_v5)', 78.1, 13.69),
    ('rl+ga (rl_v4ant)', 78.2, 13.87),
    ('rl+ga (rl_v5ant)', 78.5, 13.78),
    ('rl+ga (rl_v6)', 78.2, 13.61),
    ('rl+alns (rl_base)', 79.5, 13.81),
    ('rl+alns (rl_v3)', 64.4, 10.30),
    ('rl+alns (rl_v3ant)', 79.7, 14.38),
    ('rl+alns (rl_v4)', 77.6, 13.03),
    ('rl+alns (rl_v5)', 79.0, 14.16),
    ('rl+alns (rl_v4ant)', 79.8, 14.55),
    ('rl+alns (rl_v5ant)', 79.7, 14.49),
    ('rl+alns (rl_v6)', 79.1, 13.94),
]

fig, ax = plt.subplots(figsize=(5.5, 3.4))

# Unified node size for all scatter plots
NODE_SIZE = 30  

# Layer 1: other policies
for name, svc, wait in all_policies:
    if name not in frontier_set and name not in rl11_set:
        ax.scatter(svc, wait, color='#cccccc', s=NODE_SIZE, zorder=2, linewidths=0)

# Layer 2: rl1.1 family — orange, square marker only for non-ant variants
for name, svc, wait in all_policies:
    if name in rl11_set:
         # Use square 's' for pure v3, circle 'o' for v3ant (-a variants)
        marker = 'o' if 'ant' in name else 's' 
        ax.scatter(svc, wait, color='#f16913', s=NODE_SIZE, zorder=3,
                   marker=marker, linewidths=0.3, edgecolors='white')

# Layer 3: frontier — coloured circles
frontier_sorted_by_svc = sorted(
    [(n,s,w) for n,s,w in all_policies if n in frontier_set],
    key=lambda x: x[1])

for name, svc, wait in frontier_sorted_by_svc:
    col = frontier_colours[name]
    ax.scatter(svc, wait, color=col, s=NODE_SIZE, zorder=5,
               linewidths=0.5, edgecolors='white')

# Frontier curve (staircase connecting frontier points) — updated to thick red line
fx = [s for _,s,_ in frontier_sorted_by_svc]
fy = [w for _,_,w in frontier_sorted_by_svc]
ax.plot(fx, fy, color='red', lw=2.5, linestyle='-', zorder=4)

# Chord: rank7 (rl+ts rl_v3, 68.0, 8.71) to rank1 (greedy+alns, 79.9, 13.67)
ax.plot([68.0, 79.9], [8.71, 13.67], color='#555555', lw=1.3,
        linestyle='--', zorder=4, alpha=0.85)

ax.set_xlabel('Service rate (%)', fontsize=9, color='#333333')
ax.set_ylabel('Mean wait (min)', fontsize=9, color='#333333')
ax.tick_params(labelsize=8, colors='#555555')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#dddddd')
ax.spines['bottom'].set_color('#dddddd')
ax.tick_params(axis='both', which='both', color='#dddddd')

# Matched legend elements to uniform sizes
legend_elements = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#2171b5', markersize=6, label='RL+TS (RL2.0)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#238b45', markersize=6, label='Greedy+ALNS'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#d7301f', markersize=6, label='RL+TS (RL1.1)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#6a51a3', markersize=6, label='RL+ALNS (RL2.0)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#c2185b', markersize=6, label='RL+TS (RL2.1)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#b45309', markersize=6, label='RL+TS (RL2.2)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#636363', markersize=6, label='RL+GA (RL2.2)'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor='#f16913', markersize=6, label='RL1.1 variants'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#cccccc', markersize=6, label='Other policies'),
    Line2D([0],[0], color='#555555', lw=1.3, linestyle='--', label='Frontier chord'),
]

ax.legend(handles=legend_elements, loc='upper left',
          bbox_to_anchor=(1.02, 1.02), fontsize=7,
          frameon=False, handlelength=1.5,
          handletextpad=0.5, borderpad=0, labelspacing=0.4)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/pareto_scatter.pdf', bbox_inches='tight', dpi=300)
plt.savefig('/mnt/user-data/outputs/pareto_scatter.png', bbox_inches='tight', dpi=300)
print('done')