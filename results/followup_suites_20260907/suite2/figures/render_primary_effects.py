"""Render the frozen primary estimates without model inference."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'analysis_final/results.json'
raw = SOURCE.read_bytes()
result = json.loads(raw)
assert result['coverage']['generated_responses'] == 29812
assert result['coverage']['missing_responses'] == 0
keys = ['order_advantage', 'suppression', 'excess_suppression']
names = ['Later-installation\nadvantage', 'Loss after\nrival continuation',
         'Ordinary minus\nrival support']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'svg.fonttype': 'none', 'svg.hashsalt': 'slc-suite2-primary-20260907',
                     'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.spines.left': False})
fig, ax = plt.subplots(figsize=(9.6, 4.8))
fig.subplots_adjust(left=.25, right=.97, top=.80, bottom=.25)
for y, key in zip([2, 1, 0], keys):
    effect = result['effects']['consensus'][key]['pooled']
    lower, upper = [100 * value for value in effect['interval_adjusted']]
    low_bound, high_bound = [100 * value for value in effect['bounds']]
    assert lower <= low_bound <= high_bound <= upper
    ax.hlines(y, lower, upper, color='#384758', linewidth=1.8)
    ax.vlines([lower, upper], y-.065, y+.065, color='#384758', linewidth=1.8)
    ax.plot([low_bound, high_bound], [y, y], color='#146b89', linewidth=8,
            solid_capstyle='butt')
    ax.text(upper+.9, y, f'{lower:.2f} to {upper:.2f}', va='center', fontsize=9,
            color='#384758')
ax.axvline(0, color='#9c6c4d', linestyle='--', linewidth=1)
ax.set_yticks([2, 1, 0], names)
ax.tick_params(axis='y', length=0, pad=13)
ax.set_xlim(-25, 63)
ax.set_ylim(-.6, 2.6)
ax.set_xticks([-20, -10, 0, 10, 20, 30, 40, 50, 60])
ax.set_xlabel('Effect (percentage points)', labelpad=10)
ax.grid(axis='x', color='#e4e8ec', linewidth=.6)
ax.set_axisbelow(True)
fig.suptitle('Suite 2: primary consensus estimates', x=.035, ha='left', y=.965,
             fontsize=17, fontweight='bold')
fig.text(.035, .883, 'All three adjusted intervals include zero.', fontsize=11,
         color='#384758')
legend = [Line2D([0], [0], color='#146b89', linewidth=8, solid_capstyle='butt'),
          Line2D([0], [0], color='#384758', linewidth=1.8)]
fig.legend(legend, ['Bounds from unresolved labels', 'Adjusted 98.333% interval'],
           loc='lower left', bbox_to_anchor=(.245, .06), ncol=2, frameon=False,
           fontsize=10, handlelength=2.4)
fig.text(.035, .022, 'Four paired training seeds; 20,000 bootstrap draws. Thick bars are bounds, not point estimates.',
         fontsize=9, color='#526170')
outputs = []
for suffix in ['png', 'svg']:
    path = HERE / f'primary_effects.{suffix}'
    metadata = {'Date': None} if suffix == 'svg' else {}
    fig.savefig(path, dpi=180, facecolor='white', metadata=metadata)
    outputs.append(path)
plt.close(fig)
manifest = {'source_sha256': hashlib.sha256(raw).hexdigest(),
            'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'matplotlib': matplotlib.__version__,
            'outputs': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs},
            'scope': 'Display of frozen consensus primary estimates; no new analysis or inference.'}
(HERE / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2)+'\n')
print(json.dumps(manifest, sort_keys=True))
