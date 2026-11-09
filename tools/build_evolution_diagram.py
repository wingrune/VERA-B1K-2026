#!/usr/bin/env python3
"""Build the planner-method figure as editable SVG and a print-resolution PNG."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ('01', 'Build the Isaac digital twin', [
        'Convert the task scene, objects and R1Pro',
        'Match cameras, controls and task checks',
        'GPU physics + fast RGB-D rendering']),
    ('02', 'Implement a seed planner', [
        'RGB-D perception and geometric estimates',
        'Navigation, grasp and manipulation stages',
        'IK and motion planning where needed']),
    ('03', 'Propose planner candidates', [
        'LLM edits to planner logic and recovery',
        'Algorithmic code / parameter variations',
        'Candidate = program + configuration']),
    ('04', 'Run GPU evaluation rollouts', [
        'Execute each candidate in a closed loop',
        'Render observations; apply robot actions',
        'Evaluate across task instances / seeds']),
    ('05', 'Score and diagnose failures', [
        'Measure task progress / Q and success',
        'Inspect timeouts, actions and failure traces',
        'Identify infrastructure failures separately']),
    ('06', 'Select and iterate', [
        'Retain promising planner candidates',
        'Use failed rollouts to guide the next round',
        'Repeat proposal, evaluation and selection']),
    ('07', 'Freeze a planner artifact', [
        'Select before native validation',
        'Freeze planner code and parameter values',
        'Record program ID, source and SHA256']),
    ('08', 'Validate in native BEHAVIOR-1K', [
        'Test exact policy image and runtime',
        'Fresh episodes; public instances 301-310',
        'Check native Q / success, JSON and video']),
    ('09', 'Package and release', [
        'Ship planner, priors and dependencies',
        'Keep provenance and episode evidence',
        'Publish the reproduction instructions']),
]


def build(compact=False):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(13.8, 11.7))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 1380), ylim=(0, 1170))
    ax.axis('off')
    paper, ink, orange, muted = '#f6f4ed', '#252820', '#ce4a25', '#62685a'
    fig.patch.set_facecolor(paper)
    ax.set_facecolor(paper)
    ax.text(55, 1114, 'VERA-Volve / Planner evolution', fontsize=27, color=ink, weight='bold')
    ax.text(55, 1075, 'A fast search environment, a feedback loop, and a separate native validation gate.', fontsize=14, color=muted)
    positions = [(55, 845), (505, 845), (955, 845), (955, 535), (505, 535), (55, 535),
                 (55, 205), (505, 205), (955, 205)]
    width, height = 370, 170
    steps = STEPS
    if compact:
        titles = ['Isaac digital twin', 'Seed planner', 'Candidate proposals', 'GPU rollouts',
                  'Fitness and failures', 'Selection', 'Freeze planner', 'Native validation', 'Release']
        details = [('Task / robot conversion', 'GPU physics + RGB-D'),
                   ('Perception and task stages', 'IK + motion planning'),
                   ('LLM code edits', 'Algorithmic variations'),
                   ('Closed-loop execution', 'Multiple instances / seeds'),
                   ('Task progress / Q + success', 'Failure traces and diagnostics'),
                   ('Keep promising candidates', 'Feed results back to step 03'),
                   ('Code + parameter values', 'Program ID + source hash'),
                   ('Exact policy image', 'Native JSON + video evidence'),
                   ('Code, priors and runtime', 'Artifacts + instructions')]
        steps = [(str(i+1).zfill(2), title, lines) for i, (title, lines) in enumerate(zip(titles, details))]
    for (number, title, lines), (x, y) in zip(steps, positions):
        is_search = number in {'03', '04', '05', '06'}
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle='round,pad=0,rounding_size=6',
            facecolor='#fff8f1' if is_search else '#fffef8', edgecolor=orange if is_search else '#c9cdbf', linewidth=1.4))
        ax.text(x+18, y+139, number, fontsize=13, color=orange, weight='bold')
        ax.text(x+18, y+108, title, fontsize=18 if compact else 12.8, color=ink, weight='bold')
        for i, line in enumerate(lines):
            ax.text(x+18, y+78-i*25, line, fontsize=15 if compact else 10.7, color=muted)
    def arrow(start, end, **kwargs):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=17,
            color=orange, linewidth=1.8, **kwargs))
    for row, direction in [(0, 1), (1, -1), (2, 1)]:
        y = positions[row*3][1] + height/2
        if direction == 1:
            arrow((429, y), (499, y)); arrow((879, y), (949, y))
        else:
            arrow((949, y), (879, y)); arrow((499, y), (429, y))
    arrow((1140, 840), (1140, 712))
    arrow((240, 529), (240, 382))
    # Return from selection to candidate generation, alongside the main path.
    ax.plot([240, 240, 1353, 1353], [710, 765, 765, 930], color=orange, linewidth=1.4, linestyle=(0, (5, 4)))
    arrow((1353, 930), (1328, 930), linestyle=(0, (5, 4)))
    ax.text(380, 782, 'FEEDBACK: revise planner code and search parameters', fontsize=11.5, color=orange)
    ax.text(505, 468, 'TRANSFER BOUNDARY', fontsize=11.5, color=orange, weight='bold')
    ax.text(505, 439, 'Search scores and native evaluator results are separate measurements.', fontsize=12.2, color=muted)
    ax.text(55, 147, 'Runtime constraint: the planner acts on permitted onboard observations and static priors.', fontsize=13, color=ink)
    ax.text(55, 115, 'Task-state checks score rollouts; they are not an oracle input to the deployed planner.', fontsize=12.5, color=muted)
    ax.text(55, 66, 'If native validation reveals a failure, revise the candidate, freeze a new version, and validate again.', fontsize=11.7, color=muted)
    output = ROOT / ('docs/assets/evolution-method-print' if compact else 'docs/assets/evolution-method')
    fig.savefig(output.with_suffix('.svg'), facecolor=paper)
    fig.savefig(output.with_suffix('.png'), dpi=200, facecolor=paper)
    plt.close(fig)
    print(output.with_suffix('.svg'))


if __name__ == '__main__':
    build()
    build(compact=True)
