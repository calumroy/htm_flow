# Temporal Pooling Config Guide

This guide explains the temporal-pooling knobs used by configs such as
`configs/word_rows_2layer_delayed_temporal_pooling_text.yaml` and
`configs/overrides/word_rows_2layer_enable_temporal_pooling.yaml`.

## Recommended Starting Point

For Layer 1 delayed temporal pooling, start with:

```yaml
runtime_parameter_schedule:
  - at_timestep: 1000
    override: overrides/word_rows_2layer_enable_temporal_pooling.yaml

temporal_pooling:
  enabled: true
  enable_persistence: false
  delay_length: 16
  spatial_permanence_inc: 0.12
  active_predict_proximal_scale: 0.25
  post_active_proximal_scale: 0.5
  sequence_permanence_inc: 0.21
  sequence_permanence_dec: 0.002
```

This is meant to keep meaningful proximal temporal pooling while avoiding the
old burst spike. Headless true-burst diagnostics show that
the direct predicted-non-active proximal path was removed because it made
columns win before distal sequence memory predicted them. The post-active bridge
is safer once Layer 1 has enough cells per column.

For the current word-row Layer 1 abstraction config, use
`sequence_memory.cells_per_column: 6` and
`sequence_memory.activation_threshold: 4`. Four cells per column left familiar
branching transitions without enough cell contexts; threshold `6` left familiar
transitions with no previous prediction; threshold `3` made prediction too broad
and created excessive multi-cell predicted activity.

## What Each Knob Does

- `enabled`: turns temporal pooling on for the layer. In delayed configs, keep
  this `false` in the base config and enable it with a runtime override after
  sequence memory has had time to form.

- `enable_persistence`: lets TP keep cells predictive for a short window.
  Keep this `false` while tuning bursting. It can help continuity, but it is
  easier to make stale predictions that increase bursting pressure.

- `delay_length`: smooths the persistence estimate. It only matters when
  `enable_persistence` is true. With persistence off, it is not the main tuning
  knob.

- `spatial_permanence_inc`: proximal TP learning rate. This is the main knob for
  making predicted columns start to compete by overlap. Increase it slowly. Too
  low means TP predicts distally but does not change winners. Too high can make
  a small set of columns sticky and burst-prone.

- `active_predict_proximal_scale`: multiplier for proximal reinforcement on
  columns that were active and correctly predicted. This is the safest TP
  proximal path because the column already avoided bursting. The proximal
  permanence increment for an active input synapse is
  `spatial_permanence_inc * active_predict_count * active_predict_proximal_scale`.
  Values below `1.0` damp the base increment; values above `1.0` amplify it.

- `post_active_proximal_scale`: multiplier for the one-step bridge after a
  correctly predicted activation. This is useful for extending proximal support
  across nearby sequence inputs, but keep an eye on `true_burst_fraction`. The
  extra proximal increment is
  `spatial_permanence_inc * post_active_proximal_scale` for active input
  synapses on a column that was active-predict one timestep ago and remains
  segment-backed predictive now.

- `sequence_permanence_inc`: distal TP learning rate. This controls how quickly
  TP-created distal synapses become useful for prediction. This should usually
  be stronger than `spatial_permanence_inc`.

- `sequence_permanence_dec`: distal decay for non-matching TP synapses. Raise it
  if old distal context sticks around too long. Lower it if useful TP segments
  fail to stabilize.

## Tuning Order

1. Keep `enable_persistence: false`.
2. Tune `sequence_permanence_inc` until predictive coverage improves.
3. Increase `spatial_permanence_inc` gradually with a conservative
   `active_predict_proximal_scale`.
4. Add `post_active_proximal_scale` if active-predict columns need
   more continuity and true bursts remain low.
5. Watch Layer 1 bursting. If bursting rises, back off `post_active_proximal_scale`
   before changing persistence.
6. Only test `enable_persistence: true` after non-persistence TP is stable.

## What To Watch

Useful signs:

- Layer 1 predictive coverage increases.
- `reinforced_inputs` is non-zero after TP turns on.
- active columns broaden beyond the old saturated winner set.
- bursting does not rise sharply compared with TP-off or pre-TP behavior.
- `true_burst_fraction` and `new_true_burst_fraction` stay low in headless CLI
  metrics.

Bad signs:

- `reinforced_inputs` is non-zero but active columns do not change:
  `spatial_permanence_inc` or the non-active proximal scales may be too small.
- `true_burst_fraction` rises while `new_true_burst_causes` reports
  `no_prev_prediction`:
  proximal TP is making unpredicted columns win; reduce
  `spatial_permanence_inc` or `post_active_proximal_scale`. If proximal TP is
  already weak or disabled, lower `sequence_memory.activation_threshold`
  carefully.
- a small set of columns dominates every timestep:
  `spatial_permanence_inc` or `post_active_proximal_scale` may be too high, or
  Layer 1 spatial pooling is too collapsed before TP turns on.
- bursting rises after enabling persistence:
  turn persistence back off and tune distal/proximal learning first.

## Current Caveat

The current implementation reduces the old failure mode by requiring local
distal evidence before TP changes proximal permanence. It does not eliminate
bursting. The goal is maximum temporal pooling without a burst spike, not simply
maximum TP learning rates.
