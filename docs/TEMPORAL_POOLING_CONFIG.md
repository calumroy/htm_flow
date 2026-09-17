# Temporal Pooling Config Guide

This guide explains the temporal-pooling settings used by configs such as
`configs/word_rows_2layer_delayed_temporal_pooling_text.yaml` and
`configs/overrides/word_rows_2layer_enable_temporal_pooling.yaml`.

## Recommended Starting Point

For delayed temporal pooling in Layer 1, start with:

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

This starts temporal pooling after sequence memory has had 1000 timesteps to
learn. It keeps persistence off while the other settings are tuned.

For the current word-row Layer 1 config, use
`sequence_memory.cells_per_column: 6` and
`sequence_memory.activation_threshold: 4`. Four cells per column left familiar
branching sequences with too few cell contexts. An activation threshold of `6`
missed familiar transitions. A threshold of `3` predicted too many cells.

## Terms Used In This Guide

- **Correctly predicted activation:** A cell was predictive on the previous
  timestep, had an active distal segment, and is active now.
- **Burst:** A column wins, but none of its cells had a valid prediction from the
  previous timestep. The column activates all its cells because it does not know
  which temporal context applies.
- **Proximal permanence:** The strength of a connection from an input bit to a
  column. Raising it makes that input bit more likely to help the column win in
  the future.

## What Each Knob Does

- `enabled`: turns temporal pooling on for the layer. In delayed configs, keep
  this `false` in the base config and enable it with a runtime override after
  sequence memory has had time to form.

- `enable_persistence`: can keep a cell predictive after its normal prediction
  ends. Keep this `false` while tuning. An old prediction can interfere with a
  later sequence.

- `delay_length`: controls how quickly the learned persistence duration changes.
  It has no effect on predictions when `enable_persistence` is `false`.

- `spatial_permanence_inc`: is the base increase for proximal permanence. Both
  scale settings below multiply this value.

### `active_predict_proximal_scale`

This setting strengthens the input that a correctly predicted column just won
for.

```text
t-1: a cell in column 7 is predicted by an active distal segment
t:   column 7 wins and that cell becomes active
     -> strengthen column 7's proximal connections to the input at t
```

A burst does not qualify. The code requires the same cell to have all three
conditions:

1. Predictive at `t-1`.
2. Backed by an active distal segment at `t-1`.
3. Active at `t`.

The increase for each currently active proximal input is:

```text
spatial_permanence_inc
  * active_predict_proximal_scale
  * number of correctly predicted active cells in the column
```

The last multiplier is a count, not a yes/no check. For example, two correctly
predicted active cells cause twice the increase of one cell. The intent is to
treat more matching cells as stronger evidence.

This count has a cost. Proximal synapses belong to the column, not to individual
cells. The count therefore makes learning depend on how many cells the column
contains and how broad its predictions are. A future implementation could use a
yes/no check instead. That would make this setting easier to tune across layer
sizes. The current implementation does not do that.

Values below `1.0` reduce the base increase. Values above `1.0` increase it.

Raising this setting does not directly make a column pool across different
inputs in a sequence. It strengthens the input that the column already won for.
It can make that mapping more reliable, but a large value can also make a few
columns dominate later inputs.

### `post_active_proximal_scale`

This setting teaches a recently correct column about the next input in the
sequence.

```text
t-1: column 7 was correctly predicted and won for input B
t:   column 7 is still predicted by an active distal segment
     column 7 does not win for input C
     -> strengthen column 7's proximal connections to input C
```

All four conditions are required:

1. The column had a correctly predicted active cell at `t-1`.
2. The column is still predictive at `t`.
3. An active distal segment backs that prediction at `t`.
4. The column does not win at `t`.

The increase for each currently active proximal input is:

```text
spatial_permanence_inc * post_active_proximal_scale
```

This update does not use the number of predictive cells. Values below `1.0`
reduce the base increase. Values above `1.0` increase it.

This is the setting that can make one column win for several inputs in a learned
sequence. It can also cause bursting if it is too high. The proximal changes
remain after the prediction that allowed the update. On a later input, the
column can win from those proximal connections even when no cell in the column
was predicted on the previous timestep. The column then bursts.

- `sequence_permanence_inc`: distal TP learning rate. This controls how quickly
  new distal connections become strong enough to predict a cell. This should
  usually be higher than `spatial_permanence_inc`.

- `sequence_permanence_dec`: lowers distal permanence for connections that do
  not match the learned context. Raise it to forget old contexts faster. Lower
  it if useful distal connections disappear too soon.

## Tuning Order

1. Keep `enable_persistence: false`.
2. Tune `sequence_permanence_inc` until more familiar inputs have valid
   predictions.
3. Increase `spatial_permanence_inc` slowly with a low
   `active_predict_proximal_scale`.
4. Increase `post_active_proximal_scale` if columns do not yet win for more than
   one position in a learned sequence.
5. If bursting increases, reduce `post_active_proximal_scale` first.
6. Test `enable_persistence: true` only after temporal pooling works without it.

## What To Watch

The command-line diagnostics use these names:

- `true_burst_fraction`: the average fraction of winning columns that burst.
- `new_true_burst_fraction`: the average fraction that burst now but did not
  burst on the previous timestep.
- `new_true_burst_causes ... no_prev_prediction`: counts new bursts where the
  winning column had no predictive cell on the previous timestep.

Good signs:

- More winning columns have valid predictions from the previous timestep.
- `reinforced_inputs` is non-zero after TP turns on.
- Some columns win for more than one position in the same learned sequence.
- Unrelated sequences do not all use the same columns.
- The fraction of winning columns that burst stays low.

Bad signs:

- Proximal inputs are reinforced but no column wins for multiple sequence
  positions: `spatial_permanence_inc` or `post_active_proximal_scale` may be too
  low.
- `true_burst_fraction` rises while `new_true_burst_causes` reports
  `no_prev_prediction`: proximal learning is making columns win before sequence
  memory predicts them. Reduce `post_active_proximal_scale`, then
  `spatial_permanence_inc`.
- A small set of columns wins for unrelated inputs:
  `spatial_permanence_inc` or `post_active_proximal_scale` may be too high, or
  spatial learning had already collapsed before temporal pooling started.
- Bursting rises after persistence is enabled: turn persistence off and tune the
  other settings first.

## Current Caveat

Both proximal updates require a prediction backed by an active distal segment
when they are applied. This reduces unsafe learning, but it cannot guarantee
that the column will have a valid prediction every time those strengthened
proximal connections later make it win.

Do not tune for the largest learning rates. Tune for all three results:

1. Some columns win across several positions in one learned sequence.
2. Unrelated sequences use different columns.
3. Familiar inputs cause few bursts.
