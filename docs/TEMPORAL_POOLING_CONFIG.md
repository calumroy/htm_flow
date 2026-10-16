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
  spatial_permanence_inc: 0.12
  active_predict_proximal_scale: 0.25
  prev_predict_proximal_scale: 0.5
  sequence_permanence_inc: 0.21
  sequence_permanence_dec: 0.002
```

This starts temporal pooling after sequence memory has had 1000 timesteps to
learn.

For the current word-row Layer 1 config, use
`sequence_memory.cells_per_column: 6` and
`sequence_memory.activation_threshold: 4`. Four cells per column left familiar
branching sequences with too few cell contexts. An activation threshold of `6`
missed familiar transitions. A threshold of `3` predicted too many cells.

## Terms Used In This Guide

- **Correctly predicted activation:** A cell was predictive on the previous
  timestep, had an active distal segment, is active now, and belongs to a
  column that did not burst now.
- **Burst:** A column wins, but none of its cells had a valid prediction from the
  previous timestep. The column activates all its cells because it does not know
  which temporal context applies. A column that keeps winning while bursting
  stays bursting; the active-cells stage carries the burst forward until the
  column loses inhibition for one timestep.
- **Valid prediction for timestep `t`:** A cell was predictive at `t-1` and one
  of its segments was active at `t-1`. This is what the active-cells stage
  checks at `t` to decide whether a newly active column bursts. A cell that is
  predictive at `t` is forecasting `t+1`; that
  prediction says nothing about whether the column should have won at `t`.
- **Proximal permanence:** The strength of a connection from an input bit to a
  column. Raising it makes that input bit more likely to help the column win in
  the future.

## What Each Knob Does

- `enabled`: turns temporal pooling on for the layer. In delayed configs, keep
  this `false` in the base config and enable it with a runtime override after
  sequence memory has had time to form.

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

A burst does not qualify. The code requires all of these conditions:

1. Predictive at `t-1`.
2. Backed by an active distal segment at `t-1`.
3. Active at `t`.
4. The column did not burst at `t`.

Predictive state displayed at timestep `t` is calculated after current cell
activity and forecasts `t+1`. A cell can therefore appear predictive on the
same timestep that its column bursts. That prediction did not prevent the
current burst and cannot authorize temporal-pooling active-predict learning.

The increase for each currently active proximal input is:

```text
spatial_permanence_inc
  * active_predict_proximal_scale
```

This is a yes/no check at the column level. One or more correctly predicted
active cells authorize one increase. Several matching cells do not multiply the
increase. Proximal synapses belong to the column, so this behavior does not
depend on how many cells in that column made the correct prediction.

Values below `1.0` reduce the base increase. Values above `1.0` increase it.

This setting cannot make a column pool across different inputs. It only fires
for a column that already won, so it can only strengthen inputs the column
already wins for. Raising it on its own buys nothing for pooling. A large value
makes pooled columns sticky and can make a few columns dominate later inputs.

### `prev_predict_proximal_scale`

This setting teaches a column the input it was predicted for but lost.

```text
t-1: a cell in column 7 is predicted by an active distal segment
t:   column 7 does not win for input C
     -> strengthen column 7's proximal connections to input C
```

All three conditions are required:

1. A cell in the column was predictive at `t-1`.
2. One of that cell's distal segments was active at `t-1`.
3. The column does not win at `t`.

The increase for each currently active proximal input is:

```text
spatial_permanence_inc * prev_predict_proximal_scale
```

This update does not use the number of predictive cells. Values below `1.0`
reduce the base increase. Values above `1.0` increase it.

This is the setting that makes one column win for several inputs in a learned
sequence. The conditions it checks are the same ones the active-cells stage
checks when a column becomes active: a cell predictive at `t-1` with an active
segment at `t-1` means that cell alone activates; otherwise every cell
activates and the column bursts. So when the proximal change later makes the
column win for input C in the same context, the prediction that caused the
change is also the one that stops the burst.

Using any other prediction, for example one made at `t` (which forecasts
`t+1`), teaches the column an input it was not predicted for. The column later
wins that input with no prediction at `t-1` and bursts.

#### Pooling direction

`prev` names the time of the evidence, not a direction. This one update
extends a column's input set both forwards and backwards. For a sequence
A, B, C:

- Forwards: a column wins B. The temporal-pooling distal update gives it a
  segment onto the other cells active for B (see `TEMPORAL_POOLER.md`, "Select
  the target cells for TP segments"). Next time B appears that segment fires
  and predicts the column for C. The column loses C and this update teaches it
  C. Each cycle adds one later position.
- Backwards: a column wins C. Its sequence segment connects to cells for B.
  Once other columns pool over A and B, some of those cells are also active
  for A. The segment fires at A and predicts the column for B. The column
  loses B and this update teaches it B.

Forward extension starts as soon as TP segments exist. Backward extension
starts once adjacent inputs share pooled cells, so it grows as pooling
spreads. No separate forward setting is needed.

Two limits apply:

- Sequence learning punishes a segment that predicted a cell which did not
  become active. The prediction this setting uses is exactly that case. The
  prediction must survive long enough for proximal permanence to cross
  `connected_perm`. Keep `temporal_pooling.sequence_permanence_inc` well above
  `sequence_memory.permanence_dec`.
- Most candidate columns come from temporal-pooling distal segments onto the
  `prev2` context, which predict a column for the step after the input it just
  won. With `temporal_pooling.sequence_permanence_inc` at `0.0` this setting
  has almost nothing to act on.

- `sequence_permanence_inc`: distal TP learning rate. This controls how quickly
  new distal connections become strong enough to predict a cell. This should
  usually be higher than `spatial_permanence_inc`.

- `sequence_permanence_dec`: lowers distal permanence for connections that do
  not match the learned context. Raise it to forget old contexts faster. Lower
  it if useful distal connections disappear too soon.

## Tuning Order

1. Tune `sequence_permanence_inc` until more familiar inputs have valid
   predictions.
2. Increase `spatial_permanence_inc` slowly with a low
   `active_predict_proximal_scale`.
3. Increase `prev_predict_proximal_scale` if columns do not yet win for more
   than one position in a learned sequence.
4. If a small set of columns wins every position, reduce
   `prev_predict_proximal_scale` or `spatial_permanence_inc`.
5. If bursting increases, check `temporal_pooling.sequence_permanence_inc`
   against `sequence_memory.permanence_dec` before lowering the proximal
   settings.

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
- Columns that keep winning are not bursting. Count pooled cells only in
  non-bursting columns. A column that bursts and keeps winning stays bursting
  and would otherwise count `cells_per_column` pooled cells every step.

Bad signs:

- Proximal inputs are reinforced but no column wins for multiple sequence
  positions: `spatial_permanence_inc` or `prev_predict_proximal_scale` may be
  too low, or `temporal_pooling.sequence_permanence_inc` is too low to create
  the early predictions this path needs.
- `true_burst_fraction` rises while `new_true_burst_causes` reports
  `no_prev_prediction`: a column is winning for an input it was not predicted
  for. With `prev_predict_proximal_scale` this means the prediction that
  authorized the proximal change was lost before the column won. Check that
  `temporal_pooling.sequence_permanence_inc` is well above
  `sequence_memory.permanence_dec`, then reduce `spatial_permanence_inc`.
- A small set of columns wins for unrelated inputs:
  `spatial_permanence_inc` or `prev_predict_proximal_scale` may be too high, or
  spatial learning had already collapsed before temporal pooling started.
- One column wins every position of a sequence. That is stable but carries no
  position information. Reduce `spatial_permanence_inc` so one update does not
  set permanence to `1.0`.

## Current Caveat

Both proximal updates require a prediction backed by an active distal segment.
The `prev_predict_proximal_scale` update uses the same evidence as the burst
gate, so a later win in the same context does not burst. It cannot guarantee
that the context is the same every time the column wins. If the same input
appears in another context, the column can win there without a prediction and
burst.

`active_segs_time` stores one timestamp per segment. If a segment is active at
both `t-1` and `t`, the `t-1` timestamp is overwritten before the temporal
pooler runs, and the `prev_predict` update does not fire for that cell on that
timestep.

Do not tune for the largest learning rates. Tune for all three results:

1. Some columns win across several positions in one learned sequence.
2. Unrelated sequences use different columns.
3. Familiar inputs cause few bursts.
