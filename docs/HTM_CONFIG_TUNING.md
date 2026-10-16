# HTM Configuration Tuning

## Purpose

This guide gives practical starting values for an HTM layer.

Use it when:

- A new network does not learn.
- Familiar inputs still burst.
- The same columns win for unrelated inputs.
- Temporal pooling does not make representations more stable.

The values are starting points. Layer size, input sparsity, receptive-field
size, and sequence complexity all affect the useful range.

## Start With One Stage At A Time

Tune the layer in this order:

1. Spatial pooling.
2. Sequence memory.
3. Temporal pooling.

Do not tune all three at once. Temporal pooling changes proximal and distal
synapses. It can hide a spatial-pooling or sequence-memory problem without
fixing it.

Keep temporal pooling disabled until:

- Similar inputs produce repeatable winning columns.
- Unrelated inputs do not use the same winning columns.
- Familiar transitions produce predictive cells.
- Familiar columns usually stop bursting.

## Important Comparison Rules

Several thresholds use a strict `>` comparison.

```text
Configured threshold: 3
Required count:        4
```

This applies to:

- `min_num_syn_threshold`
- `activation_threshold`

The two thresholds count different synapses:

- `min_num_syn_threshold` counts active synapses with permanence above `0`.
  It decides whether a segment is a partial match and should be reused.
- `activation_threshold` counts active connected synapses. It decides whether
  the segment predicts its cell.

A segment can therefore be strong enough to select a learning cell but too weak
to predict that cell.

## Safe Starting Configuration

This profile is suitable for a small experiment with ten synapses per distal
segment. It is not a universal optimum.

```yaml
overlap:
  connected_perm: 0.3
  min_overlap: 2
  min_potential_overlap: 1

inhibition:
  desired_local_activity: 2
  strict_local_activity: true

spatial_learning:
  permanence_inc: 0.05
  permanence_dec: 0.02
  active_col_permanence_dec: 0.01

sequence_memory:
  cells_per_column: 4
  max_segments_per_cell: 4
  max_synapses_per_segment: 10
  min_num_syn_threshold: 3
  new_syn_permanence: 0.3
  connect_permanence: 0.2
  activation_threshold: 3
  permanence_inc: 0.05
  permanence_dec: 0.02

temporal_pooling:
  enabled: false
  spatial_permanence_inc: 0.05
  active_predict_proximal_scale: 0.25
  prev_predict_proximal_scale: 0.0
  sequence_permanence_inc: 0.05
  sequence_permanence_dec: 0.0
```

With both distal thresholds set to `3`, a partial match and a prediction each
require at least four active synapses. Prediction still has the stricter
requirement because those synapses must also be connected.

Start with four segments per cell when the same column can occur in several
temporal contexts. Reduce this only after measuring segment use.

## Spatial Pooling

### Receptive field

`pot_width` and `pot_height` define the input area visible to one column.

Start with a field large enough to contain the smallest useful input feature.
Do not make every column see the entire input unless global competition is
intentional.

### `connected_perm`

Start near `0.3` for proximal synapses.

- Raise it if too many weak synapses contribute to overlap.
- Lower it if almost no columns reach `min_overlap`.

Do not change this first. Check receptive fields and initial permanence values
before changing the connection threshold.

### `min_overlap`

Start with `2` or `3`.

- Raise it if noise activates columns.
- Lower it if valid sparse inputs produce no winners.

Keep `min_potential_overlap` at `1` while bootstrapping. A value of `0` can let
columns with no input support compete through tie-breaking alone.

### `desired_local_activity`

This is the target number of winners in each inhibition neighborhood.

- Increase it if the output is too sparse to represent the input.
- Decrease it if many columns learn the same pattern.

Judge sparsity relative to the inhibition-neighborhood size. The same value
behaves differently in a `5x5` and a `20x20` neighborhood.

### Spatial learning rates

Start with:

```yaml
permanence_inc: 0.05
permanence_dec: 0.02
active_col_permanence_dec: 0.01
```

- Increase `permanence_inc` if useful winners do not become repeatable.
- Increase `permanence_dec` if columns retain many unrelated inputs.
- Reduce both if winners oscillate rapidly or learning collapses onto a small
  set of columns.

## Sequence Memory

### `cells_per_column`

Cells represent different temporal contexts for the same spatial pattern.

Start with `4`.

- Increase it when the same input occurs after many different predecessors.
- Increasing it does not help when one cell keeps winning weak partial-match
  selection. Fix the matching thresholds first.

### `max_segments_per_cell`

Segments represent contexts that can predict one cell.

Start with `4` when:

- Inputs repeat within a sequence.
- Sequences branch.
- Temporal pooling is enabled.

Two segments can work for simple sequences. They leave little room when
sequence learning and temporal pooling both update the same segment array.

### `max_synapses_per_segment`

Start with `10` for small sparse layers and `20` for larger contexts.

The value must be greater than `activation_threshold`. A segment cannot predict
when it cannot contain enough connected active synapses to cross that threshold.

### `min_num_syn_threshold`

This controls reuse of a partially matching segment.

For initial tuning, set it equal to `activation_threshold`.

If it is much lower:

1. A weak segment repeatedly wins learning-cell selection.
2. The same cell is selected after each burst.
3. Other cells with unused segments are not selected.
4. The weak segment can remain below the prediction threshold.

Lower it only when the model fails to reuse genuinely useful partial segments.

### `activation_threshold`

Start between 30% and 50% of `max_synapses_per_segment`.

For ten synapses:

```yaml
activation_threshold: 3
```

This requires four connected active synapses.

- Raise it if too many cells predict.
- Lower it if learned transitions remain unpredicted.

Do not lower it only to hide bursting. First check whether the correct segment
contains connected synapses to the cells active on the previous timestep.

### Distal permanence

A useful starting relationship is:

```text
new_syn_permanence >= connect_permanence
```

For example:

```yaml
new_syn_permanence: 0.3
connect_permanence: 0.2
permanence_inc: 0.05
permanence_dec: 0.02
```

This lets new synapses contribute immediately. Set
`new_syn_permanence < connect_permanence` only when new synapses must earn
connection through repeated evidence.

## Temporal Pooling

Enable temporal pooling only after sequence memory predicts familiar
transitions.

Start with:

```yaml
temporal_pooling:
  enabled: true
  spatial_permanence_inc: 0.05
  active_predict_proximal_scale: 0.25
  prev_predict_proximal_scale: 0.0
  sequence_permanence_inc: 0.05
  sequence_permanence_dec: 0.0
```

### `active_predict_proximal_scale`

This strengthens the current input after a column was correctly predicted and
won.

It makes an existing input-to-column mapping more reliable. It cannot teach
the column a new input, because it only fires for columns that already won.
Raising it does not increase pooling.

### `prev_predict_proximal_scale`

This teaches a column the input it was predicted for but lost. It fires when a
cell in the column was predictive at `t-1` with an active segment at `t-1`,
and the column did not win at `t`. The column learns the input at `t`.

These are the same conditions the active-cells stage checks to decide whether
a newly active column bursts, so when the column later wins for that input in
the same context it does not burst.

It pools in both directions. For a sequence A, B, C: a column that wins B
gets a TP segment onto the other cells for B, so next cycle it is predicted
for C, loses C, and learns C (forward). A column that wins C has a sequence
segment onto cells for B; once those cells are also active for A, it is
predicted for B, loses B, and learns B (backward). There is no separate
forward setting and none is needed.

Leave it at `0.0` until distal prediction is reliable. Then try:

```yaml
prev_predict_proximal_scale: 0.25
```

Increase it to `0.5` only if cross-input pooling remains weak.

The proximal increase is:

```text
spatial_permanence_inc * prev_predict_proximal_scale
```

Large values, or a large `spatial_permanence_inc`, can make a few columns win
every position of a sequence. That is stable but carries no information.

This setting needs candidates: columns predicted at `t-1` that lose at `t`.
Sequence memory alone produces almost none on a learned sequence. The
candidates come from temporal-pooling distal segments onto the `prev2`
context, so `temporal_pooling.sequence_permanence_inc` must be above `0.0`.

Sequence learning lowers permanence on a segment that predicted a cell which
did not activate. This setting learns from exactly those predictions. Keep
`temporal_pooling.sequence_permanence_inc` well above
`sequence_memory.permanence_dec` so the prediction survives until the column
wins.

### Temporal-pooling distal rates

Start with:

```yaml
sequence_permanence_inc: 0.05
sequence_permanence_dec: 0.0
```

Add a small decay only after confirming that stale temporal-pooling synapses
need removal. TP and sequence memory share distal segments. Excessive TP decay
can damage sequence predictions.

## Common Problems

### The same cell is selected after every burst

Likely cause:

- `min_num_syn_threshold` is too low relative to `activation_threshold`.

Why:

- A weak partial segment match takes priority over another cell with unused
  segments.

Change:

1. Set `min_num_syn_threshold` equal to `activation_threshold`.
2. Confirm that the segment now needs enough support to be close to predicting.
3. Increase `cells_per_column` only if several real contexts still compete.

### A familiar column still bursts

Likely causes:

- No cell was predictive on the previous timestep.
- A segment had too few connected active synapses.
- Useful segments were reused or weakened.
- The prediction that authorized a `prev_predict` proximal update was punished
  by sequence learning and disappeared before the column won.
- The column was bursting on the previous timestep and kept winning. A
  continuing column keeps its bursting state.

Change:

1. Check prediction and active-segment state at `t-1`.
2. Check connected support against `activation_threshold`.
3. Check `temporal_pooling.sequence_permanence_inc` against
   `sequence_memory.permanence_dec`.
4. Reduce temporal-pooling `spatial_permanence_inc`.
5. Reduce `prev_predict_proximal_scale`.
6. Increase `max_segments_per_cell` only when the existing segments hold
   different useful contexts.

### A segment is updated but does not predict

An update timestamp does not prove that a segment was active.

Check:

- How many synapse targets were active.
- How many of those synapses reached `connect_permanence`.
- Whether the count was greater than `activation_threshold`.

### Two columns alternate for related inputs

Likely causes:

- Each column is spatially specialized for one input.
- `prev_predict_proximal_scale` is `0.0`.
- The predicted loser is not receiving cross-input proximal learning.

Change:

1. Confirm that sequence prediction is reliable.
2. Set `prev_predict_proximal_scale` to `0.25`.
3. Check whether one column begins winning both positions.
4. Stop increasing it if bursting or representation collapse rises.

### The same columns win for unrelated inputs

Likely causes:

- Proximal learning is too strong.
- Too many columns win inhibition.
- Receptive fields are too broad.
- `prev_predict` temporal pooling is smearing inputs together.

Change:

1. Reduce `prev_predict_proximal_scale`.
2. Reduce temporal-pooling `spatial_permanence_inc`.
3. Reduce `desired_local_activity`.
4. Check receptive-field size and input overlap.

### No columns win

Likely causes:

- `min_overlap` is too high.
- Too few proximal synapses are connected.
- Receptive fields miss the active input.

Change:

1. Inspect potential overlap.
2. Keep `min_potential_overlap` at `1`.
3. Lower `min_overlap`.
4. Check `connected_perm` and receptive-field placement.

### Too many cells predict

Likely causes:

- `activation_threshold` is too low.
- Distal segments contain mixed contexts.
- Too many synapses remain connected.

Change:

1. Raise `activation_threshold` by one.
2. Raise `min_num_syn_threshold` with it.
3. Increase distal decay only if stale synapses are confirmed.

### Familiar transitions never become predictive

Likely causes:

- `activation_threshold` is too high.
- `new_syn_permanence` is below `connect_permanence`.
- Distal permanence increase is too small.
- Too few synapses represent the active context.

Change one setting at a time in this order:

1. Confirm the expected synapse targets.
2. Lower `activation_threshold` by one.
3. Increase sequence `permanence_inc`.
4. Increase `max_synapses_per_segment`.

## Test Each Change

Use a fixed random seed and record:

- Winning columns for each input position.
- Burst fraction, split into new winners and continuing winners.
- New bursts with no previous prediction.
- Number of predictive cells.
- Connected active synapses per relevant segment.
- Pooled-column overlap within one sequence.
- Pooled-column overlap between unrelated sequences.

Exclude bursting columns from any active-cell pooling metric. A column that
bursts and keeps winning stays bursting, and counts `cells_per_column` pooled
cells every step without a single correct prediction.

### A control layer with temporal pooling off still shows some pooling

This is expected. Two spatial-pooler facts produce it:

- A receptive field wider than one sequence step sees several consecutive
  inputs with similar overlap. The same column can win two steps in a row.
- Spatial learning raises permanence to every input a column wins for. The
  decrement for the other input happens once per cycle and is smaller than
  the increment, so a column that wins two positions keeps both.

Nothing in the spatial pooler forces one input per column. There is no
boosting. Some columns win two or more positions and some never win. This
produces a small amount of cell persistence with no bursting. Measure it with
temporal pooling off and treat it as the baseline a temporal-pooling layer is
compared against.

Columns in such a layer also often have more than one predictive cell for the
same transition. Sequence memory activates every correctly predicting cell and
reinforces each one. Nothing removes a redundant one once it exists.

Change one parameter at a time. Run enough complete sequence cycles for the
change to affect permanence.

A useful configuration must satisfy all three conditions:

1. Familiar transitions are predicted and rarely burst.
2. Some columns remain useful across related sequence positions.
3. Unrelated patterns remain distinct.

## C++ Test Harness Warning

Some tests configure Layer 0 and then copy it into Layer 1:

```cpp
cfg.l1 = cfg.l0;
```

A Layer 0 change made before this line affects both layers. A Layer 1-only
change must be made after the copy:

```cpp
cfg.l1 = cfg.l0;
cfg.l1.min_num_syn_threshold = 3;
```

Without this distinction, a test can appear to prove a Layer 1 tuning change
when both layers actually changed.

## Related Documents

- `docs/TEMPORAL_POOLER.md`
- `docs/TEMPORAL_POOLING_CONFIG.md`
- `docs/TEMPORAL_POOLING_BURSTING.md`
