# Temporal Pooler

## Purpose

The temporal pooler helps an HTM layer form a stable representation of a
sequence.

Consider two input sequences:

```text
Sequence 1: A -> B -> C -> D
Sequence 2: E -> F -> G -> H
```

Without temporal pooling, each input can make a different set of columns win:

```text
Input:       A       B       C       D
Winners:   {1,4}   {2,7}   {3,8}   {5,9}
```

Temporal pooling changes synapse permanence so that some columns can win across
several positions in one learned sequence:

```text
Input:       A         B         C         D
Winners:   {1,4}     {1,7}     {1,8}     {1,9}
              \________ column 1 ________/
```

Column 1 is part of the pooled representation because it wins for several
sequence positions.

Different sequences should not all use the same pooled columns. Related
sequences can share some pooled columns when they share inputs.

Temporal pooling does not produce a separate output tensor. It changes existing
proximal and distal synapses. These changes affect which columns and cells
become active on later timesteps.

## HTM Terms

This section defines the terms used in this document.

### Sparse Distributed Representation

An input is a binary vector called a Sparse Distributed Representation, or SDR.
Most bits are zero. A small number of bits are one.

```text
Input SDR: 0 0 1 0 0 1 0 0 0 1
```

Different inputs use different active bits. Similar inputs can share some active
bits.

### Column

An HTM layer contains columns. Each column receives part of the input through
proximal synapses.

Columns compete during inhibition. The columns with the strongest input support
win and become active.

### Cell

Each column contains several cells.

The column represents an input pattern. Its cells represent that input in
different temporal contexts.

For example, the same column can use one cell when input C follows A,B and a
different cell when C follows X,Y.

### Proximal Synapse

A proximal synapse connects an input bit to a column.

Its permanence controls whether the connection contributes to the column's
input overlap. Higher permanence makes the input more likely to help that column
win inhibition.

### Distal Synapse

A distal synapse connects one cell to another cell.

Distal synapses learn temporal context. A cell becomes predictive when enough
connected distal synapses receive activity from the current context.

### Permanence

Permanence is a value from `0.0` to `1.0`.

A synapse is connected when its permanence reaches the configured connection
threshold. Learning raises or lowers permanence.

### Predictive Cell

A predictive cell expects to become active on a later timestep.

The prediction is valid for the burst gate only when an active distal segment
supports it.

### Active-Predict Cell

An active-predict cell:

1. Was predictive at timestep `t-1`.
2. Became active at timestep `t`.

This means the layer made a correct prediction.

### Bursting

A newly active column bursts when none of its cells had a segment-backed
prediction on the previous timestep.

During a burst, all cells in the column become active. Bursting is useful for
new input, but frequent bursting on a familiar sequence means prediction is not
working correctly.

## Position In The Layer

An HTM layer runs these stages for each input:

```text
Input SDR
    |
    v
+-----------+
| Overlap   |  Measure proximal input support for each column
+-----------+
    |
    v
+-----------+
| Inhibit   |  Select winning columns
+-----------+
    |
    v
+-----------+
| Spatial   |  Update normal proximal synapses
| learning  |
+-----------+
    |
    v
+-----------+
| Active    |  Select active and learning cells; burst if needed
| cells     |
+-----------+
    |
    v
+-----------+
| Predict   |  Activate distal segments and predictive cells
| cells     |
+-----------+
    |
    v
+-----------+
| Sequence  |  Update normal distal sequence synapses
| learning  |
+-----------+
    |
    v
+-------------------+
| Temporal pooling  |
| 1. Distal update  |
| 2. Proximal update|
+-------------------+
    |
    v
State used by later timesteps
```

The temporal pooler runs after the current winning columns and active cells have
already been selected. It cannot change the winners for the current timestep.
Its proximal changes affect overlap and inhibition on later timesteps.

The production call order is in `src/htm_layer.cpp`.

## Temporal Pooler Inputs And State

The temporal pooler uses state produced by earlier layer stages:

- Current learning cells.
- Current and previous active-cell timestamps.
- Current and previous predictive-cell timestamps.
- The last active timestep for each distal segment.
- Distal synapses between cells.
- Current potential proximal inputs for each column.
- Current proximal permanence values.
- Current winning columns.

It also stores:

- The length of each active-predict streak.
- The learned average streak length.
- A persistence countdown for each cell.
- Whether each column has active-predict support.
- Recent learning-cell entry events.

## Processing Steps

The temporal pooler has two main steps.

### Step 1: Update Distal State

`TemporalPoolerCalculator::update_distal()` updates temporal context between
cells.

#### 1. Record newly learning cells

The pooler records cells that entered the learning state on the current
timestep.

Cells that only remained in the learning state are not recorded as new entries.

#### 2. Select the earlier learning context

The pooler searches recent learning-cell entries and builds the `prev2` context.

The context excludes cells active at `t-1`. This selects cells from before the
immediately previous active set.

The pooler uses this context to learn longer temporal links.

```text
Earlier context       Previous input       Current input
     prev2                  t-1                  t
       |                     |                   |
       +-------- distal temporal link --------->+
```

#### 3. Find active-predict cells

For each cell, the pooler checks:

```text
predictive at t-1 AND active at t
```

Cells that satisfy both conditions are active-predict cells.

#### 4. Apply optional persistence

Persistence can keep a cell predictive for a short time after an active-predict
streak ends.

Persistence is only applied when the same cell had an active distal segment at
`t-1`. The pooler carries that segment timestamp forward with the prediction.

```text
t-1:
cell predictive + segment active
              |
              v
t:
prediction carried forward + same segment carried forward
```

The segment requirement prevents an unsupported predictive bit from bypassing
the burst gate.

Persistence is disabled when `enable_persistence` is `false`.

#### 5. Reinforce or create a distal segment

For each active-predict cell, the pooler finds the segment with the strongest
match to the `prev2` context.

If a matching segment exists:

- Synapses to `prev2` cells gain permanence.
- Other live synapses lose permanence.
- Synapses that reach zero permanence can be replaced with synapses to recent
  `prev2` cells.

If no matching segment exists:

- The least recently used segment is selected.
- Its synapses are replaced with connections to `prev2` cells.

#### 6. Count active-predict support by column

The pooler counts how many active-predict cells each column contains.

The proximal update uses this count.

### Step 2: Update Proximal Permanence

`TemporalPoolerCalculator::update_proximal()` changes how columns connect to
input bits.

Only proximal synapses whose input bit is active on the current timestep can
gain permanence.

There are two effective update paths.

#### Active-predict update

This path reinforces a column that was predicted and then won.

```text
t-1: column has a predictive cell
t:   column wins and that cell becomes active
     -> reinforce the column against input at t
```

The permanence increase is:

```text
spatial_permanence_inc
  * active_predict_proximal_scale
```

One or more active-predict cells authorize one column-level increase. Multiple
active-predict cells in the same column do not multiply the increase.

This update makes an already correct input-to-column mapping more reliable.

It does not directly teach the column to win for the next input.

#### Post-active update

This path teaches a recently correct column about the next input.

It applies when:

1. The column had active-predict support at `t-1`.
2. The column does not win at `t`.
3. The column is still predictive at `t`.
4. An active distal segment supports that prediction at `t`.

```text
t-1: predicted column wins for input B
                  |
                  v
t:   column stays segment-backed predictive for input C,
     but does not win
                  |
                  v
     reinforce that column's proximal synapses to input C
```

The permanence increase is:

```text
spatial_permanence_inc * post_active_proximal_scale
```

This is the main path that can make one column win across changing inputs in a
sequence.

All permanence values are capped at `1.0`.

## Configuration

Temporal-pooling settings appear under `temporal_pooling` in YAML.

### `enabled`

Type: Boolean.

Purpose: Enables both distal and proximal temporal-pooling updates.

When false, normal spatial learning and sequence memory still run.

It maps to `HTMLayerConfig::temp_enabled`.

### `enable_persistence`

Type: Boolean.

Purpose: Allows predictive state to continue for a learned number of timesteps
after an active-predict streak ends.

Persistence also carries forward the same cell's active segment timestamp.

Recommended current setting: `false`.

Persistence is more fragile than the distal and proximal learning paths.

### `delay_length`

Type: Positive integer.

Purpose: Controls how quickly the learned average persistence length changes.

It does not directly set a fixed number of persistence steps.

This setting has no effect on predictive state when `enable_persistence` is
false.

### `spatial_permanence_inc`

Type: Non-negative floating-point value.

Purpose: Sets the base permanence increase for temporal-pooling proximal
updates.

Both proximal scale settings multiply this value.

Increasing it makes temporal pooling change future column winners faster.
Excessive values can make a small set of columns dominate unrelated inputs.

### `active_predict_proximal_scale`

Type: Non-negative floating-point value.

Purpose: Scales proximal reinforcement for columns that were correctly
predicted and became active.

Example:

```text
spatial_permanence_inc = 0.12
active_predict_proximal_scale = 5.0

increase = 0.12 * 5.0 = 0.60
```

This setting strengthens correct winners. It does not directly make the same
column learn the next sequence input.

### `post_active_proximal_scale`

Type: Non-negative floating-point value.

Purpose: Scales the one-step later proximal update for a column that remains
segment-backed predictive after a correct activation.

Example:

```text
spatial_permanence_inc = 0.12
post_active_proximal_scale = 0.5

increase = 0.12 * 0.5 = 0.06
```

This setting is important for column-level pooling across changing inputs.

If it is too low, no columns may win across multiple sequence positions.

If it is too high, the same columns can learn many unrelated inputs and dominate
the layer.

### `sequence_permanence_inc`

Type: Non-negative floating-point value.

Purpose: Sets the permanence increase for temporal-pooling distal synapses that
match the selected `prev2` context.

Higher values make temporal context connections become usable faster.

### `sequence_permanence_dec`

Type: Non-negative floating-point value.

Purpose: Sets the permanence decrease for live distal synapses that do not match
the selected `prev2` context.

Higher values make segments forget old context faster.

Lower values retain context longer but can preserve stale connections.

## Related Sequence-Memory Settings

The temporal pooler depends on sequence memory. These settings are outside the
`temporal_pooling` YAML block but directly affect temporal-pooling behavior.

### `cells_per_column`

More cells give a column more capacity to represent the same input in different
temporal contexts.

Too few cells can merge unrelated contexts.

### `max_segments_per_cell`

More segments let one cell learn more temporal contexts.

### `max_synapses_per_segment`

This limits the number of context cells represented by one distal segment.

The temporal pooler also uses this value as the target size when selecting
recent `prev2` learning cells.

### `min_num_syn_threshold`

A temporal-pooling segment must have more than this number of positive-
permanence synapses matching `prev2` before it is reused.

### `new_syn_permanence`

This is the starting permanence for new temporal-pooling distal synapses.

### `connect_permanence`

A distal synapse must reach this permanence to contribute to prediction.

### `activation_threshold`

A distal segment needs more than this amount of connected active-synapse support
to make its cell predictive.

If the threshold is too high, familiar transitions remain under-predicted.

If it is too low, too many cells can become predictive.

## C++ Defaults

`HTMLayerConfig` currently uses:

```yaml
temporal_pooling:
  enabled: true
  enable_persistence: false
  delay_length: 4
  spatial_permanence_inc: 0.01
  active_predict_proximal_scale: 0.25
  post_active_proximal_scale: 0.0
  sequence_permanence_inc: 0.02
  sequence_permanence_dec: 0.01
```

These are generic defaults. They are not a tuned profile for every layer size or
input type.

## Current Word-Row Runtime Profile

`configs/word_rows_2layer_delayed_temporal_pooling_text.yaml` starts temporal
pooling disabled in Layer 1.

At timestep 1000 it applies:

```yaml
temporal_pooling:
  enabled: true
  enable_persistence: false
  delay_length: 16
  spatial_permanence_inc: 0.12
  active_predict_proximal_scale: 5.0
  post_active_proximal_scale: 5.0
  sequence_permanence_inc: 0.21
  sequence_permanence_dec: 0.002
```

This profile is much stronger than the C++ defaults.

## How To Recognise Correct Temporal Pooling

A good result must satisfy all of these conditions:

1. Some columns win reliably for more than one position in a familiar sequence.
2. Two unrelated sequences use different pooled-column sets.
3. A sequence that shares inputs with two other sequences shares some pooled
   columns with each.
4. Replaying a learned sequence restores its pooled-column set.
5. New unrelated inputs usually recruit columns outside existing pooled sets.
6. Familiar inputs do not cause a large increase in bursting.
7. Predictive cells remain backed by active distal segments.

Stability alone is not enough. One constant winner set for every input is stable
but carries no useful information.

## Current Issues

### The documented reference values do not match the runtime override

`TEMPORAL_POOLING_BURSTING.md` describes this delayed profile:

```yaml
active_predict_proximal_scale: 0.25
post_active_proximal_scale: 0.5
```

The current runtime override contains:

```yaml
active_predict_proximal_scale: 5.0
post_active_proximal_scale: 5.0
```

The documentation or the runtime override is stale. They cannot both describe
the current reference profile.

### Strong post-active updates can collapse unrelated sequences

The winning-column integration test uses:

```text
ABCD: no input overlap with EFGH
CDEF: shares C,D with ABCD and E,F with EFGH
```

With `spatial_permanence_inc=0.12` and
`post_active_proximal_scale=5.0`, one post-active update adds `0.60`
permanence.

In the tested 96-column topology, this produced:

```text
ABCD pooled columns:       4
EFGH pooled columns:       4
ABCD-EFGH Jaccard:         1.0
EFGH outside ABCD:         0.0
ABCD mean burst fraction:  0.75
```

The same four columns represented both disjoint sequences. The result was
stable but not useful.

### The post-active update has a narrow useful range

The same test changed only `post_active_proximal_scale` from `5.0` to `0.5`.

It produced:

```text
ABCD pooled columns:       4
EFGH pooled columns:       1
CDEF pooled columns:       2
ABCD-EFGH Jaccard:         0.0
CDEF-ABCD Jaccard:         0.2
CDEF-EFGH Jaccard:         0.5
Mean burst fractions:      0.0, 0.0, 0.0
```

With the post-active scale set to `0.0`, the same topology produced no
multi-position pooled columns.

This shows the current trade-off:

```text
Too weak                   Useful range                 Too strong
   |                            |                           |
no multi-position       sequence-specific pooled      same columns win
pooled columns          columns, low bursting         unrelated inputs
```

The useful scale depends on `spatial_permanence_inc`, the layer topology, input
sparsity, inhibition, and previous permanence values.

### Scale values are not topology-independent

The same numeric scale can behave differently when:

- The number of columns changes.
- The number of winners changes.
- Input sparsity changes.
- Potential-pool size changes.

The active-predict update is applied once per supported column. The number of
active-predict cells in that column does not change the increase.

### Proximal pooling can outrun distal prediction

If proximal permanence changes too quickly, a column can start winning for a new
input before sequence memory predicts that column.

The column then bursts because it has no segment-backed prediction from the
previous timestep.

```text
TP proximal learning makes column win
                  |
                  v
No matching prediction at t-1
                  |
                  v
Column bursts
```

The first settings to reduce are:

1. `post_active_proximal_scale`
2. `spatial_permanence_inc`

Do not weaken the burst gate to hide this problem.

### Persistence remains fragile

Persistence is segment-backed, but it can still broaden predictive activity and
increase cross-sequence interference.

Keep `enable_persistence: false` until base temporal pooling passes stability,
separation, recall, and burst tests.

### Sequence boundaries are not explicit

The layer does not have a public sequence-reset operation.

If training immediately changes from ABCD to EFGH, the layer can learn the
transition D->E and treat both as one longer sequence.

Tests currently use four empty inputs between named sequences to clear recent
cell-state timestamps.

Applications need an explicit boundary input, a reset mechanism, or a data
stream where cross-boundary transitions are intentional.

### Existing tests use different definitions of pooling

Some older integration tests measure overlap between learning-cell SDRs.

The winning-column tests define a pooled column as one that wins reliably for at
least two positions in a sequence.

Both measurements are useful, but they test different behavior. Test names and
failure messages must state which representation they measure.
