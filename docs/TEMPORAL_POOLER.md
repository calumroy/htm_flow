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

A prediction only stops a column from bursting when one of the cell's distal
segments was active at the time the prediction was made.

### Active-Predict Cell

An active-predict cell:

1. Was predictive at timestep `t-1`.
2. Became active at timestep `t`.
3. Belongs to a column that did not burst at timestep `t`.

This means the layer made a correct prediction.

### Bursting

A newly active column bursts when none of its cells had a segment-backed
prediction on the previous timestep.

During a burst, all cells in the column become active. Bursting is useful for
new input, but frequent bursting on a familiar sequence means prediction is not
working correctly.

A column that was active at `t-1` and wins again at `t` is not checked for a
prediction. Its cells keep the state they had at `t-1`. If it was bursting, it
stays bursting until it loses inhibition for one timestep.

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
- Current and previous bursting-column timestamps.
- Distal synapses between cells.
- Current potential proximal inputs for each column.
- Current proximal permanence values.
- Current winning columns.

It also stores:

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

#### 2. Select the target cells for TP segments

New TP distal synapses connect to a list of cells the code calls `prev2`.

The pooler keeps a log of every time a cell entered the learning state: one
entry of `(timestep, column, cell)` per event, in time order. Step 1 appended
this timestep's entries to that log. To build `prev2` the pooler walks the log
from the newest entry backwards and appends a cell to `prev2` if:

- the cell was not active at `t-1`, and
- the cell is not already in `prev2`.

It stops after it has `max_synapses_per_segment` cells and has finished the
timestep it was on, so all cells that entered learning on that timestep are
included.

The newest log entries are the cells that entered learning at `t`. A cell that
entered learning at `t` was not active at `t-1`, so it is accepted. The result
is that `prev2` is the set of cells that entered learning on the current
timestep, i.e. the cells that represent the current input.

So a TP segment created at `t` connects an active-predict cell to the other
cells that are active for the same input at `t`. The next time that input
appears the segment fires and makes the cell predictive. That prediction is
for the following timestep. The prev-predict proximal update in Step 2 uses it
to teach the column the following input.

#### 3. Find active-predict cells

For each cell, the pooler checks:

```text
predictive at t-1
AND active segment at t-1
AND active at t
AND column did not burst at t
```

Cells that satisfy all conditions are active-predict cells.

A cell can become predictive after its column bursts because prediction is
calculated after current activity. Predictive state shown at timestep `t`
forecasts timestep `t+1`; it did not prevent a burst at `t`. Burst-created cell
activity cannot authorize temporal-pooling learning.

The temporal pooler reads predictive state but never sets it. Only
`PredictCellsCalculator` makes a cell predictive, and only when one of its
segments has more than `activation_threshold` connected synapses onto currently
active cells.

#### 4. Reinforce or create a distal segment

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

#### 5. Count active-predict support by column

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
t:   column wins without bursting and that cell becomes active
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

It cannot teach the column a new input. It only fires for columns that won.

#### Prev-predict update

This path teaches a column the input it was predicted for but lost.

It applies when:

1. A cell in the column was predictive at `t-1`.
2. One of that cell's distal segments was active at `t-1`.
3. The column does not win at `t`.

```text
t-1: a cell in the column is segment-backed predictive
                  |
                  v
t:   the column does not win for input C
                  |
                  v
     reinforce that column's proximal synapses to input C
```

The permanence increase is:

```text
spatial_permanence_inc * prev_predict_proximal_scale
```

This is the path that makes one column win across changing inputs in a
sequence.

Conditions 1 and 2 are the same conditions the active-cells stage checks at `t`
before it decides that a newly winning column does not burst. When this update
later makes the column win for input C in the same context, that check passes.

Prediction at `t` is not used. It forecasts `t+1` and says nothing about
whether the column should have won at `t`.

All permanence values are capped at `1.0`.

## Configuration

Temporal-pooling settings appear under `temporal_pooling` in YAML.

### `enabled`

Type: Boolean.

Purpose: Enables both distal and proximal temporal-pooling updates.

When false, normal spatial learning and sequence memory still run.

It maps to `HTMLayerConfig::temp_enabled`.

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

This setting strengthens correct winners. It only applies to columns that won,
so it cannot make a column learn a new input. On its own it produces no
temporal pooling. Pooling needs `prev_predict_proximal_scale`.

### `prev_predict_proximal_scale`

Type: Non-negative floating-point value.

Purpose: Scales the proximal update for a column that was segment-backed
predictive at `t-1` and did not win at `t`.

Example:

```text
spatial_permanence_inc = 0.12
prev_predict_proximal_scale = 0.5

increase = 0.12 * 0.5 = 0.06
```

This setting is required for column-level pooling across changing inputs.

If it is too low, no columns win across multiple sequence positions.

If it is too high, or `spatial_permanence_inc` is large, a few columns can win
every position of a sequence. That is stable but carries no position
information.

This setting replaces `post_active_proximal_scale`. The old setting used
prediction at `t`, which forecasts `t+1`, to learn the input at `t`. The column
later won for that input without a prediction at `t-1` and burst.

`prev` names the time of the evidence. The update pools in both directions.
For a sequence A, B, C:

- Forwards: a column wins B. Its TP segment connects to the other cells for B.
  Next time B appears the segment fires and predicts the column for C. The
  column loses C and learns C.
- Backwards: a column wins C. Its sequence segment connects to cells for B.
  Once other columns pool over A and B, some of those cells are also active
  for A. The segment then fires at A and predicts the column for B. The
  column loses B and learns B.

Both occur in practice. Forward firings are slightly more common. Backward
firings increase as pooling spreads.

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
  spatial_permanence_inc: 0.01
  active_predict_proximal_scale: 0.25
  prev_predict_proximal_scale: 0.0
  sequence_permanence_inc: 0.02
  sequence_permanence_dec: 0.01
```

These are generic defaults. They are not a tuned profile for every layer size or
input type. With `prev_predict_proximal_scale` at `0.0` no column-level pooling
across inputs happens.

## Current Word-Row Runtime Profile

`configs/word_rows_2layer_delayed_temporal_pooling_text.yaml` starts temporal
pooling disabled in Layer 1.

At timestep 1000 it applies
`configs/overrides/word_rows_2layer_enable_temporal_pooling.yaml`:

```yaml
temporal_pooling:
  enabled: true
  spatial_permanence_inc: 0.12
  active_predict_proximal_scale: 0.25
  prev_predict_proximal_scale: 0.5
  sequence_permanence_inc: 0.21
  sequence_permanence_dec: 0.002
```

This profile is stronger than the C++ defaults. The `prev_predict` value was
carried over from the old `post_active` setting and has not been re-tuned for
the word-row config.

## Suite 5 Line-Sequence Profile

`test/integration/test_temporal_pooling_integration_suite5.cpp` uses a
deliberately strong Layer 1 profile on a 16-step moving-line sequence:

```yaml
temporal_pooling:
  spatial_permanence_inc: 1.0
  active_predict_proximal_scale: 5.0
  prev_predict_proximal_scale: 5.0
  sequence_permanence_inc: 1.0
```

Measured with bursting columns excluded from the metric: Layer 1 keeps `0.80`
of its non-bursting active cells between positions, Layer 0 keeps `0.12`.
With the old `post_active` path at the same values the all-column number was
`0.69`, of which more than half came from permanently bursting columns.

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

Measure pooling on non-bursting columns only. A bursting column that keeps
winning stays bursting and counts every cell as active on every step. On the
suite 5 line sequence, before the metric excluded bursting columns, more than
half of the "pooled" cells came from two or three permanently bursting
columns.

## Current Issues

### Strong prev-predict updates can collapse unrelated sequences

The winning-column integration test uses:

```text
ABCD: no input overlap with EFGH
CDEF: shares C,D with ABCD and E,F with EFGH
```

The measurements below were taken with the old `post_active` path. The
`prev_predict` path adds permanence in the same amount for each firing, so the
same trade-off applies. The exact numbers have not been re-measured.

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

### The proximal pooling update has a narrow useful range

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

The `prev_predict` update only changes permanence when a prediction at `t-1`
already exists for the column. A later win in the same context therefore has
the prediction that stops a burst. Two cases can still produce an unpredicted
win:

1. The prediction did not survive. Sequence learning lowers permanence on a
   segment that predicted a cell which did not become active. The `prev_predict`
   update learns from exactly those predictions. If
   `temporal_pooling.sequence_permanence_inc` is small compared with
   `sequence_memory.permanence_dec`, the prediction can disappear before the
   column wins.
2. The same input appears in a different context. The column wins from its
   proximal synapses, but the context cells that drove the prediction are not
   active.

```text
TP proximal learning makes column win
                  |
                  v
No matching prediction at t-1
                  |
                  v
Column bursts
```

The first things to check are:

1. `temporal_pooling.sequence_permanence_inc` against
   `sequence_memory.permanence_dec`.
2. `spatial_permanence_inc`. A value that sets permanence to `1.0` in one
   update leaves no time for the distal prediction to stabilise.

Do not relax the prediction check in the active-cells stage to hide this
problem.

### Continuing bursting columns never recover

A column that wins at `t-1` and `t` keeps its previous cell state. If it was
bursting at `t-1`, it is bursting at `t`. A column that pooling makes win every
position bursts once and then bursts for as long as it keeps winning.

Any metric that counts active cells must exclude these columns. The suite 5
test does. The fix in the active-cells stage is listed under future work in
`TEMPORAL_POOLING_BURSTING.md`.

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
