# Temporal Pooling And Bursting

This note explains the Layer 1 bursting issue seen while investigating
temporal pooling in `chat_htm`, what appears to cause it, where to look in the
code, and how to test it.

## Short Summary

The symptom is: enabling temporal pooling can increase column bursting in Layer
1 instead of smoothly improving temporal stability.

Current status: this is improved but not fully fixed. TP proximal learning is
now constrained by local distal evidence, and TP can strengthen real proximal
permanences for predicted columns. We still see some Layer 1 bursting with
temporal pooling enabled, so the remaining work is tuning and diagnosis rather
than declaring the issue solved.

Headless true-burst diagnostics show that the current repeated-sequence failure
mode is mostly **no previous prediction** for newly bursting Layer 1 columns.
The column had no cell that was predictive at `t-1`, so it burst. Proximal TP
can make a column win before distal sequence memory predicts it.

The main cause was in the proximal update itself. It learned an input using a
prediction made for a different timestep. It now uses only the prediction made
at `t-1` for `t`, which is what the active-cells stage checks before deciding
whether a column bursts. See "Implemented Fixes" item 5.

The reference delayed config uses a single runtime override with
active-predict reinforcement plus a moderate `prev_predict` update.
Layer 1 also uses `sequence_memory.cells_per_column: 6` and
`sequence_memory.activation_threshold: 4`. Four cells per column left familiar
branching transitions under-represented; threshold `6` left familiar transitions
under-predicted, while threshold `3` produced too much multi-cell predictive
activity.

The main reason is that a column only avoids bursting when a cell was both:

- predictive at `t-1`
- backed by an active sequence segment at `t-1`

Temporal pooling interferes with that because it updates the same proximal
permanence tensor already used by spatial learning, so TP can make columns win
inhibition before distal sequence structure is strong enough to support
non-bursting predictions.

The current calculator applies local safeguards:

- TP proximal reinforcement only applies to columns that had active-predict
  cells in the current temporal-pooler distal update, or columns that had a
  predictive cell at `t-1` and did not win inhibition at `t`.
- Cells made active because their column is currently bursting cannot count as
  active-predict support for TP distal or proximal learning.
- `temporal_pooling.spatial_permanence_inc` now strengthens real proximal
  permanences for those supported columns instead of adding a temporary overlap
  bonus.
- TP proximal reinforcement is split into two scale knobs:
  `active_predict_proximal_scale` for columns that were predicted and won, and
  `prev_predict_proximal_scale` for columns that were predicted and lost.
- TP distal reinforcement now rewards synapses targeting the same `prev2`
  learning-cell context used to select a temporal-pooling segment.

## Implemented Fixes

Several local fixes are now in place.

1. **Persistence was removed.**

   Temporal pooling used to have an optional persistence mode. It kept a cell
   predictive for a few timesteps after its active-predict streak ended, and
   copied the cell's active segment timestamp forward so the active-cells
   stage treated the prediction as valid and did not burst the column. Those
   predictions had no support from current distal input,
   and the GUI showed predictive cells with no segment over threshold.

   The temporal pooler now reads `predict_cells_time` and `active_segs_time`
   but never writes them. Only `PredictCellsCalculator` sets predictive state.

2. **TP proximal learning is trust-gated.**

   `TemporalPoolerCalculator::update_proximal()` is now the runtime proximal
   TP update. It consumes active-predict support recorded by the preceding
   `update_distal()` call, plus current predictive-cell and active-segment
   state, before changing proximal permanence.

   This keeps high `temporal_pooling.spatial_permanence_inc` values from being
   a generic spatial-learning shortcut. TP proximal learning only reinforces
   columns with local distal evidence: active-predict support, or a
   prediction at `t-1` for a column that lost inhibition at `t`.

   Active-predict support also requires that the column did not burst on the
   current timestep. A burst activates every cell, so treating that activity as
   a successful prediction would let a prediction created after one burst
   reinforce itself during a later burst.

Both fixes use local per-cell/per-column state already available in the layer.
No global burst-rate rule or layer-wide normalization was added.

3. **Active-predict support reinforces proximal permanence.**

   `TemporalPoolerCalculator` records which columns had active-predict cells
   during its distal update. After that same update, `HTMLayer::step_once()` asks
   the temporal pooler to run `update_proximal()` and reinforce active proximal
   inputs for those columns using `temporal_pooling.spatial_permanence_inc`.

   This means temporal pooling improves the real proximal support that future
   overlap calculations use. It is deliberately not a new synapse or long-lived
   trace; the bookkeeping is derived from the same active-predict condition that
   TP distal learning uses to reinforce or create segments.

   This path only fires for columns that already won. It cannot add a new
   input to a column, so on its own it produces no temporal pooling.

4. **TP distal matching and reinforcement use the same context.**

   `TemporalPoolerCalculator::update_distal()` selects a best-matching segment
   by counting synapses whose targets are in the `prev2` learning-cell context.
   It now reinforces synapses targeting that same `prev2` context.

   Previously, a segment could be selected because it matched `prev2`, but then
   reinforced according to cells active at the current timestep. That mismatch
   made it hard for TP-created distal synapses to become connected and later
   drive earlier/later predictive state.

5. **A column only learns a new input if it was predicted for that input.**

   `prev_predict_proximal_scale` controls the one proximal update that can
   add an input to a column. It fires when:

   - a cell in the column was predictive at `t-1`, and
   - the column did not win at `t`.

   The column's proximal synapses to the input at `t` are strengthened.

   When a column becomes active, `ActiveCellsCalculator` looks for the same
   `t-1` prediction. If a cell in the column was predictive at `t-1` (and so
   had an active segment at `t-1`), only that cell activates. Otherwise every
   cell activates: the column bursts. So when the proximal change later makes
   the column win for this input in the same context, the prediction that
   caused the change is also the prediction that stops the burst.

   A proximal update that uses any other evidence, such as prediction at `t`
   (which forecasts `t+1`), teaches the column an input it was never predicted
   for. When the column later wins that input, no cell was predictive at
   `t-1`, and the column bursts. That is the `no_prev_prediction` burst.

## Why Bursting Happens

For newly active columns, the burst/no-burst decision is made in:

- `htm_flow/src/sequence_pooler/active_cells/active_cells.cpp`

The key rule is:

- if no cell in the column was predictive at `t-1` and had an active segment at
  `t-1`, the column bursts

Prediction is calculated after active cells. A cell displayed as predictive at
time `t` is forecasting `t+1`; it did not predict or prevent a burst at `t`.
After a burst, all cells are active and can therefore make segments predictive.
The temporal pooler excludes that burst-created activity from active-predict
learning while the column is still bursting.

A column that was active at `t-1` and wins again at `t` is not checked for a
prediction. Its cells keep the state they had at `t-1`. If it was bursting at
`t-1`, it is bursting at `t`. A column that temporal pooling makes win every
position of a sequence therefore bursts once and then bursts for as long as it
keeps winning. Such a column looks perfectly pooled at the column level. So
measure pooling on winning columns and, separately, the fraction of winning
columns that are bursting, read from `ActiveCellsCalculator`'s
`burst_cols_time`. That record is stamped for new and continuing bursts. Do not
infer bursting from all cells in a column being active; several correctly
predicted cells give the same picture.

Important supporting code paths:

- `htm_flow/src/htm_layer.cpp`
  - step order is: overlap -> inhibition -> spatial learning -> active cells ->
    predict cells -> sequence learning -> temporal pooler
  - TP runs after normal sequence memory work and writes back into shared state
  - TP distal records active-predict support, then TP proximal reinforces active
    proximal inputs for those supported columns so the effect appears in later
    overlap calculations
- `htm_flow/src/temporal_pooler/temporal_pooler.cpp`
  - `update_proximal()` changes `col_syn_perm_`
  - TP proximal reinforcement requires active-predict support from the current
    TP distal update, or a prediction at `t-1` for a column that did not win
    inhibition at `t`
  - current bursting columns do not produce active-predict support
  - `active_predict_proximal_scale` and `prev_predict_proximal_scale` control
    how much each path contributes
  - `update_distal()` changes `distal_synapses_`
  - reused TP distal segments are matched and reinforced against `prev2`
    learning-cell context
- `htm_flow/src/sequence_pooler/predict_cells/predict_cells.cpp`
  - `predict_cells_time_` and `active_segs_time_` are produced only here,
    from connected distal support
  - connected active synapses are saved as packed bits for one timestep
- `htm_flow/src/sequence_pooler/sequence_learning/sequence_learning.cpp`
  - each segment that predicted an inactive cell loses permanence on the
    synapses that caused that prediction
  - another active segment can keep the cell predictive, but it cannot hide the
    failed segment
  - the current best segment is queued for positive learning on the next
    timestep

## Important Investigation Notes

- Temporal pooling is not a separate clean layer of logic. It shares learning
  tensors with the normal layer algorithm.
- The burst spike observed when turning TP on later was useful as a diagnosis,
  but the real goal is a static config that works from the start.
- At the moment, the safest configuration is to keep TP proximal learning weak
  and active-predict-gated, using `temporal_pooling.spatial_permanence_inc` as
  the main "more TP / less TP" proximal knob.
- The current proximal reinforcement model is intentionally local:
  - active-predict winners reinforce the input that made them win
  - columns predicted at `t-1` that lost inhibition at `t` learn the input at
    `t`
- Because TP proximal is now trust-gated, stronger TP settings can be explored
  with less risk of immediately reintroducing Layer 1 burst spikes.
- If `new_true_burst_causes` reports `no_prev_prediction`, a column won for an
  input it was not predicted for. With the `prev_predict` path this means the
  prediction that authorized the proximal change did not survive until the
  column won. Sequence learning punishes a segment that predicted a cell which
  did not activate, and the `prev_predict` path learns from exactly those
  predictions. Check that `temporal_pooling.sequence_permanence_inc` is well
  above `sequence_memory.permanence_dec` before lowering the proximal settings.
- The `prev_predict` path needs candidate columns: columns predicted at `t-1`
  that lose at `t`. In a well-learned sequence, sequence memory alone produces
  almost none, because its predictions are correct. The candidates come from
  TP distal segments, which predict a column for the position after the one it
  just won. If `temporal_pooling.sequence_permanence_inc` is `0.0`, no TP
  segments are built and the `prev_predict` path has nothing to act on.
- A layer with temporal pooling off can still show some column persistence.
  When a column's receptive field spans several sequence positions, spatial
  learning raises its permanence to every input it wins for, and it can win
  two adjacent positions. That is spatial-pooler behavior, not a
  temporal-pooling leak, and it does not cause bursting.

## Current Config Direction

The delayed-runtime reference config is:

- `configs/word_rows_2layer_delayed_temporal_pooling_text.yaml`
- `configs/overrides/word_rows_2layer_enable_temporal_pooling.yaml`

Layer 1 temporal pooling is enabled at timestep 1000 with:

- `spatial_permanence_inc: 0.12`
- `active_predict_proximal_scale: 0.25`
- `prev_predict_proximal_scale: 0.5`
- moderate TP distal learning

Layer 1 also uses six cells per column and sequence-memory
`activation_threshold: 4` so learned distal segments can predict familiar
word-row transitions without becoming overly broad.

Generic `HTMLayerConfig` defaults now also bias toward safer TP startup
(`temp_sequence_permanence_inc > temp_spatial_permanence_inc`).

This is meant to avoid temporary selection effects that make columns win without
improving their real proximal support. TP proximal learning now changes the same
permanences that future overlap calculations inspect, but only for columns that
had active-predict support or were predicted at `t-1`. Runtime logs
now report only the number of reinforced active proximal inputs. If that count
is non-zero but active columns do not broaden over time,
`temporal_pooling.spatial_permanence_inc` may be too small relative to
`connected_perm` and normal proximal decay. If a small set of columns dominates
every timestep, the value may be too large or the temporal support gates may be
too permissive.

## How To Test

The main regression coverage is in:

- `tests/integration/test_text_htm.cpp`

Relevant tests:

- `TextHTMIntegration.RuntimeEnableTemporalPoolingDoesNotSpikeLayer1Bursting`
  - keeps the original diagnostic scenario where TP is applied at runtime
- `TextHTMIntegration.AlwaysOnTemporalPoolingDoesNotRaiseLayer1BurstingVsNoTP`
  - compares always-on TP against a TP-off control
- `TextHTMIntegration.StrongTemporalPoolingDoesNotRaiseLayer1BurstingVsNoTP`
  - verifies that a much stronger Layer 1 TP setup does not automatically bring
    back the old burst failure mode

Calculator-level coverage lives in:

- `htm_flow/test/unit/test_temporal_pooler.cpp`
  - checks active-predict-gated TP proximal updates
  - checks that a column predicted at `t-1` with an active segment that lost
    inhibition at `t` learns the input at `t`
  - checks that a prediction made at `t` does not authorize learning the input
    at `t`
  - checks that a prediction at `t-1` still counts when its segment fired
    again at `t` and overwrote the `t-1` stamp

Two-layer column pooling with a separate bursting-fraction assertion:

- `htm_flow/test/integration/test_temporal_pooling_integration_suite5.cpp`

Run them with:

```bash
cmake --build build --target chat_htm_tests
./build/chat_htm_tests --gtest_filter=TextHTMIntegration.RuntimeEnableTemporalPoolingDoesNotSpikeLayer1Bursting:TextHTMIntegration.AlwaysOnTemporalPoolingDoesNotRaiseLayer1BurstingVsNoTP
```

For stronger TP tuning experiments, also run:

```bash
./build/chat_htm_tests --gtest_filter=TextHTMIntegration.StrongTemporalPoolingDoesNotRaiseLayer1BurstingVsNoTP
```

## What To Look At During Debugging

When investigating this issue again, check:

- whether Layer 1 active columns are increasing without a matching increase in
  predictive coverage
- whether bursting rises because TP proximal learning is bypassing the intended
  local temporal-support gate
- whether distal learning rates and thresholds make it too hard for TP-created
  synapses to become useful before new columns start winning
- whether TP distal segment creation/reuse is actually producing connected
  synapses that meet `activation_threshold` in `PredictCellsCalculator`
- whether TP proximal reinforcement is firing outside the old saturated winner
  set; add temporary instrumentation around `update_proximal()` if the aggregate
  `reinforced_inputs` log is not enough
- whether `temporal_pooling.spatial_permanence_inc` is too small to overcome
  normal proximal decay or so large that active-predict-supported columns become
  overly sticky
- the bursting fraction of winning columns alongside the pooling number; high
  pooling with high bursting is not evidence of temporal pooling

## Future Work

Future improvements should focus on making temporal learning stronger without
reintroducing burst spikes while keeping the state model easy to inspect.

Open items, in order:

1. A continuing column that is bursting stays bursting until it loses
   inhibition. Let it leave the bursting state when one of its cells was
   segment-backed predictive at `t-1`. This changes `ActiveCellsCalculator`
   and affects every test, so it needs its own measurement.
2. `temp_spatial_permanence_inc = 1.0` sets permanence to `1.0` in one update.
   That is assignment, not learning. Lower values spread pooling across more
   columns, each winning a run of several positions, instead of a few columns
   winning every position.
3. Tune `temporal_pooling.sequence_permanence_inc` against
   `sequence_memory.permanence_dec` together. The `prev_predict` path only has
   candidates while the TP distal prediction survives sequence-learning
   punishment.
4. Sequence memory never prunes a second correctly predicting cell in the same
   column. Both are activated and both are reinforced. This is why many
   columns show more than one predictive cell on a single repeating sequence.

Avoid adding another persistent synapse-like trace unless this simpler local
permanence update proves insufficient.

Do not treat the current implementation as a final bursting fix. It narrows when
TP proximal learning is allowed to change shared proximal permanence, but config
tuning still needs to balance stronger temporal pooling against remaining Layer
1 bursting.
