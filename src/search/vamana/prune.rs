/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Vamana RobustPrune, adapted from Microsoft DiskANN
//! (<https://github.com/microsoft/DiskANN>, commit `8a9ff2d1fc68441297a5761c5647d1d4c16fe39e`):
//!
//! * [`robust_prune`] and [`State`] come from `diskann/src/graph/internal/prune.rs`.
//! * [`update_occlude_factor`] is the L2 rule (`PruneKind::TriangleInequality`) of
//!   `PruneKind::update_occlude_factor` in `diskann/src/graph/config/mod.rs`.
//!
//! The license and notice are in `LICENSE-DISKANN.txt` at the repository root.
//!
//! Changes from the original:
//! * Candidates are a plain slice of [`Candidate`] (already sorted and truncated by the
//!   caller) instead of DiskANN's `SortedNeighbors<Option<V>>`.
//! * Only the L2 pruning rule is kept, since Catapult only searches with squared L2.
//! * Unchecked indexing is replaced by checked indexing.

/// A prune candidate: its distance to the node being pruned, and its vector, or `None`
/// to exclude it (e.g. the node itself) without changing positional alignment.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Candidate<V> {
    pub(crate) distance: f32,
    pub(crate) vector: Option<V>,
}

/// Position-wise state tracking.
///
/// Refer to the inline documentation in [`robust_prune`] for documentation on the use
/// of these fields.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct State {
    /// The occlude factor for the pool item at the corresponding index.
    pub(crate) occlude_factor: f32,
    /// The index of the last checked neighbor.
    pub(crate) last_checked: u16,
    /// The candidate index of this neighbor.
    pub(crate) neighbor: u16,
}

/// The L2 occlusion rule: candidate `k` is occluded by an already chosen neighbor `j`
/// by the factor `d(i, k) / d(j, k)`, where `i` is the node being pruned.
///
/// With squared L2 distances, as here and in DiskANN, `alpha` therefore bounds a ratio
/// of squared distances.
pub(crate) fn update_occlude_factor(
    distance_ik: f32,
    distance_jk: f32,
    current_factor: f32,
) -> f32 {
    if distance_jk == 0.0 {
        f32::MAX
    } else {
        current_factor.max(distance_ik / distance_jk)
    }
}

/// Select a degree-bounded neighbor set with Vamana RobustPrune.
///
/// `candidates` stores each source distance and candidate vector in ascending
/// source-distance order. `None` excludes that candidate without changing
/// positional alignment. `states` has one entry for each candidate position.
///
/// The function writes selected candidate indexes to `states[..result]` and
/// returns `result`. The caller converts those indexes to graph IDs.
pub(crate) fn robust_prune<V, D>(
    candidates: &[Candidate<V>],
    states: &mut [State],
    degree: usize,
    alpha: f32,
    mut compute_distance: D,
) -> usize
where
    D: FnMut(&V, &V) -> f32,
{
    let mut current_alpha = 1.0f32;
    let increment_factor = alpha.min(1.2);

    // For an alpha value `A`, a candidate `i` is promoted to a neighbor if for all
    // ```
    // max{j < i | j is a neighbor}(occlude_factor(i, j))
    // ```
    // This process happens with multiple values of `A`.
    //
    // We can compute this efficiently using the following rules:
    //
    // 1. For a candidate `i`, start scanning `j < i`, computing occlude factors.
    // 2. If we find an occlude factor greater than `A`, record that `i` has visited
    //    `j`, stop computing occlude factors, and move on to `i + 1`.
    // 3. If we reach `j == i - 1` with the maximum occlude factor less than `A`, then
    //    `i` gets promoted to a neighbor.
    //
    // On the implementation side, we use `states` in the following way:
    //
    // * `states[n].neighbor` is the **index** in `candidates` of the `n`th **neighbor**.
    //   Note that a "neighbor" is a candidate that passes pruning.
    //
    //   Very important: to get the index `j` in the above description, we need to
    //   check `candidates[states[n].neighbor]`.
    //
    //   This indexing naturally skips candidates `j` that have not been promoted to
    //   neighbors.
    //
    // * `states[i].occlude_factor` is the maximum occlude factor found for a candidate
    //   `i`. This gets set to `f32::MAX` when `i` is promoted to a neighbor which
    //   excludes it from future consideration.
    //
    // * `states[i].last_checked` is the highest value of `n` against which the
    //   occlude factor for `j = candidates[states[n].neighbor]` has been checked.
    //
    //   The maximum value this should reach is `i`.
    //
    // Note that we use `states` for both "candidate" and "neighbor" tracking.
    let mut found = 0;
    while found < degree {
        for (i, candidate) in candidates.iter().enumerate() {
            if found >= degree {
                break;
            }

            // The tracking states for candidate `i`.
            let State {
                mut occlude_factor,
                mut last_checked,
                ..
            } = states[i];

            // If the occlusion factor for this neighbor is too high, skip it.
            if occlude_factor > current_alpha {
                continue;
            }

            // Excluded candidates are skipped for good.
            let neighbor_distance = candidate.distance;
            let neighbor = match candidate.vector.as_ref() {
                Some(n) => n,
                None => {
                    states[i].occlude_factor = f32::MAX;
                    continue;
                }
            };

            // Increment `position` until we've compared with all current entries in
            // `result`.
            //
            // When the list is empty, the loop is skipped allowing the first undeleted
            // element to be added.
            while last_checked as usize != found {
                let result_position = states[last_checked as usize].neighbor as usize;
                last_checked += 1;

                // If the position of this result in `candidates` is greater than or equal
                // to the current working position, then skip this candidate.
                if result_position >= i {
                    states[i].last_checked = last_checked;
                    continue;
                }

                // Otherwise, compute the distance between the result and this neighbor
                // and update the occlude factor.
                let distance = match candidates[result_position].vector.as_ref() {
                    Some(v) => compute_distance(neighbor, v),
                    None => f32::MAX,
                };

                // Update occlude factor
                occlude_factor = update_occlude_factor(neighbor_distance, distance, occlude_factor);

                // Check if the most recent update to the occlusion factor removes this
                // neighbor from consideration.
                if occlude_factor > current_alpha {
                    break;
                }
            }

            let state = &mut states[i];

            state.last_checked = last_checked;
            if occlude_factor > current_alpha {
                state.occlude_factor = occlude_factor;
                continue;
            }

            // This neighbor has passed all the requirements of being a candidate.
            state.occlude_factor = f32::MAX;

            // This conversion should always succeed.
            states[found].neighbor = i as u16;
            found += 1;
        }

        // Exit if we completed the final iteration.
        if current_alpha == alpha {
            break;
        }
        // Update current alpha for the next iteration.
        current_alpha = (current_alpha * increment_factor).min(alpha);
    }

    found
}
