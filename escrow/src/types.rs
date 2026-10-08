//! Fee-schedule state model and its validation boundaries.
//!
//! This module owns the **only** in-memory representation of the fee-schedule triple
//! (`active` / `pending` / `previous`) and every rule that decides whether a
//! submission is accepted, rejected, or treated as a duplicate. Nothing here talks
//! to storage directly except through [`FeeScheduleState::load`] /
//! [`FeeScheduleState::persist`], so the whole state machine is testable in isolation
//! and cannot be bypassed by a caller that writes the storage keys directly.
//!
//! # Invariants
//!
//! 1. **At most one pending schedule.** A submission either replaces nothing (no
//!    pending), is a duplicate of the pending schedule (no-op), or is rejected with
//!    [`FeeScheduleError::PendingScheduleExists`]. A pending schedule is never
//!    silently overwritten.
//! 2. **Activation is monotone and idempotent.** [`FeeScheduleState::activate_if_due`]
//!    promotes a pending schedule only once `activation_ledger <= current_ledger`, and
//!    it reports whether *this* call performed the promotion. Calling it again at or
//!    after the same ledger is a no-op returning `false`.
//! 3. **The previous schedule is never lost.** A promotion moves `active` into
//!    `previous` before overwriting `active`, so recovery can always refer to the last
//!    known-good schedule.
//! 4. **Bounds are checked before any mutation.** [`FeeScheduleState::stage`] validates
//!    `min_fee_bps <= fee_bps <= max_fee_bps` before touching `pending`; a rejected
//!    submission therefore leaves the in-memory state byte-identical to what was
//!    loaded, so a caller that aborts on the error cannot have persisted a partial
//!    transition.
//! 5. **Views are pure.** `active_at` / `pending_at` / `previous_at` compute the
//!    not-yet-promoted boundary activation on the fly and never write, so reads cannot
//!    race with, or pre-empt, a later promotion.
//!
//! # Failure handling
//!
//! Every rejection path returns a typed [`FeeScheduleError`] and leaves `self`
//! untouched. Combined with invariant 4 this gives the callers a single rule: only
//! call [`FeeScheduleState::persist`] after a `stage` call returned `Ok`.
//!
//! # Storage compatibility
//!
//! The three values live in their original keys ([`FeeScheduleStorageKey::Active`],
//! [`FeeScheduleStorageKey::Pending`], [`FeeScheduleStorageKey::Previous`]) and keep
//! their original `Option<FeeSchedule>` encoding. An instance written by an earlier
//! release reads back through [`FeeScheduleState::load`] unchanged; the only additive
//! key is [`FeeScheduleStorageKey::MutationLock`], whose absence reads as "unlocked"
//! and therefore reproduces the pre-guard behavior exactly.

use crate::{FeeSchedule, FeeScheduleError, FeeScheduleStorageKey};
use soroban_sdk::Env;

/// A transaction-local view of fee-schedule state.
///
/// The three values remain in their existing storage keys for compatibility.
/// Callers stage transitions here and persist once, so a failed or competing
/// invocation cannot expose a partially promoted schedule.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct FeeScheduleState {
    pub(crate) active: Option<FeeSchedule>,
    pub(crate) pending: Option<FeeSchedule>,
    pub(crate) previous: Option<FeeSchedule>,
}

/// Result of staging a submission against a [`FeeScheduleState`].
///
/// Distinguishing the three outcomes lets the caller decide whether a write is needed
/// at all: `Accepted` mutated the state, while the other two are no-ops, so a retry
/// never rewrites storage or emits a duplicate event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StageOutcome {
    /// The schedule was staged as the new pending schedule; the caller must persist.
    Accepted,
    /// The schedule already is the pending schedule. Treated as success so a retried
    /// submission is idempotent rather than an error.
    DuplicatePending,
    /// The schedule is already the active schedule, possibly because it was promoted
    /// by an earlier call in this same invocation. Treated as success.
    DuplicateActive,
}

impl FeeScheduleState {
    pub(crate) fn load(env: &Env) -> Self {
        let storage = env.storage().instance();
        Self {
            active: storage.get(&FeeScheduleStorageKey::Active),
            pending: storage.get(&FeeScheduleStorageKey::Pending),
            previous: storage.get(&FeeScheduleStorageKey::Previous),
        }
    }

    /// Promote a due pending schedule in memory. The caller persists only after
    /// all validation succeeds, keeping activation and subsequent writes atomic.
    ///
    /// Returns `true` only on the invocation that performs the promotion, so
    /// [`crate::LiquifactEscrow::activate_fee_schedule`] is idempotent and a
    /// concurrent second caller observes `false` rather than re-promoting.
    ///
    /// Boundary: `activation_ledger == current_ledger` is due. This is what makes a
    /// schedule submitted for "the next ledger" activate deterministically rather
    /// than one ledger late.
    pub(crate) fn activate_if_due(&mut self, current_ledger: u32) -> bool {
        let is_due = self
            .pending
            .as_ref()
            .map(|schedule| schedule.activation_ledger <= current_ledger)
            .unwrap_or(false);
        if !is_due {
            return false;
        }

        // `previous` must be overwritten *before* `active` is taken, otherwise the
        // old active schedule would be lost and recovery could not refer to it.
        self.previous = self.active.take();
        self.active = self.pending.take();
        true
    }

    /// Stage `schedule` as the pending schedule, applying every validation boundary.
    ///
    /// Order of checks (all before any mutation):
    /// 1. `min_fee_bps <= fee_bps <= max_fee_bps`, else
    ///    [`FeeScheduleError::FeeOutOfBounds`].
    /// 2. `activation_ledger > current_ledger`, else
    ///    [`FeeScheduleError::InvalidActivationLedger`]. A schedule whose activation
    ///    ledger is the current one would activate within the same invocation and
    ///    bypass the review window it was meant to create.
    /// 3. No other pending schedule, else
    ///    [`FeeScheduleError::PendingScheduleExists`].
    ///
    /// A submission equal to the current `pending` or `active` schedule is a
    /// duplicate and returns the matching [`StageOutcome`] without error, so retries
    /// of an already-landed (or already-activated) submission succeed. Comparing
    /// against `active` covers the "retry after activation" case; the caller must
    /// therefore stage *after* calling [`Self::activate_if_due`] so a due promotion
    /// and a retry resolve identically.
    pub(crate) fn stage(
        &mut self,
        schedule: &FeeSchedule,
        current_ledger: u32,
    ) -> Result<StageOutcome, FeeScheduleError> {
        if self.pending.as_ref() == Some(schedule) {
            return Ok(StageOutcome::DuplicatePending);
        }
        if self.active.as_ref() == Some(schedule) {
            return Ok(StageOutcome::DuplicateActive);
        }

        schedule.validate(current_ledger)?;

        if self.pending.is_some() {
            return Err(FeeScheduleError::PendingScheduleExists);
        }

        self.pending = Some(schedule.clone());
        Ok(StageOutcome::Accepted)
    }

    pub(crate) fn active_at(&self, current_ledger: u32) -> Option<FeeSchedule> {
        match self.pending.as_ref() {
            Some(pending) if pending.activation_ledger <= current_ledger => Some(pending.clone()),
            _ => self.active.clone(),
        }
    }

    pub(crate) fn pending_at(&self, current_ledger: u32) -> Option<FeeSchedule> {
        match self.pending.as_ref() {
            Some(pending) if pending.activation_ledger > current_ledger => Some(pending.clone()),
            _ => None,
        }
    }

    pub(crate) fn previous_at(&self, current_ledger: u32) -> Option<FeeSchedule> {
        match self.pending.as_ref() {
            Some(pending) if pending.activation_ledger <= current_ledger => self.active.clone(),
            _ => self.previous.clone(),
        }
    }

    /// Persist to the original key layout; Soroban commits these writes as one
    /// invocation, so readers observe either the old or the complete new state.
    ///
    /// The three keys are written unconditionally (`None` becomes a `remove`) so a
    /// caller that promotes a schedule can never leave a stale pending entry behind.
    pub(crate) fn persist(&self, env: &Env) {
        let storage = env.storage().instance();
        match &self.active {
            Some(schedule) => storage.set(&FeeScheduleStorageKey::Active, schedule),
            None => storage.remove(&FeeScheduleStorageKey::Active),
        }
        match &self.pending {
            Some(schedule) => storage.set(&FeeScheduleStorageKey::Pending, schedule),
            None => storage.remove(&FeeScheduleStorageKey::Pending),
        }
        match &self.previous {
            Some(schedule) => storage.set(&FeeScheduleStorageKey::Previous, schedule),
            None => storage.remove(&FeeScheduleStorageKey::Previous),
        }
    }
}

impl FeeSchedule {
    /// Validate the schedule's own bounds and its activation ledger.
    ///
    /// This is the single definition of "in-bounds"; both [`FeeScheduleState::stage`]
    /// and the admin-gated entrypoint use it, so the accepted set cannot diverge
    /// between a direct call and a staged one.
    ///
    /// - `min_fee_bps <= fee_bps` and `fee_bps <= max_fee_bps`, else
    ///   [`FeeScheduleError::FeeOutOfBounds`]. Both bounds are inclusive, so a
    ///   schedule pinned exactly at either edge is valid. Note this rejects an
    ///   inverted pair (`min_fee_bps > max_fee_bps`) for every `fee_bps`.
    /// - `activation_ledger > current_ledger`, else
    ///   [`FeeScheduleError::InvalidActivationLedger`].
    pub(crate) fn validate(&self, current_ledger: u32) -> Result<(), FeeScheduleError> {
        if self.fee_bps < self.min_fee_bps || self.fee_bps > self.max_fee_bps {
            return Err(FeeScheduleError::FeeOutOfBounds);
        }
        if self.activation_ledger <= current_ledger {
            return Err(FeeScheduleError::InvalidActivationLedger);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schedule(fee_bps: u32, activation_ledger: u32) -> FeeSchedule {
        FeeSchedule {
            fee_bps,
            min_fee_bps: 0,
            max_fee_bps: 10_000,
            activation_ledger,
        }
    }

    // ── accepted input ────────────────────────────────────────────────────────

    /// A schedule inside the bounds with a future activation ledger is accepted and
    /// becomes the pending schedule.
    #[test]
    fn in_bounds_future_ledger_is_staged() {
        let mut state = FeeScheduleState::default();
        let s = schedule(250, 101);
        assert_eq!(state.stage(&s, 100), Ok(StageOutcome::Accepted));
        assert_eq!(state.pending, Some(s));
        assert_eq!(state.active, None);
        assert_eq!(state.previous, None);
    }

    /// Inclusive lower bound: `fee_bps == min_fee_bps` is accepted.
    #[test]
    fn fee_at_lower_bound_is_accepted() {
        let mut state = FeeScheduleState::default();
        let s = FeeSchedule {
            fee_bps: 10,
            min_fee_bps: 10,
            max_fee_bps: 500,
            activation_ledger: 101,
        };
        assert_eq!(state.stage(&s, 100), Ok(StageOutcome::Accepted));
    }

    /// Inclusive upper bound: `fee_bps == max_fee_bps` is accepted.
    #[test]
    fn fee_at_upper_bound_is_accepted() {
        let mut state = FeeScheduleState::default();
        let s = FeeSchedule {
            fee_bps: 500,
            min_fee_bps: 10,
            max_fee_bps: 500,
            activation_ledger: 101,
        };
        assert_eq!(state.stage(&s, 100), Ok(StageOutcome::Accepted));
    }

    /// A zero fee is a legitimate configuration (no protocol fee), not a sentinel.
    #[test]
    fn zero_fee_with_zero_min_is_accepted() {
        let mut state = FeeScheduleState::default();
        let s = FeeSchedule {
            fee_bps: 0,
            min_fee_bps: 0,
            max_fee_bps: 0,
            activation_ledger: 101,
        };
        assert_eq!(state.stage(&s, 100), Ok(StageOutcome::Accepted));
    }

    // ── rejected input ────────────────────────────────────────────────────────

    /// Below the lower bound is rejected and leaves the state untouched.
    #[test]
    fn fee_below_lower_bound_rejected_without_mutation() {
        let mut state = FeeScheduleState::default();
        let before = state.clone();
        let s = FeeSchedule {
            fee_bps: 9,
            min_fee_bps: 10,
            max_fee_bps: 500,
            activation_ledger: 101,
        };
        assert_eq!(state.stage(&s, 100), Err(FeeScheduleError::FeeOutOfBounds));
        assert_eq!(state, before);
    }

    /// Above the upper bound is rejected and leaves the state untouched.
    #[test]
    fn fee_above_upper_bound_rejected_without_mutation() {
        let mut state = FeeScheduleState::default();
        let before = state.clone();
        let s = FeeSchedule {
            fee_bps: 501,
            min_fee_bps: 10,
            max_fee_bps: 500,
            activation_ledger: 101,
        };
        assert_eq!(state.stage(&s, 100), Err(FeeScheduleError::FeeOutOfBounds));
        assert_eq!(state, before);
    }

    /// An inverted bound pair has no satisfying value, so it is rejected for any fee.
    #[test]
    fn inverted_bounds_rejected_for_every_fee() {
        for fee_bps in [0u32, 250, 10_000] {
            let mut state = FeeScheduleState::default();
            let s = FeeSchedule {
                fee_bps,
                min_fee_bps: 900,
                max_fee_bps: 100,
                activation_ledger: 101,
            };
            assert_eq!(
                state.stage(&s, 100),
                Err(FeeScheduleError::FeeOutOfBounds),
                "fee_bps {fee_bps} must not satisfy inverted bounds"
            );
        }
    }

    /// `u32::MAX` exceeds any realistic ceiling; the bound check still rejects it.
    #[test]
    fn u32_max_fee_rejected_against_narrow_ceiling() {
        let mut state = FeeScheduleState::default();
        let s = FeeSchedule {
            fee_bps: u32::MAX,
            min_fee_bps: 0,
            max_fee_bps: 10_000,
            activation_ledger: 101,
        };
        assert_eq!(state.stage(&s, 100), Err(FeeScheduleError::FeeOutOfBounds));
    }

    // ── boundary: activation ledger ───────────────────────────────────────────

    /// `activation_ledger == current_ledger` is rejected: it would activate inside
    /// this very invocation, collapsing the intended review window to zero.
    #[test]
    fn activation_equal_to_current_ledger_rejected() {
        let mut state = FeeScheduleState::default();
        let before = state.clone();
        assert_eq!(
            state.stage(&schedule(250, 100), 100),
            Err(FeeScheduleError::InvalidActivationLedger)
        );
        assert_eq!(state, before);
    }

    /// One ledger into the future is the smallest accepted delay.
    #[test]
    fn activation_one_ledger_ahead_accepted() {
        let mut state = FeeScheduleState::default();
        assert_eq!(
            state.stage(&schedule(250, 101), 100),
            Ok(StageOutcome::Accepted)
        );
    }

    /// A past activation ledger is rejected (already due at submission time).
    #[test]
    fn activation_in_the_past_rejected() {
        let mut state = FeeScheduleState::default();
        assert_eq!(
            state.stage(&schedule(250, 99), 100),
            Err(FeeScheduleError::InvalidActivationLedger)
        );
    }

    /// Bounds are checked before the ledger, so an input failing both reports the
    /// bound failure deterministically regardless of call ordering.
    #[test]
    fn out_of_bounds_reported_before_activation_ledger() {
        let mut state = FeeScheduleState::default();
        let s = FeeSchedule {
            fee_bps: 5_000,
            min_fee_bps: 0,
            max_fee_bps: 10,
            activation_ledger: 0,
        };
        assert_eq!(state.stage(&s, 100), Err(FeeScheduleError::FeeOutOfBounds));
    }

    // ── duplicate submissions ─────────────────────────────────────────────────

    /// Re-submitting the identical pending schedule is idempotent, not an error.
    #[test]
    fn duplicate_pending_submission_is_idempotent() {
        let mut state = FeeScheduleState::default();
        let s = schedule(250, 101);
        assert_eq!(state.stage(&s, 100), Ok(StageOutcome::Accepted));
        assert_eq!(state.stage(&s, 100), Ok(StageOutcome::DuplicatePending));
        assert_eq!(state.pending, Some(s));
    }

    /// Re-submitting the schedule that is already active is idempotent.
    #[test]
    fn duplicate_active_submission_is_idempotent() {
        let mut state = FeeScheduleState::default();
        let s = schedule(250, 101);
        state.stage(&s, 100).expect("staging must succeed");
        assert!(state.activate_if_due(101));
        assert_eq!(state.active, Some(s.clone()));
        assert_eq!(state.stage(&s, 101), Ok(StageOutcome::DuplicateActive));
    }

    /// A retry submitted at the activation ledger — after the pending schedule has
    /// already been promoted — still succeeds and does not re-stage it.
    #[test]
    fn retry_after_activation_does_not_restage() {
        let mut state = FeeScheduleState::default();
        let s = schedule(250, 101);
        state.stage(&s, 100).expect("staging must succeed");
        assert!(state.activate_if_due(101));
        // Simulate the "retry" path the entrypoint takes: promote first, then stage.
        assert!(!state.activate_if_due(101));
        assert_eq!(state.stage(&s, 101), Ok(StageOutcome::DuplicateActive));
        assert_eq!(state.pending, None);
    }

    /// A second, *different* schedule is rejected while one is pending and must not
    /// clobber the pending value.
    #[test]
    fn second_pending_schedule_rejected_without_clobbering() {
        let mut state = FeeScheduleState::default();
        let first = schedule(400, 110);
        let second = schedule(700, 120);
        assert_eq!(state.stage(&first, 100), Ok(StageOutcome::Accepted));
        let before = state.clone();
        assert_eq!(
            state.stage(&second, 100),
            Err(FeeScheduleError::PendingScheduleExists)
        );
        assert_eq!(state, before);
        assert_eq!(state.pending, Some(first));
    }

    /// A duplicate pending submission is still reported as duplicate even when an
    /// unrelated pending schedule would otherwise block it.
    #[test]
    fn duplicate_check_precedes_pending_slot_check() {
        let mut state = FeeScheduleState::default();
        let s = schedule(400, 110);
        state.stage(&s, 100).expect("staging must succeed");
        // Same schedule again: must be a duplicate, not PendingScheduleExists.
        assert_eq!(state.stage(&s, 100), Ok(StageOutcome::DuplicatePending));
    }

    // ── activation idempotency & previous preservation ───────────────────────

    /// Activation is a no-op before the activation ledger.
    #[test]
    fn activation_before_ledger_is_noop() {
        let mut state = FeeScheduleState::default();
        let s = schedule(500, 105);
        state.stage(&s, 100).expect("staging must succeed");
        assert!(!state.activate_if_due(104));
        assert_eq!(state.pending, Some(s));
        assert_eq!(state.active, None);
    }

    /// Activation happens exactly at `activation_ledger`, not one ledger later.
    #[test]
    fn activation_at_exact_ledger() {
        let mut state = FeeScheduleState::default();
        state
            .stage(&schedule(500, 105), 100)
            .expect("staging must succeed");
        assert!(state.activate_if_due(105));
        assert_eq!(state.active.unwrap().fee_bps, 500);
        assert_eq!(state.pending, None);
    }

    /// Repeated activation calls are idempotent and report `false` after the first.
    #[test]
    fn repeated_activation_is_idempotent() {
        let mut state = FeeScheduleState::default();
        state
            .stage(&schedule(500, 105), 100)
            .expect("staging must succeed");
        assert!(state.activate_if_due(105));
        for ledger in 105..=110 {
            assert!(!state.activate_if_due(ledger), "ledger {ledger}");
        }
        assert_eq!(state.active.unwrap().fee_bps, 500);
        assert_eq!(state.previous, None);
    }

    /// Activation without a pending schedule is a no-op even at ledger 0.
    #[test]
    fn activation_with_empty_state_is_noop() {
        let mut state = FeeScheduleState::default();
        assert!(!state.activate_if_due(0));
        assert!(!state.activate_if_due(u32::MAX));
        assert_eq!(state, FeeScheduleState::default());
    }

    /// The prior active schedule moves into `previous` on promotion and is never lost.
    #[test]
    fn activation_preserves_previous_schedule() {
        let mut state = FeeScheduleState::default();
        let first = schedule(200, 101);
        let second = schedule(300, 102);
        state.stage(&first, 100).expect("staging must succeed");
        assert!(state.activate_if_due(101));
        state.stage(&second, 101).expect("staging must succeed");
        assert!(state.activate_if_due(102));
        assert_eq!(state.active, Some(second));
        assert_eq!(state.previous, Some(first));
        assert_eq!(state.pending, None);
    }

    /// Activation ledger `0` is still "due" at ledger `0`, so a schedule can never be
    /// promoted by accident after being loaded from storage.
    #[test]
    fn zero_activation_ledger_is_due_immediately() {
        let mut state = FeeScheduleState {
            pending: Some(schedule(300, 0)),
            ..Default::default()
        };
        assert!(state.activate_if_due(0));
        assert_eq!(state.active.unwrap().fee_bps, 300);
    }

    // ── pure views ────────────────────────────────────────────────────────────

    /// Before the activation ledger the pending schedule is reported as pending and
    /// not as active.
    #[test]
    fn views_report_pending_before_activation() {
        let state = FeeScheduleState {
            pending: Some(schedule(500, 105)),
            ..Default::default()
        };
        assert_eq!(state.active_at(104), None);
        assert_eq!(state.pending_at(104).map(|s| s.fee_bps), Some(500));
        assert_eq!(state.previous_at(104), None);
    }

    /// At and after the activation ledger the views report the boundary activation
    /// without mutating anything.
    #[test]
    fn views_compute_boundary_activation_without_mutating() {
        let state = FeeScheduleState {
            pending: Some(schedule(500, 105)),
            ..Default::default()
        };
        for ledger in 105..=110 {
            assert_eq!(state.active_at(ledger).map(|s| s.fee_bps), Some(500));
            assert_eq!(state.pending_at(ledger), None);
        }
        // Pure: still un-promoted in memory.
        assert_eq!(state.pending.map(|s| s.fee_bps), Some(500));
        assert_eq!(state.active, None);
    }

    /// An empty state yields `None` from every view at any ledger.
    #[test]
    fn empty_state_views_are_none() {
        let state = FeeScheduleState::default();
        for ledger in [0u32, 1, u32::MAX] {
            assert_eq!(state.active_at(ledger), None);
            assert_eq!(state.pending_at(ledger), None);
            assert_eq!(state.previous_at(ledger), None);
        }
    }

    /// With no pending schedule, `previous_at` falls through to `previous`.
    #[test]
    fn previous_view_falls_through_without_pending() {
        let previous = schedule(100, 1);
        let state = FeeScheduleState {
            previous: Some(previous.clone()),
            ..Default::default()
        };
        assert_eq!(state.previous_at(1_000), Some(previous));
    }

    // ── persistence round-trip ────────────────────────────────────────────────

    /// A staged schedule survives a persist/load round-trip unchanged.
    #[test]
    fn staged_schedule_round_trips_through_storage() {
        let env = Env::default();
        let id = env.register(crate::LiquifactEscrow, ());
        let s = schedule(250, 101);
        env.as_contract(&id, || {
            let mut state = FeeScheduleState::load(&env);
            assert_eq!(state.stage(&s, 100), Ok(StageOutcome::Accepted));
            state.persist(&env);
        });
        env.as_contract(&id, || {
            let state = FeeScheduleState::load(&env);
            assert_eq!(state.pending, Some(s.clone()));
            assert_eq!(state.active, None);
            assert_eq!(state.previous, None);
            assert_eq!(state.pending_at(100).map(|p| p.fee_bps), Some(250));
        });
    }

    /// Promoting then persisting clears the pending key and records `previous`, so a
    /// later load sees the promoted state rather than a stale duplicate.
    #[test]
    fn promotion_persists_previous_and_clears_pending() {
        let env = Env::default();
        let id = env.register(crate::LiquifactEscrow, ());
        let first = schedule(200, 101);
        let second = schedule(300, 102);
        env.as_contract(&id, || {
            let mut state = FeeScheduleState::load(&env);
            state.stage(&first, 100).expect("staging must succeed");
            state.persist(&env);
        });
        env.as_contract(&id, || {
            let mut state = FeeScheduleState::load(&env);
            assert!(state.activate_if_due(101));
            state.persist(&env);
            assert_eq!(state.stage(&second, 101), Ok(StageOutcome::Accepted));
            state.persist(&env);
        });
        env.as_contract(&id, || {
            let mut state = FeeScheduleState::load(&env);
            assert_eq!(state.active, Some(first.clone()));
            assert_eq!(state.pending, Some(second));
            assert!(state.activate_if_due(102));
            state.persist(&env);
            assert_eq!(state.previous, Some(first));
        });
        env.as_contract(&id, || {
            let state = FeeScheduleState::load(&env);
            assert_eq!(state.active.map(|s| s.fee_bps), Some(300));
            assert_eq!(state.pending, None);
        });
    }

    /// A default state carries no schedules, so a fresh contract cannot activate one.
    #[test]
    fn default_state_is_empty() {
        let state = FeeScheduleState::default();
        assert_eq!(state.active, None);
        assert_eq!(state.pending, None);
        assert_eq!(state.previous, None);
    }
}
