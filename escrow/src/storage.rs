//! Fee-schedule persistence: the single write path for the fee-schedule triple.
//!
//! All validation lives in [`crate::types`]; this module only owns storage layout,
//! authorization, and the mutual-exclusion lock that keeps overlapping transitions
//! from interleaving.
//!
//! # Invariants
//!
//! 1. **Single write path.** Every mutation of [`FeeScheduleStorageKey::Active`],
//!    [`FeeScheduleStorageKey::Pending`], or [`FeeScheduleStorageKey::Previous`]
//!    goes through [`FeeScheduleState::persist`] (re-exported here), so a transition
//!    cannot leave the triple inconsistent.
//! 2. **Staged, then persisted.** Validation runs against a loaded
//!    [`FeeScheduleState`] and only a `Ok` result is persisted. A rejected
//!    submission writes nothing.
//! 3. **Mutual exclusion.** [`acquire_mutation_lock`] rejects a re-entrant or
//!    overlapping write with [`FeeScheduleError::ConcurrentMutation`] and changes
//!    nothing. The lock is *released on every exit path*, including the error paths,
//!    so a failed attempt cannot wedge the contract.
//!
//! # Failure modes
//!
//! | Condition | Result |
//! |-----------|--------|
//! | Not initialized | [`FeeScheduleError::NotInitialized`] |
//! | Another mutation in flight | [`FeeScheduleError::ConcurrentMutation`] |
//! | `fee_bps` outside `[min_fee_bps, max_fee_bps]` | [`FeeScheduleError::FeeOutOfBounds`] |
//! | `activation_ledger <= current_ledger` | [`FeeScheduleError::InvalidActivationLedger`] |
//! | A different schedule is already pending | [`FeeScheduleError::PendingScheduleExists`] |
//! | Duplicate of pending / active schedule | success, no write |
//!
//! **Backwards compatibility:** [`FeeScheduleStorageKey::MutationLock`] is an
//! additive key (ADR-007). An instance written before this guard has no lock entry,
//! which reads as unlocked and therefore behaves exactly as it did before — no
//! migration is required.
//!
//! **Retry safety:** the lock is released before the error propagates and Soroban
//! rolls the invocation back on a panic, so a retried submission always starts from
//! an unlocked state and cannot observe a half-applied transition.

use crate::{
    types::{FeeScheduleState, StageOutcome},
    Address, Env, FeeSchedule, FeeScheduleError, FeeScheduleStorageKey,
};

/// Take the fee-schedule mutation lock, or fail if it is already held.
///
/// # Rejecting (not waiting) is deliberate
///
/// A blocking wait is not expressible in the Soroban host and would let a
/// re-entrant call observe partially staged state. Failing fast with
/// [`FeeScheduleError::ConcurrentMutation`] keeps the boundary observable and
/// deterministic: the caller can retry, and the retry sees a clean state.
pub(crate) fn acquire_mutation_lock(env: &Env) -> Result<(), FeeScheduleError> {
    let storage = env.storage().instance();
    let held: bool = storage
        .get(&FeeScheduleStorageKey::MutationLock)
        .unwrap_or(false);
    if held {
        return Err(FeeScheduleError::ConcurrentMutation);
    }
    storage.set(&FeeScheduleStorageKey::MutationLock, &true);
    Ok(())
}

/// Release the fee-schedule mutation lock. Must be called on every exit path,
/// including the error paths, so a rejected transition never strands the lock.
pub(crate) fn release_mutation_lock(env: &Env) {
    env.storage()
        .instance()
        .remove(&FeeScheduleStorageKey::MutationLock);
}

/// Admin-authorized submission of a new pending fee schedule.
///
/// Returns [`StageOutcome`] so the caller knows whether a write is required.
/// `Accepted` mutates and must be persisted; the duplicate outcomes are no-ops.
///
/// Authorization is required **before** any storage write, and the mutation lock is
/// held only around the load/stage/persist sequence so a validation failure cannot
/// leave the contract locked.
pub(crate) fn submit_fee_schedule(
    env: &Env,
    admin: &Address,
    schedule: &FeeSchedule,
) -> Result<StageOutcome, FeeScheduleError> {
    admin.require_auth();

    let current_ledger = env.ledger().sequence();

    // Lock first: two overlapping invocations must not both load and stage, or the
    // loser's `persist` would overwrite the winner's promotion.
    acquire_mutation_lock(env)?;

    let mut state = FeeScheduleState::load(env);

    // Promote a due pending schedule before evaluating the submission so a retry
    // that arrives at or after the activation ledger resolves identically to one
    // that arrived before it.
    let promoted = state.activate_if_due(current_ledger);

    let outcome = state.stage(schedule, current_ledger);

    match outcome {
        // Accepted: the pending slot now holds the new schedule.
        Ok(StageOutcome::Accepted) => {
            state.persist(env);
            release_mutation_lock(env);
            Ok(StageOutcome::Accepted)
        }
        // Duplicate of the pending schedule: nothing changed.
        Ok(StageOutcome::DuplicatePending) => {
            if promoted {
                state.persist(env);
            }
            release_mutation_lock(env);
            Ok(StageOutcome::DuplicatePending)
        }
        // Duplicate of the active schedule: the submission is a retry of an
        // already-landed schedule. Persist only if this call also promoted.
        Ok(StageOutcome::DuplicateActive) => {
            if promoted {
                state.persist(env);
            }
            release_mutation_lock(env);
            Ok(StageOutcome::DuplicateActive)
        }
        // Rejected: release the lock and leave storage exactly as it was. `state`
        // may carry an in-memory promotion, but it is never persisted, so the
        // stored triple is untouched by a failed submission.
        Err(err) => {
            release_mutation_lock(env);
            Err(err)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::StageOutcome;
    use soroban_sdk::{
        testutils::{Address as _, Ledger as _},
        Address, Env,
    };

    fn schedule(fee_bps: u32, activation_ledger: u32) -> FeeSchedule {
        FeeSchedule {
            fee_bps,
            min_fee_bps: 0,
            max_fee_bps: 10_000,
            activation_ledger,
        }
    }

    fn harness(env: &Env) -> (soroban_sdk::Address, Address) {
        env.mock_all_auths();
        let id = env.register(crate::LiquifactEscrow, ());
        let admin = Address::generate(env);
        (id, admin)
    }

    /// The lock is absent on a fresh instance, so a first submission succeeds.
    #[test]
    fn first_submission_succeeds_and_locks_are_released() {
        let env = Env::default();
        env.ledger().set_sequence_number(100);
        let (id, admin) = harness(&env);
        env.as_contract(&id, || {
            assert_eq!(
                submit_fee_schedule(&env, &admin, &schedule(250, 101)),
                Ok(StageOutcome::Accepted)
            );
            assert_eq!(
                FeeScheduleState::load(&env).pending.map(|s| s.fee_bps),
                Some(250)
            );
            // Lock released on the success path.
            let held: bool = env
                .storage()
                .instance()
                .get(&FeeScheduleStorageKey::MutationLock)
                .unwrap_or(false);
            assert!(!held);
        });
    }

    /// A second, different schedule is rejected and the pending value is preserved.
    #[test]
    fn second_distinct_schedule_rejected_state_preserved() {
        let env = Env::default();
        env.ledger().set_sequence_number(100);
        let (id, admin) = harness(&env);
        env.as_contract(&id, || {
            submit_fee_schedule(&env, &admin, &schedule(400, 110)).unwrap();
        });
        env.as_contract(&id, || {
            assert_eq!(
                submit_fee_schedule(&env, &admin, &schedule(700, 120)),
                Err(FeeScheduleError::PendingScheduleExists)
            );
            assert_eq!(
                FeeScheduleState::load(&env).pending.map(|s| s.fee_bps),
                Some(400)
            );
        });
    }

    /// An out-of-bounds submission writes nothing and does not strand the lock, so a
    /// later valid submission still succeeds.
    #[test]
    fn rejected_submission_releases_lock_and_allows_retry() {
        let env = Env::default();
        env.ledger().set_sequence_number(100);
        let (id, admin) = harness(&env);
        env.as_contract(&id, || {
            let bad = FeeSchedule {
                fee_bps: 10_000,
                min_fee_bps: 0,
                max_fee_bps: 1_000,
                activation_ledger: 101,
            };
            assert_eq!(
                submit_fee_schedule(&env, &admin, &bad),
                Err(FeeScheduleError::FeeOutOfBounds)
            );
        });
        env.as_contract(&id, || {
            let held: bool = env
                .storage()
                .instance()
                .get(&FeeScheduleStorageKey::MutationLock)
                .unwrap_or(false);
            assert!(!held, "rejected submission must not strand the lock");
            assert_eq!(
                FeeScheduleState::load(&env).pending,
                None,
                "rejected submission must not write"
            );
        });
        // Retry in a fresh frame: the lock is free, so the same admin can proceed.
        env.as_contract(&id, || {
            assert_eq!(
                submit_fee_schedule(&env, &admin, &schedule(250, 101)),
                Ok(StageOutcome::Accepted)
            );
        });
    }

    /// An activation ledger at or before the current ledger is rejected without a write.
    #[test]
    fn invalid_activation_rejected_without_write() {
        let env = Env::default();
        env.ledger().set_sequence_number(100);
        let (id, admin) = harness(&env);
        env.as_contract(&id, || {
            assert_eq!(
                submit_fee_schedule(&env, &admin, &schedule(250, 100)),
                Err(FeeScheduleError::InvalidActivationLedger)
            );
            assert_eq!(FeeScheduleState::load(&env).pending, None);
        });
    }

    /// Re-submitting the identical pending schedule succeeds and does not rewrite it.
    #[test]
    fn duplicate_pending_submission_is_idempotent() {
        let env = Env::default();
        env.ledger().set_sequence_number(100);
        let (id, admin) = harness(&env);
        env.as_contract(&id, || {
            submit_fee_schedule(&env, &admin, &schedule(250, 101)).unwrap();
        });
        env.as_contract(&id, || {
            assert_eq!(
                submit_fee_schedule(&env, &admin, &schedule(250, 101)),
                Ok(StageOutcome::DuplicatePending)
            );
            assert_eq!(
                FeeScheduleState::load(&env).pending.map(|s| s.fee_bps),
                Some(250)
            );
        });
    }

    /// A retry submitted at the activation ledger succeeds and promotes the schedule
    /// exactly once.
    #[test]
    fn retry_at_activation_ledger_promotes_once() {
        let env = Env::default();
        env.ledger().set_sequence_number(100);
        let (id, admin) = harness(&env);
        let s = schedule(250, 101);
        env.as_contract(&id, || {
            submit_fee_schedule(&env, &admin, &s).unwrap();
        });
        env.ledger().set_sequence_number(101);
        env.as_contract(&id, || {
            assert_eq!(
                submit_fee_schedule(&env, &admin, &s),
                Ok(StageOutcome::DuplicateActive)
            );
            let state = FeeScheduleState::load(&env);
            assert_eq!(state.active.map(|a| a.fee_bps), Some(250));
            assert_eq!(state.pending, None);
        });
    }

    /// While the lock is held, a submission is rejected with `ConcurrentMutation`
    /// and changes nothing — the re-entrancy guard is observable, not silent.
    #[test]
    fn held_lock_rejects_overlapping_submission() {
        let env = Env::default();
        env.ledger().set_sequence_number(100);
        let (id, admin) = harness(&env);
        env.as_contract(&id, || {
            acquire_mutation_lock(&env).unwrap();
        });
        env.as_contract(&id, || {
            assert_eq!(
                submit_fee_schedule(&env, &admin, &schedule(250, 101)),
                Err(FeeScheduleError::ConcurrentMutation)
            );
            assert_eq!(FeeScheduleState::load(&env).pending, None);
        });
        // Once the lock is released, the same submission succeeds.
        env.as_contract(&id, || {
            release_mutation_lock(&env);
        });
        env.as_contract(&id, || {
            assert_eq!(
                submit_fee_schedule(&env, &admin, &schedule(250, 101)),
                Ok(StageOutcome::Accepted)
            );
        });
    }

    /// Acquiring twice fails, and the failed attempt leaves the lock held for the
    /// original owner rather than clearing it.
    #[test]
    fn double_acquire_fails_without_clearing_lock() {
        let env = Env::default();
        let (id, _admin) = harness(&env);
        env.as_contract(&id, || {
            acquire_mutation_lock(&env).unwrap();
        });
        env.as_contract(&id, || {
            assert_eq!(
                acquire_mutation_lock(&env),
                Err(FeeScheduleError::ConcurrentMutation)
            );
            let held: bool = env
                .storage()
                .instance()
                .get(&FeeScheduleStorageKey::MutationLock)
                .unwrap_or(false);
            assert!(held, "a failed acquire must not release the existing lock");
            release_mutation_lock(&env);
        });
    }

    /// An instance with no lock key (pre-guard data) reads as unlocked, preserving
    /// backwards-compatible behavior.
    #[test]
    fn absent_lock_reads_as_unlocked() {
        let env = Env::default();
        let (id, _admin) = harness(&env);
        env.as_contract(&id, || {
            assert!(!env
                .storage()
                .instance()
                .has(&FeeScheduleStorageKey::MutationLock));
            assert!(acquire_mutation_lock(&env).is_ok());
        });
    }
}
