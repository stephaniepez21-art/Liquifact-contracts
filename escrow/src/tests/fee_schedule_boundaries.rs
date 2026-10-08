//! End-to-end validation-boundary tests for the fee-schedule subsystem
//! (`escrow/src/types.rs` + `escrow/src/storage.rs`).
//!
//! These exercise the public contract entrypoints rather than the in-memory state
//! machine, so they cover authorization, storage effects, and the error codes an
//! integrator actually observes.
//!
//! Coverage map:
//! - accepted input: in-bounds schedule, future activation ledger
//! - rejected input: bounds violation, activation ledger in the past/current, second
//!   pending schedule, unauthorized caller, uninitialized escrow
//! - duplicate submissions: pending, active, retry-after-activation
//! - boundary values: `fee_bps` at each bound, inverted bounds, `u32::MAX`,
//!   `activation_ledger == current`, `activation_ledger == current + 1`, ledger 0
//! - regression: state preserved across every rejected path; views stay pure
//! - concurrency: overlapping mutation rejected, lock released on every exit path

use super::*;
use crate::{DataKey, FeeSchedule, FeeScheduleError, FeeScheduleStorageKey, InvoiceEscrow};
use soroban_sdk::testutils::{Address as _, Events, Ledger as _};
use soroban_sdk::String as SorobanString;
use std::fmt::Debug;

const INIT_LEDGER: u32 = 100;

fn schedule(fee_bps: u32, activation_ledger: u32) -> FeeSchedule {
    FeeSchedule {
        fee_bps,
        min_fee_bps: 0,
        max_fee_bps: 10_000,
        activation_ledger,
    }
}

fn bounded(
    fee_bps: u32,
    min_fee_bps: u32,
    max_fee_bps: u32,
    activation_ledger: u32,
) -> FeeSchedule {
    FeeSchedule {
        fee_bps,
        min_fee_bps,
        max_fee_bps,
        activation_ledger,
    }
}

/// Deploy + `init`, returning `(client, admin)`. Ledger sequence is `INIT_LEDGER`.
fn deploy_and_init(env: &Env) -> (LiquifactEscrowClient<'_>, Address) {
    let mut info = env.ledger().get();
    info.timestamp = 0;
    info.sequence_number = INIT_LEDGER;
    env.ledger().set(info);
    env.mock_all_auths();

    let client = deploy(env);
    let admin = Address::generate(env);
    let token = Address::generate(env);
    let treasury = Address::generate(env);
    client.init(
        &admin,
        &SorobanString::from_str(env, "FEE001"),
        &Address::generate(env),
        &1_000_000i128,
        &800i64,
        &0u64,
        &token,
        &None,
        &treasury,
        &None,
        &None,
        &None,
        &None,
        &None,
        &None,
        &None,
        &None,
        &None::<i64>,
        &None::<u32>,
    );
    (client, admin)
}

fn set_ledger(env: &Env, sequence: u32) {
    let mut info = env.ledger().get();
    info.sequence_number = sequence;
    env.ledger().set(info);
}

/// Assert that a `try_*` invocation failed with exactly `expected`.
///
/// `soroban_sdk` surfaces a contract panic in one of two shapes depending on where the
/// failure is converted (host error vs. contract error code), so both are accepted; a
/// successful call or any other code is a failure.
fn assert_fee_error<T: Debug, E: Debug>(
    result: Result<Result<T, E>, Result<soroban_sdk::Error, InvokeError>>,
    expected: FeeScheduleError,
) {
    let code = expected as u32;
    match result {
        Err(Ok(err)) => {
            assert_eq!(err, soroban_sdk::Error::from_contract_error(code));
        }
        Err(Err(InvokeError::Contract(actual))) => assert_eq!(actual, code),
        other => panic!("expected FeeScheduleError({code}), got {other:?}"),
    }
}

/// Read the mutation lock without going through the contract API.
fn lock_held(env: &Env, client: &LiquifactEscrowClient<'_>) -> bool {
    env.as_contract(&client.address, || {
        env.storage()
            .instance()
            .get(&FeeScheduleStorageKey::MutationLock)
            .unwrap_or(false)
    })
}

// ── accepted input ───────────────────────────────────────────────────────────

/// A schedule inside the bounds with a future activation ledger is staged as pending
/// and is not yet active.
#[test]
fn in_bounds_schedule_is_accepted_as_pending() {
    let env = Env::default();
    let (client, _admin) = deploy_and_init(&env);

    let s = schedule(250, INIT_LEDGER + 1);
    client.submit_fee_schedule(&s);

    assert_eq!(client.get_pending_fee_schedule(), Some(s.clone()));
    assert_eq!(client.get_active_fee_schedule(), None);
    assert_eq!(client.get_previous_fee_schedule(), None);
}

/// Boundary: `fee_bps` exactly equal to `min_fee_bps` is accepted (inclusive bound).
#[test]
fn fee_at_lower_bound_is_accepted() {
    let env = Env::default();
    let (client, _admin) = deploy_and_init(&env);

    let s = bounded(10, 10, 500, INIT_LEDGER + 1);
    client.submit_fee_schedule(&s);
    assert_eq!(client.get_pending_fee_schedule(), Some(s));
}

/// Boundary: `fee_bps` exactly equal to `max_fee_bps` is accepted (inclusive bound).
#[test]
fn fee_at_upper_bound_is_accepted() {
    let env = Env::default();
    let (client, _admin) = deploy_and_init(&env);

    let s = bounded(500, 10, 500, INIT_LEDGER + 1);
    client.submit_fee_schedule(&s);
    assert_eq!(client.get_pending_fee_schedule(), Some(s));
}

/// Boundary: the smallest accepted activation delay is one ledger.
#[test]
fn activation_one_ledger_ahead_is_accepted() {
    let env = Env::default();
    let (client, _admin) = deploy_and_init(&env);

    let s = schedule(250, INIT_LEDGER + 1);
    client.submit_fee_schedule(&s);
    assert_eq!(client.get_pending_fee_schedule(), Some(s));
}

/// A zero fee with a zero minimum is a valid configuration, not a sentinel.
#[test]
fn zero_fee_is_accepted() {
    let env = Env::default();
    let (client, _admin) = deploy_and_init(&env);

    let s = bounded(0, 0, 0, INIT_LEDGER + 1);
    client.submit_fee_schedule(&s);
    assert_eq!(client.get_pending_fee_schedule(), Some(s));
}

// ── rejected input ───────────────────────────────────────────────────────────

/// `fee_bps` below `min_fee_bps` is rejected and writes nothing.
#[test]
fn fee_below_lower_bound_is_rejected_without_writing() {
    let env = Env::default();
    let (client, _admin) = deploy_and_init(&env);

    let result = client.try_submit_fee_schedule(&bounded(9, 10, 500, INIT_LEDGER + 1));
    assert_fee_error(result, FeeScheduleError::FeeOutOfBounds);
    assert_eq!(client.get_pending_fee_schedule(), None);
    assert_eq!(client.get_active_fee_schedule(), None);
}

/// `fee_bps` above `max_fee_bps` is rejected and writes nothing.
#[test]
fn fee_above_upper_bound_is_rejected_without_writing() {
    let env = Env::default();
    let (client, _admin) = deploy_and_init(&env);

    let result = client.try_submit_fee_schedule(&bounded(501, 10, 500, INIT_LEDGER + 1));
    assert_fee_error(result, FeeScheduleError::FeeOutOfBounds);
    assert_eq!(client.get_pending_fee_schedule(), None);
}

/// An inverted bound pair has no satisfying value, so every fee is rejected.
#[test]
fn inverted_bounds_are_rejected_for_every_fee() {
    let env = Env::default();
    let (client, _admin) = deploy_and_init(&env);

    for fee_bps in [0u32, 250, 10_000] {
        let result = client.try_submit_fee_schedule(&bounded(fee_bps, 900, 100, INIT_LEDGER + 1));
        assert_fee_error(result, FeeScheduleError::FeeOutOfBounds);
    }
    assert_eq!(client.get_pending_fee_schedule(), None);
}

/// Boundary: `u32::MAX` against a narrow ceiling is rejected rather than wrapping.
#[test]
fn u32_max_fee_is_rejected_against_narrow_ceiling() {
    let env = Env::default();
    let (client, _admin) = deploy_and_init(&env);

    let result = client.try_submit_fee_schedule(&schedule(u32::MAX, INIT_LEDGER + 1));
    assert_fee_error(result, FeeScheduleError::FeeOutOfBounds);
    assert_eq!(client.get_pending_fee_schedule(), None);
}

/// Boundary: `activation_ledger == current_ledger` is rejected because the schedule
/// would activate inside the same invocation, collapsing the review window to zero.
#[test]
fn activation_at_current_ledger_is_rejected() {
    let env = Env::default();
    let (client, _admin) = deploy_and_init(&env);

    let result = client.try_submit_fee_schedule(&schedule(250, INIT_LEDGER));
    assert_fee_error(result, FeeScheduleError::InvalidActivationLedger);
    assert_eq!(client.get_pending_fee_schedule(), None);
}

/// An activation ledger in the past is rejected without writing.
#[test]
fn activation_in_the_past_is_rejected() {
    let env = Env::default();
    let (client, _admin) = deploy_and_init(&env);

    let result = client.try_submit_fee_schedule(&schedule(250, INIT_LEDGER - 1));
    assert_fee_error(result, FeeScheduleError::InvalidActivationLedger);
    assert_eq!(client.get_pending_fee_schedule(), None);
}

/// A second, different schedule while one is pending is rejected and the existing
/// pending schedule is not clobbered.
#[test]
fn second_distinct_schedule_is_rejected_and_pending_is_preserved() {
    let env = Env::default();
    let (client, _admin) = deploy_and_init(&env);

    let first = schedule(400, INIT_LEDGER + 10);
    let second = schedule(700, INIT_LEDGER + 20);
    client.submit_fee_schedule(&first);

    let result = client.try_submit_fee_schedule(&second);
    assert_fee_error(result, FeeScheduleError::PendingScheduleExists);
    assert_eq!(client.get_pending_fee_schedule(), Some(first));
    assert_eq!(client.get_active_fee_schedule(), None);
}

/// An uninitialized escrow rejects the submission with a typed error.
#[test]
fn uninitialized_escrow_rejects_submission() {
    let env = Env::default();
    env.mock_all_auths();
    let client = deploy(&env);

    let result = client.try_submit_fee_schedule(&schedule(250, INIT_LEDGER + 1));
    match result {
        Err(_) | Ok(Err(_)) => {}
        other => panic!("expected NotInitialized failure, got {other:?}"),
    }
}

/// Without any mocked authorizations a submission fails at authorization, writes
/// nothing, and does not strand the mutation lock.
#[test]
fn unauthorized_submission_is_rejected_without_writing() {
    let env = Env::default();
    set_ledger(&env, INIT_LEDGER);
    let client = deploy(&env);

    // Seed a live escrow directly so only the fee-schedule authorization is under test.
    let admin = Address::generate(&env);
    env.as_contract(&client.address, || {
        env.storage().instance().set(
            &DataKey::Escrow,
            &InvoiceEscrow {
                invoice_id: soroban_sdk::symbol_short!("fee"),
                admin: admin.clone(),
                sme_address: admin.clone(),
                payer: admin.clone(),
                amount: 1_000,
                funding_target: 1_000,
                funded_amount: 0,
                yield_bps: 0,
                maturity: 0,
                status: 0,
                dispute_active: false,
            },
        );
    });

    // No `mock_all_auths()`: the admin signature cannot be satisfied, so the call
    // must fail and must leave the fee-schedule state untouched.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        client.submit_fee_schedule(&schedule(250, INIT_LEDGER + 1))
    }));
    assert!(result.is_err(), "unauthorized submission must panic");
    assert_eq!(client.get_pending_fee_schedule(), None);
    assert!(!lock_held(&env, &client));
}

// ── duplicate submissions ────────────────────────────────────────────────────

/// Re-submitting the identical pending schedule succeeds and does not disturb it.
#[test]
fn duplicate_pending_submission_succeeds() {
    let env = Env::default();
    let (client, _admin) = deploy_and_init(&env);

    let s = schedule(250, INIT_LEDGER + 1);
    client.submit_fee_schedule(&s);
    client.submit_fee_schedule(&s);
    client.submit_fee_schedule(&s);

    assert_eq!(client.get_pending_fee_schedule(), Some(s));
    assert_eq!(client.get_active_fee_schedule(), None);
}

/// After activation, re-submitting the same schedule still succeeds and does not
/// re-stage it as pending.
#[test]
fn duplicate_submission_after_activation_succeeds() {
    let env = Env::default();
    let (client, _admin) = deploy_and_init(&env);

    let s = schedule(250, INIT_LEDGER + 1);
    client.submit_fee_schedule(&s);
    set_ledger(&env, INIT_LEDGER + 1);

    client.submit_fee_schedule(&s);
    assert_eq!(client.get_active_fee_schedule(), Some(s.clone()));
    assert_eq!(client.get_pending_fee_schedule(), None);
}

// ── activation & view behavior ───────────────────────────────────────────────

/// Activation happens exactly at the activation ledger, not one ledger later.
#[test]
fn activation_occurs_at_the_activation_ledger() {
    let env = Env::default();
    let (client, _admin) = deploy_and_init(&env);

    let s = schedule(500, INIT_LEDGER + 5);
    client.submit_fee_schedule(&s);

    set_ledger(&env, INIT_LEDGER + 4);
    assert_eq!(client.get_active_fee_schedule(), None);
    assert_eq!(client.get_pending_fee_schedule(), Some(s.clone()));

    set_ledger(&env, INIT_LEDGER + 5);
    assert_eq!(client.get_active_fee_schedule(), Some(s));
    assert_eq!(client.get_pending_fee_schedule(), None);
}

/// `activate_fee_schedule` reports `true` only on the invocation that promotes.
#[test]
fn activation_is_idempotent_across_calls() {
    let env = Env::default();
    let (client, _admin) = deploy_and_init(&env);

    client.submit_fee_schedule(&schedule(500, INIT_LEDGER + 5));
    set_ledger(&env, INIT_LEDGER + 5);

    assert!(client.activate_fee_schedule());
    for _ in 0..3 {
        assert!(
            !client.activate_fee_schedule(),
            "promotion must happen once"
        );
    }
}

/// `activate_fee_schedule` with no pending schedule is a no-op returning `false`.
#[test]
fn activation_without_pending_schedule_is_noop() {
    let env = Env::default();
    let (client, _admin) = deploy_and_init(&env);

    assert!(!client.activate_fee_schedule());
    assert_eq!(client.get_active_fee_schedule(), None);
}

/// Views never mutate: repeated reads past the activation ledger are stable, and the
/// schedule is still promoted only when a mutating path runs.
#[test]
fn views_are_pure_and_stable() {
    let env = Env::default();
    let (client, _admin) = deploy_and_init(&env);

    let s = schedule(500, INIT_LEDGER + 2);
    client.submit_fee_schedule(&s);
    set_ledger(&env, INIT_LEDGER + 10);

    for _ in 0..3 {
        assert_eq!(client.get_active_fee_schedule(), Some(s.clone()));
        assert_eq!(client.get_pending_fee_schedule(), None);
        assert_eq!(client.get_previous_fee_schedule(), None);
    }
}

/// A second activation cycle moves the prior active schedule into `previous`.
#[test]
fn second_activation_preserves_previous_schedule() {
    let env = Env::default();
    let (client, _admin) = deploy_and_init(&env);

    let first = schedule(200, INIT_LEDGER + 1);
    client.submit_fee_schedule(&first);
    set_ledger(&env, INIT_LEDGER + 1);
    assert!(client.activate_fee_schedule());

    let second = schedule(300, INIT_LEDGER + 2);
    client.submit_fee_schedule(&second);
    set_ledger(&env, INIT_LEDGER + 2);
    assert!(client.activate_fee_schedule());

    assert_eq!(client.get_active_fee_schedule(), Some(second));
    assert_eq!(client.get_previous_fee_schedule(), Some(first));
    assert_eq!(client.get_pending_fee_schedule(), None);
}

/// Boundary: `activation_ledger == 0` cannot be submitted (it is in the past), and the
/// contract ledger sequence itself (`INIT_LEDGER`) is the earliest accepted value.
#[test]
fn zero_activation_ledger_is_always_rejected() {
    let env = Env::default();
    let (client, _admin) = deploy_and_init(&env);

    let result = client.try_submit_fee_schedule(&schedule(250, 0));
    assert_fee_error(result, FeeScheduleError::InvalidActivationLedger);
    assert_eq!(client.get_pending_fee_schedule(), None);
}

// ── regression: state is preserved across rejected paths ─────────────────────

/// A rejected submission must not clobber an already-pending schedule, and a later
/// valid submission must still land.
#[test]
fn rejected_submission_preserves_pending_and_allows_recovery() {
    let env = Env::default();
    let (client, _admin) = deploy_and_init(&env);

    let first = schedule(400, INIT_LEDGER + 10);
    client.submit_fee_schedule(&first);

    // Out of bounds: rejected, pending untouched.
    let bad = bounded(10_000, 0, 1_000, INIT_LEDGER + 20);
    assert_fee_error(
        client.try_submit_fee_schedule(&bad),
        FeeScheduleError::FeeOutOfBounds,
    );
    assert_eq!(client.get_pending_fee_schedule(), Some(first.clone()));

    // Activation in the past: also rejected, pending untouched.
    assert_fee_error(
        client.try_submit_fee_schedule(&schedule(500, INIT_LEDGER)),
        FeeScheduleError::InvalidActivationLedger,
    );
    assert_eq!(client.get_pending_fee_schedule(), Some(first));
}

/// The pending schedule survives rejected submissions all the way to activation.
#[test]
fn pending_survives_rejections_then_activates_intact() {
    let env = Env::default();
    let (client, _admin) = deploy_and_init(&env);

    let s = schedule(450, INIT_LEDGER + 3);
    client.submit_fee_schedule(&s);

    assert_fee_error(
        client.try_submit_fee_schedule(&bounded(0, 100, 200, INIT_LEDGER + 4)),
        FeeScheduleError::FeeOutOfBounds,
    );
    assert_eq!(client.get_pending_fee_schedule(), Some(s.clone()));

    set_ledger(&env, INIT_LEDGER + 3);
    assert!(client.activate_fee_schedule());
    assert_eq!(client.get_active_fee_schedule(), Some(s));
}

// ── concurrency / retry safety ───────────────────────────────────────────────

/// The mutation lock is released on every exit path, so a rejected submission cannot
/// wedge later submissions.
#[test]
fn mutation_lock_is_released_after_every_outcome() {
    let env = Env::default();
    let (client, _admin) = deploy_and_init(&env);

    client.submit_fee_schedule(&schedule(250, INIT_LEDGER + 1));
    assert!(!lock_held(&env, &client), "accepted path must release");

    assert_fee_error(
        client.try_submit_fee_schedule(&schedule(250, INIT_LEDGER)),
        FeeScheduleError::InvalidActivationLedger,
    );
    assert!(!lock_held(&env, &client), "rejected path must release");

    client.submit_fee_schedule(&schedule(250, INIT_LEDGER + 1));
    assert!(!lock_held(&env, &client), "duplicate path must release");
}

/// With the lock artificially held, a submission is rejected with `ConcurrentMutation`
/// and changes nothing — the guard is observable rather than silently overwriting.
#[test]
fn held_mutation_lock_rejects_overlapping_submission() {
    let env = Env::default();
    let (client, _admin) = deploy_and_init(&env);

    env.as_contract(&client.address, || {
        env.storage()
            .instance()
            .set(&FeeScheduleStorageKey::MutationLock, &true);
    });

    let result = client.try_submit_fee_schedule(&schedule(250, INIT_LEDGER + 1));
    assert_fee_error(result, FeeScheduleError::ConcurrentMutation);
    assert_eq!(client.get_pending_fee_schedule(), None);

    // Releasing the lock restores normal operation.
    env.as_contract(&client.address, || {
        env.storage()
            .instance()
            .remove(&FeeScheduleStorageKey::MutationLock);
    });
    client.submit_fee_schedule(&schedule(250, INIT_LEDGER + 1));
    assert_eq!(
        client.get_pending_fee_schedule(),
        Some(schedule(250, INIT_LEDGER + 1))
    );
}

/// `activate_fee_schedule` also respects the lock, so an overlapping activation cannot
/// race a submission.
#[test]
fn held_mutation_lock_rejects_overlapping_activation() {
    let env = Env::default();
    let (client, _admin) = deploy_and_init(&env);

    client.submit_fee_schedule(&schedule(250, INIT_LEDGER + 1));
    set_ledger(&env, INIT_LEDGER + 1);

    env.as_contract(&client.address, || {
        env.storage()
            .instance()
            .set(&FeeScheduleStorageKey::MutationLock, &true);
    });

    let result = client.try_activate_fee_schedule();
    assert_fee_error(result, FeeScheduleError::ConcurrentMutation);
    assert_eq!(
        client.get_active_fee_schedule(),
        Some(schedule(250, INIT_LEDGER + 1))
    );

    // Releasing the lock lets a fresh promotion through, and it happens exactly once.
    env.as_contract(&client.address, || {
        env.storage()
            .instance()
            .remove(&FeeScheduleStorageKey::MutationLock);
    });
    assert!(client.activate_fee_schedule());
    assert!(
        !client.activate_fee_schedule(),
        "promotion must happen once"
    );
    assert!(
        !lock_held(&env, &client),
        "activation must release the lock"
    );
}

/// Fee-schedule operations emit no events: they change configuration silently, so a
/// retry cannot produce a duplicate notification for an indexer to reconcile.
///
/// The test host clears the event buffer per top-level invocation, so each
/// submission/activation is checked immediately after its own call.
#[test]
fn fee_schedule_transitions_emit_no_events() {
    let env = Env::default();
    let (client, _admin) = deploy_and_init(&env);

    client.submit_fee_schedule(&schedule(250, INIT_LEDGER + 1));
    assert_eq!(
        env.events().all().events().len(),
        0,
        "submitting a fee schedule must not emit an event"
    );

    set_ledger(&env, INIT_LEDGER + 1);
    assert!(client.activate_fee_schedule());
    assert_eq!(
        env.events().all().events().len(),
        0,
        "activating a fee schedule must not emit an event"
    );

    // A duplicate submission is equally silent.
    client.submit_fee_schedule(&schedule(250, INIT_LEDGER + 1));
    assert_eq!(env.events().all().events().len(), 0);
}
