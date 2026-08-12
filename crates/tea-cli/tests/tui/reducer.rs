use std::pin::Pin;
use std::sync::{Arc, Mutex};

use tea_cli::tui::{
    Action, ActionLoop, ComposerAttachment, DispatchError, Effect, EffectExecutor,
    MAX_COMPOSER_IMAGE_BASE64_BYTES, MAX_SKILL_CATALOG_ROWS, Presentation, TuiState, reduce,
};
use tea_protocol::{
    AgentEvent, EventDelta, HostedToolError, HostedToolOutcome, RunStatus, TokenCount, Usage,
};

use crate::common;

#[tokio::test(flavor = "current_thread")]
async fn snapshot_projects_canonical_transcript_tool_and_redacted_approval() {
    let snapshot = common::pending_snapshot().await;
    let state = TuiState::from_snapshot(&snapshot, common::startup());

    assert_eq!(state.messages().len(), 2);
    assert_eq!(state.tools().len(), 1);
    let approval = state.approval().expect("pending approval projection");
    assert_eq!(approval.tool_name, "write_text_file");
    assert_eq!(approval.effects, ["fs.write"]);
    assert_eq!(approval.target, "native");
    assert!(approval.arguments.contains("notes.txt"));
    assert!(!state.is_resyncing());
}

#[tokio::test(flavor = "current_thread")]
async fn local_run_completion_preserves_the_measured_elapsed_time() {
    let snapshot = common::archive_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());

    let _ = reduce(&mut state, Action::StartRunActivity);
    let _ = reduce(&mut state, Action::AdvanceRunElapsed(1));
    let _ = reduce(&mut state, Action::AdvanceRunElapsed(1));
    let _ = reduce(&mut state, Action::FinishRunActivity);

    assert!(!state.is_running());
    assert_eq!(state.run_elapsed_seconds(), 2);
}

#[tokio::test(flavor = "current_thread")]
async fn new_run_clears_transient_notifications_from_the_previous_turn() {
    let snapshot = common::archive_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());

    let _ = reduce(
        &mut state,
        Action::Notify("model retry policy was exhausted".to_owned()),
    );
    assert_eq!(Presentation::from_state(&state).notifications().len(), 1);

    let _ = reduce(&mut state, Action::StartRunActivity);
    assert!(Presentation::from_state(&state).notifications().is_empty());

    let _ = reduce(
        &mut state,
        Action::Notify("failure observed before an external run".to_owned()),
    );
    let next = state.event_cursor().get() + 1;
    let _ = reduce(
        &mut state,
        Action::Event(Box::new(common::event(
            next,
            499,
            AgentEvent::RunStarted {},
        ))),
    );
    assert!(Presentation::from_state(&state).notifications().is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn skill_catalog_is_bounded_and_sanitized_as_ephemeral_state() {
    let snapshot = common::archive_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let rows = (0..=MAX_SKILL_CATALOG_ROWS)
        .map(|index| format!("skill-{index}\n"))
        .collect::<Vec<_>>();

    let effects = reduce(&mut state, Action::SetSkillCatalog(rows));

    assert_eq!(effects, [Effect::Render]);
    assert_eq!(state.skill_catalog().len(), MAX_SKILL_CATALOG_ROWS);
    assert_eq!(state.skill_catalog()[0], "skill-0");
    assert!(!state.skill_catalog().iter().any(|row| row.contains('\n')));
}

#[tokio::test(flavor = "current_thread")]
async fn model_retry_clears_failed_ephemera_and_tracks_countdown() {
    let snapshot = common::archive_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let first = state.event_cursor().get() + 1;
    let message_id = common::message_id();
    let hosted_id = "0195a0b1-7e00-7000-8000-000000000093".parse().unwrap();

    for (offset, event) in [
        AgentEvent::RunStarted {},
        AgentEvent::MessageDelta {
            message_id,
            content_index: 0,
            delta: EventDelta::TextDelta {
                text: "discard me".to_owned(),
            },
        },
        AgentEvent::HostedToolStarted {
            tool_call_id: hosted_id,
            tool_name: "web_search".to_owned(),
        },
        AgentEvent::ModelRetryScheduled {
            message_id,
            attempt: 1,
            max_retries: 3,
            delay_ms: 2_000,
        },
    ]
    .into_iter()
    .enumerate()
    {
        let _ = reduce(
            &mut state,
            Action::Event(Box::new(common::event(
                first + u64::try_from(offset).unwrap(),
                520 + u16::try_from(offset).unwrap(),
                event,
            ))),
        );
    }

    assert!(state.streaming().is_empty());
    assert!(state.hosted_tools().is_empty());
    let retry = state.model_retry().unwrap();
    assert_eq!(retry.attempt, 1);
    assert_eq!(retry.max_retries, 3);
    assert_eq!(retry.remaining_seconds(), 2);

    let _ = reduce(&mut state, Action::AdvanceRunElapsed(1));
    assert_eq!(state.model_retry().unwrap().remaining_seconds(), 1);
    let _ = reduce(&mut state, Action::AdvanceRunElapsed(2));
    assert_eq!(state.model_retry().unwrap().remaining_seconds(), 0);

    let _ = reduce(
        &mut state,
        Action::Event(Box::new(common::event(
            first + 4,
            524,
            AgentEvent::ModelRetryStarted {
                message_id,
                attempt: 1,
                max_retries: 3,
            },
        ))),
    );
    assert!(state.model_retry().is_none());
}

#[tokio::test(flavor = "current_thread")]
async fn hosted_tool_observations_track_success_failure_and_clear_on_snapshot_rebuild() {
    let snapshot = common::archive_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let successful_id = "0195a0b1-7e00-7000-8000-000000000091".parse().unwrap();
    let failed_id = "0195a0b1-7e00-7000-8000-000000000092".parse().unwrap();
    let first = state.event_cursor().get() + 1;

    assert_eq!(
        reduce(
            &mut state,
            Action::Event(Box::new(common::event(
                first,
                501,
                AgentEvent::HostedToolStarted {
                    tool_call_id: successful_id,
                    tool_name: "web_search".to_owned(),
                },
            ))),
        ),
        [Effect::Render]
    );
    let running = &state.hosted_tools()[&successful_id];
    assert_eq!(running.tool_name, "web_search");
    assert!(running.arguments.is_none());
    assert!(running.outcome.is_none());

    let success = AgentEvent::HostedToolCompleted {
        tool_call_id: successful_id,
        tool_name: "web_search".to_owned(),
        arguments: serde_json::json!({"query": "tea-rs hosted search"}),
        outcome: HostedToolOutcome::Success,
        source_count: 2,
    };
    assert_eq!(
        reduce(
            &mut state,
            Action::Event(Box::new(common::event(first + 1, 502, success))),
        ),
        [Effect::Render]
    );
    let completed = &state.hosted_tools()[&successful_id];
    assert_eq!(
        completed.arguments,
        Some(serde_json::json!({"query": "tea-rs hosted search"}))
    );
    assert_eq!(completed.outcome, Some(HostedToolOutcome::Success));
    assert_eq!(completed.source_count, Some(2));

    for (offset, event) in [
        AgentEvent::HostedToolStarted {
            tool_call_id: failed_id,
            tool_name: "web_search".to_owned(),
        },
        AgentEvent::HostedToolCompleted {
            tool_call_id: failed_id,
            tool_name: "web_search".to_owned(),
            arguments: serde_json::json!({"query": "provider outage"}),
            outcome: HostedToolOutcome::Error(
                HostedToolError::new("provider_error", "search unavailable").unwrap(),
            ),
            source_count: 0,
        },
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(
            reduce(
                &mut state,
                Action::Event(Box::new(common::event(
                    first + 2 + u64::try_from(offset).unwrap(),
                    503 + u16::try_from(offset).unwrap(),
                    event,
                ))),
            ),
            [Effect::Render]
        );
    }
    assert!(matches!(
        state.hosted_tools()[&failed_id].outcome,
        Some(HostedToolOutcome::Error(_))
    ));

    let _ = reduce(&mut state, Action::SnapshotLoaded(Box::new(snapshot)));
    assert!(state.hosted_tools().is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn attachments_are_bounded_reducible_and_survive_snapshot_rebuilds() {
    let snapshot = common::archive_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let attachment =
        ComposerAttachment::inline("image/png", "iVBORw0KGgo=", "fixture.png").unwrap();

    assert_eq!(
        reduce(&mut state, Action::AddAttachment(attachment.clone())),
        [Effect::Render]
    );
    assert_eq!(state.attachments(), std::slice::from_ref(&attachment));
    let _ = reduce(
        &mut state,
        Action::SnapshotLoaded(Box::new(snapshot.clone())),
    );
    assert_eq!(state.attachments(), std::slice::from_ref(&attachment));
    let _ = reduce(
        &mut state,
        Action::SnapshotLoadFailed("unrelated failure".to_owned()),
    );
    assert_eq!(state.attachments(), std::slice::from_ref(&attachment));

    for _ in 1..4 {
        let _ = reduce(&mut state, Action::AddAttachment(attachment.clone()));
    }
    assert_eq!(state.attachments().len(), 4);
    let _ = reduce(&mut state, Action::AddAttachment(attachment.clone()));
    assert_eq!(state.attachments().len(), 4);

    let _ = reduce(&mut state, Action::RemoveAttachment { index: 0 });
    let _ = reduce(&mut state, Action::RemoveAttachment { index: 5 });
    assert_eq!(state.attachments().len(), 4);
    let _ = reduce(&mut state, Action::RemoveAttachment { index: 1 });
    assert_eq!(state.attachments().len(), 3);
    let _ = reduce(&mut state, Action::ClearAttachments);
    assert!(state.attachments().is_empty());

    let maximum = ComposerAttachment::inline(
        "image/png",
        "AAAA".repeat(MAX_COMPOSER_IMAGE_BASE64_BYTES / 4),
        "maximum.png",
    )
    .unwrap();
    let _ = reduce(&mut state, Action::AddAttachment(maximum));
    let _ = reduce(&mut state, Action::AddAttachment(attachment));
    assert_eq!(state.attachments().len(), 1);
    assert_eq!(
        state.attachment_encoded_bytes(),
        MAX_COMPOSER_IMAGE_BASE64_BYTES
    );
}

#[tokio::test(flavor = "current_thread")]
async fn forward_sequence_jump_is_not_treated_as_event_loss() {
    let snapshot = common::pending_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let start_sequence = state.event_cursor().get() + 1;

    assert_eq!(
        reduce(
            &mut state,
            Action::Event(Box::new(common::event(
                start_sequence,
                1,
                AgentEvent::RunStarted {},
            ))),
        ),
        [Effect::Render]
    );

    let effects = reduce(
        &mut state,
        Action::Event(Box::new(common::event(
            start_sequence + 2,
            2,
            AgentEvent::MessageDelta {
                message_id: common::message_id(),
                content_index: 0,
                delta: EventDelta::TextDelta {
                    text: "resumed".to_owned(),
                },
            },
        ))),
    );

    assert_eq!(effects, [Effect::Render]);
    assert!(!state.is_resyncing());
    assert_eq!(
        state.streaming()[&common::message_id()].blocks[&0].text,
        "resumed"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn reducer_handles_partial_stream_duplicates_late_observations_and_checkpoint_rebuild() {
    let snapshot = common::pending_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let start = common::event(6, 1, AgentEvent::RunStarted {});
    let first = common::event(
        7,
        2,
        AgentEvent::MessageDelta {
            message_id: common::message_id(),
            content_index: 0,
            delta: EventDelta::TextDelta {
                text: "hel".to_owned(),
            },
        },
    );
    let second = common::event(
        8,
        3,
        AgentEvent::MessageDelta {
            message_id: common::message_id(),
            content_index: 0,
            delta: EventDelta::TextDelta {
                text: "lo".to_owned(),
            },
        },
    );

    assert_eq!(
        reduce(&mut state, Action::Event(Box::new(start))),
        [Effect::Render]
    );
    assert_eq!(
        reduce(&mut state, Action::Event(Box::new(first.clone()))),
        [Effect::Render]
    );
    assert_eq!(
        reduce(&mut state, Action::Event(Box::new(second))),
        [Effect::Render]
    );
    assert_eq!(
        state.streaming()[&common::message_id()].blocks[&0].text,
        "hello"
    );
    assert!(reduce(&mut state, Action::Event(Box::new(first))).is_empty());

    let late = common::event(
        7,
        4,
        AgentEvent::ToolExecutionProgress {
            tool_call_id: common::tool_call_id(),
            message: "late".to_owned(),
            completed_units: 1,
            total_units: Some(1),
        },
    );
    assert!(reduce(&mut state, Action::Event(Box::new(late))).is_empty());

    let checkpoint = common::event(10, 5, AgentEvent::TurnCheckpointed {});
    let effects = reduce(&mut state, Action::Event(Box::new(checkpoint)));
    assert!(state.is_resyncing());
    assert!(matches!(effects[0], Effect::ReloadSnapshot { .. }));

    let _ = reduce(&mut state, Action::QueueFollowUp("keep me".to_owned()));
    let _ = reduce(&mut state, Action::SetEditor("draft".to_owned()));
    let _ = reduce(
        &mut state,
        Action::SnapshotLoaded(Box::new(snapshot.clone())),
    );
    assert!(!state.is_resyncing());
    assert!(state.streaming().is_empty());
    assert_eq!(state.editor(), "draft");
}

#[tokio::test(flavor = "current_thread")]
async fn read_tool_snapshot_reloads_preserve_the_live_event_cursor() {
    let snapshot = common::pending_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let first = state.event_cursor().get() + 1;

    let events = [
        AgentEvent::RunStarted {},
        AgentEvent::ToolCallRequested {
            tool_call_id: common::tool_call_id(),
            tool_name: "read".to_owned(),
            arguments: serde_json::json!({"path": "README.md"}),
        },
        AgentEvent::TurnCheckpointed {},
    ];
    for (offset, event) in events.into_iter().enumerate() {
        let sequence = first + u64::try_from(offset).unwrap();
        let effects = reduce(
            &mut state,
            Action::Event(Box::new(common::event(
                sequence,
                u16::try_from(100 + offset).unwrap(),
                event,
            ))),
        );
        assert!(!effects.is_empty());
        assert_eq!(state.event_cursor().get(), sequence);
    }

    let tool_checkpoint = state.event_cursor();
    assert!(state.is_resyncing());
    let _ = reduce(
        &mut state,
        Action::SnapshotLoaded(Box::new(snapshot.clone())),
    );
    assert!(!state.is_resyncing());
    assert_eq!(state.event_cursor(), tool_checkpoint);

    let message_sequence = tool_checkpoint.get() + 1;
    assert_eq!(
        reduce(
            &mut state,
            Action::Event(Box::new(common::event(
                message_sequence,
                103,
                AgentEvent::MessageDelta {
                    message_id: common::message_id(),
                    content_index: 0,
                    delta: EventDelta::TextDelta {
                        text: "README summary".to_owned(),
                    },
                },
            ))),
        ),
        [Effect::Render]
    );

    let final_checkpoint_sequence = message_sequence + 1;
    let effects = reduce(
        &mut state,
        Action::Event(Box::new(common::event(
            final_checkpoint_sequence,
            104,
            AgentEvent::TurnCheckpointed {},
        ))),
    );
    assert!(matches!(effects[0], Effect::ReloadSnapshot { .. }));
    let _ = reduce(
        &mut state,
        Action::SnapshotLoaded(Box::new(snapshot.clone())),
    );
    assert_eq!(state.event_cursor().get(), final_checkpoint_sequence);

    let usage = Usage::new(TokenCount::new(11).unwrap(), TokenCount::new(7).unwrap());
    for _ in 0..125 {
        let _ = reduce(&mut state, Action::AdvanceRunElapsed(1));
    }
    let finished_sequence = final_checkpoint_sequence + 1;
    let effects = reduce(
        &mut state,
        Action::Event(Box::new(common::event(
            finished_sequence,
            105,
            AgentEvent::RunFinished {
                status: RunStatus::Completed,
                usage: Some(usage.clone()),
                cost: None,
            },
        ))),
    );
    assert!(matches!(effects[0], Effect::ReloadSnapshot { .. }));
    assert_eq!(state.event_cursor().get(), finished_sequence);
    assert_eq!(state.usage(), Some(&usage));
    assert_eq!(state.run_elapsed_seconds(), 125);
}

#[tokio::test(flavor = "current_thread")]
async fn checkpoint_compaction_branch_approval_cancellation_and_reconnect_request_snapshots() {
    let snapshot = common::archive_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let next = state.event_cursor().get() + 1;
    let cases = [
        AgentEvent::TurnCheckpointed {},
        AgentEvent::SessionCompacted {
            summary_message_id: common::message_id(),
            compacted_through_message_id: common::message_id(),
        },
        AgentEvent::SessionForked {
            source_branch_id: "0195a0b1-5e4e-728c-bfe1-0aa7aa000020".parse().unwrap(),
            branch_id: "0195a0b1-5e4f-7bd5-9760-0aa7aa000021".parse().unwrap(),
            from_message_id: common::message_id(),
        },
        AgentEvent::ApprovalRequested {
            approval_id: "0195a0b1-5e46-7e2a-b230-0aa7aa000012".parse().unwrap(),
            tool_call_id: common::tool_call_id(),
            capabilities: vec!["fs.write".to_owned()],
            resources: vec!["file:notes.txt".to_owned()],
            expires_at: "2026-07-24T10:05:00.000Z".parse().unwrap(),
        },
    ];
    for (index, event) in cases.into_iter().enumerate() {
        state = TuiState::from_snapshot(&snapshot, common::startup());
        let event = common::event(next, u16::try_from(index + 20).unwrap(), event);
        let effects = reduce(&mut state, Action::Event(Box::new(event)));
        assert!(state.is_resyncing());
        assert!(
            effects
                .iter()
                .any(|effect| matches!(effect, Effect::ReloadSnapshot { .. }))
        );
    }

    state = TuiState::from_snapshot(&snapshot, common::startup());
    let start = common::event(next, 30, AgentEvent::RunStarted {});
    let _ = reduce(&mut state, Action::Event(Box::new(start)));
    let cancelled = common::event(
        next + 1,
        31,
        AgentEvent::RunFinished {
            status: RunStatus::Cancelled,
            usage: None,
            cost: None,
        },
    );
    let effects = reduce(&mut state, Action::Event(Box::new(cancelled)));
    assert!(
        effects
            .iter()
            .any(|effect| matches!(effect, Effect::ReloadSnapshot { .. }))
    );
    assert!(state.is_resyncing());

    let _ = reduce(
        &mut state,
        Action::SnapshotLoaded(Box::new(snapshot.clone())),
    );
    let effects = reduce(&mut state, Action::Reconnected);
    assert!(matches!(effects[0], Effect::ReloadSnapshot { .. }));
}

#[derive(Default)]
struct RecordingExecutor {
    effects: Arc<Mutex<Vec<Effect>>>,
}

impl EffectExecutor for RecordingExecutor {
    fn execute(
        &self,
        effect: Effect,
        _state: &TuiState,
    ) -> Pin<Box<dyn Future<Output = Option<Action>> + Send + 'static>> {
        self.effects.lock().unwrap().push(effect);
        Box::pin(async { None })
    }
}

#[tokio::test(flavor = "current_thread")]
async fn action_effect_loop_is_bounded_owned_and_coalesces_rendering() {
    let snapshot = common::pending_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let (sender, application) = ActionLoop::new(1, 1);
    sender.try_send(Action::Notify("one".to_owned())).unwrap();
    assert_eq!(
        sender.try_send(Action::Notify("overflow".to_owned())),
        Err(DispatchError::Full)
    );
    drop(sender);

    let effects = Arc::new(Mutex::new(Vec::new()));
    let executor = RecordingExecutor {
        effects: Arc::clone(&effects),
    };
    application.run(&mut state, &executor).await;
    assert_eq!(*effects.lock().unwrap(), [Effect::Render]);
}
