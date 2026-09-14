#![allow(dead_code)]

use std::future::pending;
use std::str::FromStr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU16, AtomicUsize, Ordering};

use tea_kernel::{
    KernelClock, KernelDeadlineFuture, KernelError, KernelEventFuture, KernelEventSink,
    KernelIdSource,
};
use tea_model::{
    BoxModelStream, ModelCancellation, ModelCapabilities, ModelDisplayName, ModelFailure,
    ModelFailureCode, ModelProvider, ModelRequest, ModelSpec, ProviderId,
};
use tea_policy::GrantId;
use tea_protocol::{
    ApprovalId, CanonicalMessage, ContentBlock, EventEnvelope, EventId, MessageId, ModelId,
    ProfileId, ProtocolMetadata, ProtocolTimestamp, RecordEnvelope, RecordId, RetryClass, RunId,
    SessionId, SessionRecord, SessionSequence, TokenCount, ToolCallId, TurnId,
};
use tea_session::{AppendTransaction, InMemorySessionStore, SessionStore};
use tea_testkit::{ScriptedModelProvider, ScriptedModelResponse};

pub const SESSION: &str = "0195a0b1-5e3a-7d72-a902-c4e85d828bf1";
const CREATED: &str = "0195a0b1-5e50-7af4-8972-0aa7aa000022";
const CONFIGURED: &str = "0195a0b1-5e51-79e1-8f4a-0aa7aa000023";
const MESSAGE_RECORD: &str = "0195a0b1-5e52-7b3e-93f1-0aa7aa000024";
const MESSAGE: &str = "0195a0b1-5e53-74b2-8c25-0aa7aa000025";
pub const NOW: &str = "2026-07-23T09:30:12.125Z";

#[derive(Debug)]
pub struct FixedClock;
impl KernelClock for FixedClock {
    fn now(&self) -> Result<ProtocolTimestamp, KernelError> {
        Ok(timestamp())
    }
    fn sleep_until(&self, _deadline: ProtocolTimestamp) -> KernelDeadlineFuture<'_> {
        Box::pin(pending())
    }
}

#[derive(Debug, Default)]
pub struct TestIds(AtomicU16);
impl TestIds {
    pub const fn with_start(value: u16) -> Self {
        Self(AtomicU16::new(value))
    }

    fn next<T>(&self) -> Result<T, KernelError>
    where
        T: FromStr,
        T::Err: std::fmt::Display,
    {
        let value = self.0.fetch_add(1, Ordering::SeqCst);
        format!("0195a0b1-{value:04x}-7000-8000-000000000001")
            .parse()
            .map_err(|error: T::Err| {
                KernelError::new(tea_kernel::KernelErrorCode::IdExhausted, error.to_string())
            })
    }
}
impl KernelIdSource for TestIds {
    fn next_run_id(&self) -> Result<RunId, KernelError> {
        self.next()
    }
    fn next_turn_id(&self) -> Result<TurnId, KernelError> {
        self.next()
    }
    fn next_message_id(&self) -> Result<MessageId, KernelError> {
        self.next()
    }
    fn next_tool_call_id(&self) -> Result<ToolCallId, KernelError> {
        self.next()
    }
    fn next_approval_id(&self) -> Result<ApprovalId, KernelError> {
        self.next()
    }
    fn next_grant_id(&self) -> Result<GrantId, KernelError> {
        self.next()
    }
    fn next_event_id(&self) -> Result<EventId, KernelError> {
        self.next()
    }
    fn next_record_id(&self) -> Result<RecordId, KernelError> {
        self.next()
    }
}

#[derive(Debug, Default)]
pub struct EventCollector(Mutex<Vec<EventEnvelope>>);
impl EventCollector {
    pub fn events(&self) -> Vec<EventEnvelope> {
        self.0.lock().unwrap().clone()
    }
}
impl KernelEventSink for EventCollector {
    fn last_sequence(&self, session_id: SessionId) -> Option<SessionSequence> {
        self.0
            .lock()
            .ok()?
            .iter()
            .rev()
            .find(|event| event.session_id() == session_id)
            .map(EventEnvelope::sequence)
    }

    fn emit(&self, event: EventEnvelope) -> KernelEventFuture<'_> {
        Box::pin(async move {
            self.0.lock().unwrap().push(event);
            Ok(())
        })
    }
}

pub fn timestamp() -> ProtocolTimestamp {
    ProtocolTimestamp::from_str(NOW).unwrap()
}
pub fn session_id() -> SessionId {
    SessionId::from_str(SESSION).unwrap()
}
fn envelope(sequence: u64, record_id: &str, record: SessionRecord) -> RecordEnvelope {
    RecordEnvelope::new(
        RecordId::from_str(record_id).unwrap(),
        session_id(),
        SessionSequence::new(sequence),
        timestamp(),
        None,
        None,
        None,
        ProtocolMetadata::default(),
        record,
    )
    .unwrap()
}

pub async fn store() -> InMemorySessionStore {
    let store = InMemorySessionStore::new();
    store
        .append(AppendTransaction::new(
            session_id(),
            None,
            vec![
                envelope(
                    0,
                    CREATED,
                    SessionRecord::SessionCreated {
                        profile_id: ProfileId::from_str("coding").unwrap(),
                        metadata: ProtocolMetadata::default(),
                    },
                ),
                envelope(
                    1,
                    CONFIGURED,
                    SessionRecord::ConfigurationChanged {
                        model: Some(tea_protocol::ModelRef::new(
                            "fake".parse().unwrap(),
                            ModelId::from_str("fake/model").unwrap(),
                        )),
                        profile_id: None,
                        reasoning_effort: None,
                    },
                ),
                envelope(
                    2,
                    MESSAGE_RECORD,
                    SessionRecord::MessageCommitted {
                        message: CanonicalMessage::user(
                            MessageId::from_str(MESSAGE).unwrap(),
                            vec![ContentBlock::text("answer briefly").unwrap()],
                            timestamp(),
                        )
                        .unwrap(),
                    },
                ),
            ],
        ))
        .await
        .unwrap();
    store
}

pub fn provider(scripts: impl IntoIterator<Item = ScriptedModelResponse>) -> ScriptedModelProvider {
    provider_with_capabilities(scripts, ModelCapabilities::text().with_tools(false))
}

pub fn provider_with_capabilities(
    scripts: impl IntoIterator<Item = ScriptedModelResponse>,
    capabilities: ModelCapabilities,
) -> ScriptedModelProvider {
    let provider_id = ProviderId::from_str("fake").unwrap();
    let model = ModelSpec::new(
        ModelId::from_str("fake/model").unwrap(),
        provider_id.clone(),
        ModelDisplayName::from_str("Fake Model").unwrap(),
        TokenCount::new(32_000).unwrap(),
        TokenCount::new(4_000).unwrap(),
        capabilities,
    )
    .unwrap();
    ScriptedModelProvider::new(provider_id, vec![model], scripts)
}

#[derive(Debug)]
pub struct RejectingPreflightProvider {
    inner: ScriptedModelProvider,
    fail_on_call: usize,
    validate_calls: AtomicUsize,
}

impl RejectingPreflightProvider {
    pub fn new(
        scripts: impl IntoIterator<Item = ScriptedModelResponse>,
        fail_on_call: usize,
    ) -> Self {
        Self {
            inner: provider(scripts),
            fail_on_call,
            validate_calls: AtomicUsize::new(0),
        }
    }

    pub fn captured_requests(&self) -> Vec<ModelRequest> {
        self.inner.captured_requests().unwrap()
    }

    pub fn remaining_scripts(&self) -> usize {
        self.inner.remaining_scripts().unwrap()
    }

    pub fn validate_calls(&self) -> usize {
        self.validate_calls.load(Ordering::SeqCst)
    }
}

impl ModelProvider for RejectingPreflightProvider {
    fn provider_id(&self) -> &ProviderId {
        self.inner.provider_id()
    }

    fn models(&self) -> &[ModelSpec] {
        self.inner.models()
    }

    fn validate_request(&self, request: &ModelRequest) -> Result<(), ModelFailure> {
        <ScriptedModelProvider as ModelProvider>::validate_request(&self.inner, request)?;
        let call = self.validate_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if call == self.fail_on_call {
            return Err(ModelFailure::new(
                ModelFailureCode::InvalidRequest,
                "provider-specific preflight rejected request",
                RetryClass::Never,
            )
            .unwrap());
        }
        Ok(())
    }

    fn stream(&self, request: ModelRequest, cancellation: ModelCancellation) -> BoxModelStream {
        self.inner.stream(request, cancellation)
    }
}
