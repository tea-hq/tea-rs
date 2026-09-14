use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Deserializer, Serialize};
use tea_policy::{ApprovalRequest, ApprovalResolution, GrantId, PolicyGrant};
use tea_protocol::{
    ApprovalId, FinalOutputFormat, ProfileId, RecordEnvelope, RecordId, SessionId, SessionRecord,
};

use crate::{SessionReducer, SessionStoreErrorCode};

/// Rich policy approval value linked to a canonical durable approval transition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
#[allow(clippy::large_enum_variant)] // Boxing would break the public artifact value API.
pub enum ApprovalArtifactEntry {
    /// Self-contained redacted request snapshot.
    Requested {
        /// Canonical record containing `approval_requested`.
        record_id: RecordId,
        /// Validated policy request.
        request: ApprovalRequest,
        /// Run-scoped final-output contract to restore after approval.
        #[serde(deserialize_with = "deserialize_final_output_format")]
        final_output_format: Option<FinalOutputFormat>,
    },
    /// Self-contained terminal resolution snapshot.
    Resolved {
        /// Canonical record containing `approval_resolved`.
        record_id: RecordId,
        /// Validated policy resolution.
        resolution: ApprovalResolution,
    },
}

impl ApprovalArtifactEntry {
    /// Returns the linked canonical record.
    #[must_use]
    pub const fn record_id(&self) -> RecordId {
        match self {
            Self::Requested { record_id, .. } | Self::Resolved { record_id, .. } => *record_id,
        }
    }
}

fn deserialize_final_output_format<'de, D>(
    deserializer: D,
) -> Result<Option<FinalOutputFormat>, D::Error>
where
    D: Deserializer<'de>,
{
    Option::deserialize(deserializer)
}

/// Append-only authorization-grant journal fact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GrantJournalEntry {
    /// A matching approval resolution issued the grant.
    Issued {
        /// Canonical approval-resolution record authorizing issuance.
        approval_record_id: RecordId,
        /// Immutable issued grant.
        grant: PolicyGrant,
    },
    /// A previously issued grant became immutably revoked.
    Revoked {
        /// Immutable revoked form retaining the original grant identity.
        grant: PolicyGrant,
    },
}

impl GrantJournalEntry {
    /// Returns the stable grant identity.
    #[must_use]
    pub const fn grant_id(&self) -> GrantId {
        match self {
            Self::Issued { grant, .. } | Self::Revoked { grant } => grant.id(),
        }
    }
}

#[derive(Debug, Clone)]
struct ApprovalContext {
    profile_id: ProfileId,
    tool_name: String,
}

/// Derived authorization state rebuilt from approval and grant journals.
#[derive(Debug, Clone, Default)]
pub struct ArtifactState {
    requests: BTreeMap<ApprovalId, ApprovalRequest>,
    resolutions: BTreeSet<ApprovalId>,
    grants: BTreeMap<GrantId, PolicyGrant>,
}

impl ArtifactState {
    /// Reconstructs grant state from a persisted grant journal.
    ///
    /// Use [`rebuild_from_journals`](Self::rebuild_from_journals) when approval
    /// artifacts are available and future approval resolutions may be appended.
    #[must_use]
    pub fn rebuild_from_journal(grant_journal: &[GrantJournalEntry]) -> Self {
        Self::rebuild_from_journals(&[], grant_journal)
    }

    /// Reconstructs authorization state from persisted side journals.
    ///
    /// Unlike [`apply`](Self::apply), this trusts persisted facts without
    /// re-validating them against canonical records; durable stores use it on
    /// load where each journal fact was validated before persistence.
    #[must_use]
    pub fn rebuild_from_journals(
        approvals: &[ApprovalArtifactEntry],
        grant_journal: &[GrantJournalEntry],
    ) -> Self {
        let mut state = Self::default();
        for entry in approvals {
            match entry {
                ApprovalArtifactEntry::Requested { request, .. } => {
                    state
                        .requests
                        .insert(*request.approval_id(), request.clone());
                }
                ApprovalArtifactEntry::Resolved { resolution, .. } => {
                    state
                        .resolutions
                        .insert(*resolution.request().approval_id());
                }
            }
        }
        for entry in grant_journal {
            match entry {
                GrantJournalEntry::Issued { grant, .. } | GrantJournalEntry::Revoked { grant } => {
                    state.grants.insert(grant.id(), grant.clone());
                }
            }
        }
        state
    }

    /// Returns non-revoked grant candidates in stable grant-id order.
    #[must_use]
    pub fn active_grants(&self) -> Vec<PolicyGrant> {
        self.grants
            .values()
            .filter(|grant| grant.revoked_at().is_none())
            .cloned()
            .collect()
    }

    /// Applies approval and grant journal facts linked to a transaction.
    ///
    /// # Errors
    ///
    /// Returns an error when an artifact references a missing canonical record.
    pub fn apply(
        &mut self,
        session_id: SessionId,
        durable_records: &[RecordEnvelope],
        transaction_records: &[RecordEnvelope],
        approvals: &[ApprovalArtifactEntry],
        grants: &[GrantJournalEntry],
    ) -> Result<(), ArtifactValidationError> {
        let record_index = transaction_records
            .iter()
            .map(|record| (record.record_id(), record))
            .collect::<BTreeMap<_, _>>();
        let contexts = approval_contexts(durable_records)?;
        let resolutions = self.apply_approvals(session_id, &record_index, &contexts, approvals)?;
        self.apply_grants(grants, &resolutions)
    }

    fn apply_approvals<'a>(
        &mut self,
        session_id: SessionId,
        record_index: &BTreeMap<RecordId, &RecordEnvelope>,
        contexts: &BTreeMap<ApprovalId, ApprovalContext>,
        approvals: &'a [ApprovalArtifactEntry],
    ) -> Result<BTreeMap<RecordId, &'a ApprovalResolution>, ArtifactValidationError> {
        let mut resolutions = BTreeMap::new();
        for entry in approvals {
            let record = record_index
                .get(&entry.record_id())
                .ok_or(ArtifactValidationError::MissingCanonicalRecord)?;
            match (entry, record.record()) {
                (
                    ApprovalArtifactEntry::Requested { request, .. },
                    SessionRecord::ApprovalRequested {
                        approval_id,
                        tool_call_id,
                        expires_at,
                    },
                ) => self.apply_request(
                    session_id,
                    record,
                    request,
                    *approval_id,
                    *tool_call_id,
                    *expires_at,
                    contexts,
                )?,
                (
                    ApprovalArtifactEntry::Resolved { resolution, .. },
                    SessionRecord::ApprovalResolved {
                        approval_id,
                        decision,
                    },
                ) => {
                    self.apply_resolution(record, resolution, *approval_id, *decision)?;
                    resolutions.insert(record.record_id(), resolution);
                }
                _ => return Err(ArtifactValidationError::CanonicalApprovalMismatch),
            }
        }
        Ok(resolutions)
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_request(
        &mut self,
        session_id: SessionId,
        record: &RecordEnvelope,
        request: &ApprovalRequest,
        approval_id: ApprovalId,
        tool_call_id: tea_protocol::ToolCallId,
        expires_at: tea_protocol::ProtocolTimestamp,
        contexts: &BTreeMap<ApprovalId, ApprovalContext>,
    ) -> Result<(), ArtifactValidationError> {
        let context = contexts
            .get(&approval_id)
            .ok_or(ArtifactValidationError::CanonicalApprovalMismatch)?;
        if request.approval_id() != &approval_id
            || request.tool_call_id() != &tool_call_id
            || *request.session_id() != session_id
            || request.expires_at() != expires_at
            || request.created_at() != record.timestamp()
            || request.profile_id() != &context.profile_id
            || request.tool_name().as_str() != context.tool_name
        {
            return Err(ArtifactValidationError::CanonicalApprovalMismatch);
        }
        if self.requests.insert(approval_id, request.clone()).is_some() {
            return Err(ArtifactValidationError::DuplicateApprovalArtifact);
        }
        Ok(())
    }

    fn apply_resolution(
        &mut self,
        record: &RecordEnvelope,
        resolution: &ApprovalResolution,
        approval_id: ApprovalId,
        decision: tea_protocol::ApprovalDecision,
    ) -> Result<(), ArtifactValidationError> {
        let request = self
            .requests
            .get(&approval_id)
            .ok_or(ArtifactValidationError::MissingApprovalRequest)?;
        if resolution.request().approval_id() != &approval_id
            || resolution.decision() != decision
            || resolution.decided_at() != record.timestamp()
            || request != resolution.request()
            || !self.resolutions.insert(approval_id)
        {
            return Err(ArtifactValidationError::ApprovalResolutionMismatch);
        }
        Ok(())
    }

    fn apply_grants(
        &mut self,
        grants: &[GrantJournalEntry],
        resolutions: &BTreeMap<RecordId, &ApprovalResolution>,
    ) -> Result<(), ArtifactValidationError> {
        let mut issued_resolution_records = BTreeSet::new();
        for entry in grants {
            match entry {
                GrantJournalEntry::Issued {
                    approval_record_id,
                    grant,
                } => {
                    let resolution = resolutions
                        .get(approval_record_id)
                        .ok_or(ArtifactValidationError::GrantWithoutResolution)?;
                    if resolution.issued_grant() != Some(grant) || grant.revoked_at().is_some() {
                        return Err(ArtifactValidationError::GrantMismatch);
                    }
                    if self.grants.insert(grant.id(), grant.clone()).is_some() {
                        return Err(ArtifactValidationError::DuplicateGrant);
                    }
                    issued_resolution_records.insert(*approval_record_id);
                }
                GrantJournalEntry::Revoked { grant } => self.apply_revocation(grant)?,
            }
        }
        if resolutions.iter().any(|(record_id, resolution)| {
            resolution.issued_grant().is_some() && !issued_resolution_records.contains(record_id)
        }) {
            return Err(ArtifactValidationError::MissingGrantJournalEntry);
        }
        Ok(())
    }

    fn apply_revocation(&mut self, grant: &PolicyGrant) -> Result<(), ArtifactValidationError> {
        let issued = self
            .grants
            .get(&grant.id())
            .ok_or(ArtifactValidationError::UnknownGrant)?;
        if issued.revoked_at().is_some()
            || grant.revoked_at().is_none()
            || !same_grant_before_revocation(issued, grant)
        {
            return Err(ArtifactValidationError::InvalidRevocation);
        }
        self.grants.insert(grant.id(), grant.clone());
        Ok(())
    }
}

fn approval_contexts(
    records: &[RecordEnvelope],
) -> Result<BTreeMap<ApprovalId, ApprovalContext>, ArtifactValidationError> {
    let mut reducer = SessionReducer::new();
    let mut contexts = BTreeMap::new();
    for record in records {
        reducer
            .apply(record)
            .map_err(|_| ArtifactValidationError::InvalidCanonicalLog)?;
        if let SessionRecord::ApprovalRequested {
            approval_id,
            tool_call_id,
            ..
        } = record.record()
        {
            let state = reducer
                .state()
                .ok_or(ArtifactValidationError::InvalidCanonicalLog)?;
            let tool = state
                .tool_calls()
                .get(tool_call_id)
                .ok_or(ArtifactValidationError::InvalidCanonicalLog)?;
            contexts.insert(
                *approval_id,
                ApprovalContext {
                    profile_id: state.configuration().profile_id().clone(),
                    tool_name: tool.tool_name().to_owned(),
                },
            );
        }
    }
    Ok(contexts)
}

fn same_grant_before_revocation(issued: &PolicyGrant, revoked: &PolicyGrant) -> bool {
    let (Ok(mut issued_value), Ok(mut revoked_value)) =
        (serde_json::to_value(issued), serde_json::to_value(revoked))
    else {
        return false;
    };
    issued_value["revokedAt"] = serde_json::Value::Null;
    revoked_value["revokedAt"] = serde_json::Value::Null;
    issued_value == revoked_value
}

/// Typed approval/grant persistence invariant failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ArtifactValidationError {
    /// Canonical source records do not form a valid session log.
    #[error("approval artifact canonical session log is invalid")]
    InvalidCanonicalLog,
    /// The linked canonical record is absent from durable history.
    #[error("approval artifact canonical record is missing")]
    MissingCanonicalRecord,
    /// Rich request and canonical request fields differ.
    #[error("approval artifact does not match canonical transition")]
    CanonicalApprovalMismatch,
    /// Rich resolution has no prior matching rich request.
    #[error("approval resolution artifact has no matching request")]
    MissingApprovalRequest,
    /// A rich request identity was persisted more than once.
    #[error("approval request artifact is duplicated")]
    DuplicateApprovalArtifact,
    /// Rich resolution snapshot differs from its stored request.
    #[error("approval resolution artifact does not match request")]
    ApprovalResolutionMismatch,
    /// Grant issuance is not linked to a rich approval resolution in this transaction.
    #[error("grant issuance has no matching approval resolution")]
    GrantWithoutResolution,
    /// Rich resolution issued a grant without the matching journal fact.
    #[error("approval resolution grant is missing from the grant journal")]
    MissingGrantJournalEntry,
    /// Issued grant differs from the validated approval resolution.
    #[error("issued grant does not match approval resolution")]
    GrantMismatch,
    /// Grant identity was issued more than once.
    #[error("grant identity is duplicated")]
    DuplicateGrant,
    /// Revocation references a grant that was never issued.
    #[error("grant revocation references an unknown grant")]
    UnknownGrant,
    /// Revoked value is missing or changes immutable grant fields.
    #[error("grant revocation is invalid")]
    InvalidRevocation,
}

impl ArtifactValidationError {
    /// Returns stable storage-facing classification.
    #[must_use]
    pub const fn store_code(self) -> SessionStoreErrorCode {
        match self {
            Self::MissingCanonicalRecord
            | Self::MissingApprovalRequest
            | Self::GrantWithoutResolution
            | Self::UnknownGrant => SessionStoreErrorCode::InvalidReference,
            Self::InvalidCanonicalLog
            | Self::CanonicalApprovalMismatch
            | Self::DuplicateApprovalArtifact
            | Self::ApprovalResolutionMismatch
            | Self::MissingGrantJournalEntry
            | Self::GrantMismatch
            | Self::DuplicateGrant
            | Self::InvalidRevocation => SessionStoreErrorCode::InvalidRecord,
        }
    }
}
