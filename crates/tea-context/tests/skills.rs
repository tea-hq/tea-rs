use std::str::FromStr;

use futures_util::FutureExt;
use tea_context::{
    ContextProvider, ContextRequest, PromptBudget, PromptCompiler, SkillCommand, SkillId,
    SkillInvocation, SkillMetadata, SkillMetadataProvider,
};
use tea_protocol::{ProfileId, ProtocolMetadata, SessionId};

fn request() -> ContextRequest {
    ContextRequest::new(
        ProfileId::from_str("coding").unwrap(),
        SessionId::from_str("0195a0b1-5e3a-7d72-a902-c4e85d828bf1").unwrap(),
        None,
        vec![],
        ProtocolMetadata::default(),
    )
    .unwrap()
}

#[test]
fn skill_invocation_uses_one_exact_explicit_format() {
    let invocation = SkillInvocation::from_str("@skill code.review").unwrap();
    assert_eq!(invocation.skill_id().as_str(), "code.review");
    assert_eq!(invocation.to_string(), "@skill code.review");
    let legacy_command = invocation.command();
    assert_eq!(legacy_command.skill_id().as_str(), "code.review");
    assert!(legacy_command.arguments().is_empty());
    assert_eq!(legacy_command.to_string(), "/skill:code.review");
    for invalid in [
        "code.review",
        "@code.review",
        "@skill",
        "@skill  code.review",
        "@skill code.review extra",
    ] {
        assert!(SkillInvocation::from_str(invalid).is_err(), "{invalid}");
    }
}

#[test]
fn skill_command_parses_canonical_slash_form() {
    let command = SkillCommand::from_str("/skill:code.review src/lib.rs").unwrap();
    assert_eq!(command.skill_id().as_str(), "code.review");
    assert_eq!(command.arguments(), "src/lib.rs");
    assert_eq!(command.to_string(), "/skill:code.review src/lib.rs");
    for invalid in [
        "skill:code.review",
        "/skill:",
        "/skill:code.review\textra",
        "/skill:code.review\0",
    ] {
        assert!(SkillCommand::from_str(invalid).is_err(), "{invalid}");
    }
}

#[test]
fn active_skill_metadata_is_sorted_and_does_not_execute() {
    let provider = SkillMetadataProvider::new(vec![
        SkillMetadata::new(SkillId::from_str("z.skill").unwrap(), "Z description.").unwrap(),
        SkillMetadata::new(SkillId::from_str("a.skill").unwrap(), "A description.").unwrap(),
    ])
    .unwrap();
    let modules = provider.provide(request()).now_or_never().unwrap().unwrap();
    let prompt = PromptCompiler
        .compile(modules, PromptBudget::new(4096, 4096).unwrap())
        .unwrap();
    assert!(prompt.text().starts_with("Skill `a.skill`"));
    assert!(prompt.text().contains("`/skill:a.skill`"));
    assert!(prompt.text().contains("`/skill:z.skill`"));
}

#[test]
fn duplicate_skills_and_description_bounds_fail_closed() {
    let skill = SkillMetadata::new(SkillId::from_str("same").unwrap(), "description").unwrap();
    assert!(SkillMetadataProvider::new(vec![skill.clone(), skill]).is_err());
    assert!(SkillMetadata::new(SkillId::from_str("empty").unwrap(), "").is_err());
}
