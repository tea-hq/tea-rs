use std::str::FromStr;
use std::sync::Arc;

use futures_util::stream;
use tea_model::{
    BoxModelStream, ModelCancellation, ModelCapabilities, ModelDisplayName, ModelProvider,
    ModelRef, ModelRegistry, ModelRegistryError, ModelRequest, ModelRouter, ModelSpec, ProviderId,
};
use tea_protocol::{ModelId, TokenCount};

#[derive(Debug)]
struct FixtureProvider {
    provider_id: ProviderId,
    models: Vec<ModelSpec>,
}

impl ModelProvider for FixtureProvider {
    fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    fn models(&self) -> &[ModelSpec] {
        &self.models
    }

    fn stream(&self, _request: ModelRequest, _cancellation: ModelCancellation) -> BoxModelStream {
        Box::pin(stream::empty())
    }
}

fn model(provider_id: &str, model_id: &str) -> ModelSpec {
    ModelSpec::new(
        ModelId::from_str(model_id).unwrap(),
        ProviderId::from_str(provider_id).unwrap(),
        ModelDisplayName::from_str(&format!("{provider_id} {model_id}")).unwrap(),
        TokenCount::new(8_000).unwrap(),
        TokenCount::new(2_000).unwrap(),
        ModelCapabilities::text(),
    )
    .unwrap()
}

fn provider(provider_id: &str, models: Vec<ModelSpec>) -> Arc<dyn ModelProvider> {
    Arc::new(FixtureProvider {
        provider_id: ProviderId::from_str(provider_id).unwrap(),
        models,
    })
}

fn model_ref(provider_id: &str, model_id: &str) -> ModelRef {
    ModelRef::new(
        ProviderId::from_str(provider_id).unwrap(),
        ModelId::from_str(model_id).unwrap(),
    )
}

#[test]
fn empty_registry_is_a_valid_generation() {
    let registry = ModelRegistry::empty();

    assert_eq!(registry.provider_count(), 0);
    assert!(registry.provider_ids().is_empty());
    assert!(registry.models().is_empty());
}

#[test]
fn registry_rejects_duplicate_provider_identities() {
    let error = ModelRegistry::new([
        provider("one", vec![model("one", "shared")]),
        provider("one", vec![model("one", "other")]),
    ])
    .unwrap_err();

    assert_eq!(
        error,
        ModelRegistryError::DuplicateProvider(ProviderId::from_str("one").unwrap())
    );
}

#[test]
fn registry_rejects_models_owned_by_another_provider() {
    let error = ModelRegistry::new([provider("one", vec![model("two", "shared")])]).unwrap_err();

    assert_eq!(
        error,
        ModelRegistryError::ProviderCatalogMismatch(ProviderId::from_str("one").unwrap())
    );
}

#[test]
fn same_model_id_is_resolved_by_provider_qualified_identity() {
    let registry = ModelRegistry::new([
        provider("one", vec![model("one", "shared")]),
        provider("two", vec![model("two", "shared")]),
    ])
    .unwrap();

    assert_eq!(registry.provider_count(), 2);
    assert_eq!(
        registry
            .model(&model_ref("one", "shared"))
            .unwrap()
            .provider_id()
            .as_str(),
        "one"
    );
    assert_eq!(
        registry
            .model(&model_ref("two", "shared"))
            .unwrap()
            .provider_id()
            .as_str(),
        "two"
    );
    assert!(registry.model(&model_ref("missing", "shared")).is_none());
}

#[test]
fn registration_publishes_a_new_generation_without_mutating_the_old_one() {
    let first = provider("one", vec![model("one", "shared")]);
    let original = ModelRegistry::new([Arc::clone(&first)]).unwrap();

    let next = original
        .with_registered([provider("two", vec![model("two", "shared")])])
        .unwrap();

    assert_eq!(original.provider_ids(), ["one".parse().unwrap()]);
    assert_eq!(
        next.provider_ids(),
        ["one".parse().unwrap(), "two".parse().unwrap()]
    );
    assert!(Arc::ptr_eq(
        &original.provider_arc(first.provider_id()).unwrap(),
        &first
    ));
}

#[test]
fn batch_registration_is_atomic_when_a_provider_id_is_duplicated() {
    let original = ModelRegistry::new([provider("one", vec![model("one", "shared")])]).unwrap();

    let error = original
        .with_registered([
            provider("two", vec![model("two", "shared")]),
            provider("one", vec![model("one", "other")]),
        ])
        .unwrap_err();

    assert_eq!(
        error,
        ModelRegistryError::DuplicateProvider("one".parse().unwrap())
    );
    assert_eq!(original.provider_ids(), ["one".parse().unwrap()]);
    assert!(original.provider(&"two".parse().unwrap()).is_none());
}

#[test]
fn removal_publishes_an_empty_generation_while_the_old_generation_stays_usable() {
    let first = provider("one", vec![model("one", "shared")]);
    let original = ModelRegistry::new([Arc::clone(&first)]).unwrap();

    let next = original
        .without_providers([first.provider_id().clone()])
        .unwrap();

    assert_eq!(next.provider_count(), 0);
    assert!(next.provider(first.provider_id()).is_none());
    assert!(Arc::ptr_eq(
        &original.provider_arc(first.provider_id()).unwrap(),
        &first
    ));
    assert!(original.model(&model_ref("one", "shared")).is_some());
}

#[test]
fn removal_rejects_an_unknown_provider_without_changing_the_generation() {
    let original = ModelRegistry::new([provider("one", vec![model("one", "shared")])]).unwrap();
    let missing = ProviderId::from_str("missing").unwrap();

    let error = original.without_providers([missing.clone()]).unwrap_err();

    assert_eq!(error, ModelRegistryError::UnknownProvider(missing));
    assert_eq!(original.provider_ids(), ["one".parse().unwrap()]);
}
