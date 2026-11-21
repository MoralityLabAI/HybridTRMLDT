//! GDK-01 provider-boundary view: bound a serialized representation of the entire
//! provider input while leaving the state machine's durable conversation unchanged.
//!
//! This is a *byte* gate over Goose's internal request objects, not a provider
//! tokenizer / chat-template token gate. A model-specific final wire preflight
//! remains a separate integration milestone.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use goose_provider_types::base::{MessageStream, Provider};
use goose_provider_types::conversation::message::Message;
use goose_provider_types::errors::ProviderError;
use goose_provider_types::model::ModelConfig;
use rmcp::model::{Role, Tool};
use serde::Serialize;
use serde_json::json;

#[derive(Clone, Debug, Serialize)]
pub struct LeanReceipt {
    pub gate: &'static str,
    pub max_serialized_input_bytes: usize,
    pub serialized_input_bytes: Option<usize>,
    pub original_messages: usize,
    pub visible_messages: usize,
    pub omitted_messages: usize,
    pub blocked: bool,
    pub reason: Option<String>,
}

/// Measure a conservative **internal JSON representation**, including every
/// tool schema. This is not the wire-format request sent by the provider.
fn request_bytes(
    model: &ModelConfig,
    system: &str,
    messages: &[Message],
    tools: &[Tool],
) -> Result<usize, ProviderError> {
    serde_json::to_vec(&json!({
        "model": &model.model_name,
        "system": system,
        "messages": messages,
        "tools": tools
    }))
    .map(|bytes| bytes.len())
    .map_err(|e| ProviderError::ExecutionError(format!("GDK lean serialization: {e}")))
}

/// Select whole kickoff-to-kickoff turns. A user-role tool response is NOT a
/// kickoff, so its assistant tool request and response stay together. The
/// newest user kickoff and all subsequent tool results are mandatory.
/// When this current turn cannot fit, fail closed rather than trim evidence.
pub fn plan_request(
    model: &ModelConfig,
    system: &str,
    messages: &[Message],
    tools: &[Tool],
    max_bytes: usize,
    previous_turns: usize,
) -> Result<(Vec<Message>, LeanReceipt), ProviderError> {
    if max_bytes == 0 {
        return Err(ProviderError::InvalidValue("GDK byte limit must be positive".into()));
    }
    let kickoffs: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, msg)| msg.role == Role::User && msg.get_tool_response_ids().is_empty())
        .map(|(index, _)| index)
        .collect();
    let current = *kickoffs
        .last()
        .ok_or_else(|| ProviderError::InvalidValue("GDK view requires a user kickoff".into()))?;

    let mut offset = current;
    let mut counted = request_bytes(model, system, &messages[offset..], tools)?;
    if counted > max_bytes {
        return Err(ProviderError::ContextLengthExceeded(format!(
            "GDK current task + packet + tool schemas need {counted} internal JSON bytes; limit {max_bytes}"
        )));
    }
    for &older in kickoffs.iter().rev().skip(1).take(previous_turns) {
        let trial = request_bytes(model, system, &messages[older..], tools)?;
        if trial > max_bytes {
            break;
        }
        offset = older;
        counted = trial;
    }

    let view = messages[offset..].to_vec();
    Ok((
        view,
        LeanReceipt {
            gate: "internal_goose_json_bytes_v1",
            max_serialized_input_bytes: max_bytes,
            serialized_input_bytes: Some(counted),
            original_messages: messages.len(),
            visible_messages: messages.len() - offset,
            omitted_messages: offset,
            blocked: false,
            reason: None,
        },
    ))
}

/// Provider wrapper, so the GDK inference runner can keep its normal
/// effects, retries and session lifetime. RequestPreparer cannot remove
/// historical messages; the provider boundary receives the complete view.
pub struct LeanProvider {
    inner: Arc<dyn Provider>,
    max_bytes: usize,
    previous_turns: usize,
    last: Mutex<Option<LeanReceipt>>,
}

impl LeanProvider {
    pub fn new(inner: Arc<dyn Provider>, max_bytes: usize, previous_turns: usize) -> Self {
        Self { inner, max_bytes, previous_turns, last: Mutex::new(None) }
    }

    pub fn receipt(&self) -> Option<LeanReceipt> {
        self.last.lock().unwrap().clone()
    }
}

#[async_trait]
impl Provider for LeanProvider {
    fn get_name(&self) -> &str {
        self.inner.get_name()
    }

    async fn stream(
        &self,
        model_config: &ModelConfig,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        let planned = plan_request(
            model_config,
            system,
            messages,
            tools,
            self.max_bytes,
            self.previous_turns,
        );
        let (view, receipt) = match planned {
            Ok(value) => value,
            Err(error) => {
                *self.last.lock().unwrap() = Some(LeanReceipt {
                    gate: "internal_goose_json_bytes_v1",
                    max_serialized_input_bytes: self.max_bytes,
                    serialized_input_bytes: None,
                    original_messages: messages.len(),
                    visible_messages: 0,
                    omitted_messages: 0,
                    blocked: true,
                    reason: Some(error.to_string()),
                });
                return Err(error);
            }
        };
        *self.last.lock().unwrap() = Some(receipt);
        self.inner.stream(model_config, system, &view, tools).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn messages() -> Vec<Message> {
        vec![
            Message::user().with_text("prior task marker: should be outside lean request"),
            Message::assistant().with_text("prior response"),
            Message::user().with_text("current task"),
            Message::assistant().with_text("working"),
            Message::user().with_text("tool response placeholder"), // synthetic ordinary user
        ]
    }

    #[test]
    fn current_turn_is_mandatory_and_history_is_bounded() {
        let input = messages();
        let model = ModelConfig::new("dummy");
        let (view, receipt) = plan_request(&model, "", &input, &[], 32_768, 0).unwrap();
        assert_eq!(view.len(), 1);
        assert_eq!(view[0].as_concat_text(), "tool response placeholder");
        assert_eq!(receipt.omitted_messages, 4);
        let (view, receipt) = plan_request(&model, "", &input, &[], 32_768, 1).unwrap();
        assert_eq!(view.len(), 3);
        assert_eq!(receipt.visible_messages, 3);
        assert!(view.iter().any(|m| m.as_concat_text() == "current task"));
    }

    #[test]
    fn whole_turn_retention() {
        let model = ModelConfig::new("dummy");
        let input = vec![
            Message::user().with_text("old task"),
            Message::assistant().with_text("old answer"),
            Message::user().with_text("current task"),
            Message::assistant().with_text("intermediate answer"),
        ];
        let (view, receipt) = plan_request(&model, "", &input, &[], 10_000, 0).unwrap();
        assert_eq!(receipt.omitted_messages, 2);
        assert_eq!(view.len(), 2);
        assert_eq!(view[0].as_concat_text(), "current task");
    }

    #[test]
    fn overlarge_current_task_blocks_before_inference() {
        let model = ModelConfig::new("dummy");
        let input = vec![Message::user().with_text("x".repeat(8000))];
        assert!(matches!(
            plan_request(&model, "", &input, &[], 1000, 0),
            Err(ProviderError::ContextLengthExceeded(_))
        ));
    }

    #[test]
    fn schemas_and_system_count_toward_limit() {
        let model = ModelConfig::new("dummy");
        let input = vec![Message::user().with_text("task")];
        let small = request_bytes(&model, "", &input, &[]).unwrap();
        assert!(request_bytes(&model, &"s".repeat(1000), &input, &[]).unwrap() > small);
        assert!(plan_request(&model, &"s".repeat(1000), &input, &[], small, 0).is_err());
    }

    #[test]
    fn no_user_task_is_rejected() {
        let model = ModelConfig::new("dummy");
        assert!(plan_request(
            &model, "", &[Message::assistant().with_text("orphan")], &[], 1000, 0
        ).is_err());
    }
}
