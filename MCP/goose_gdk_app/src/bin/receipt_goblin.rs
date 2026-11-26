//! receipt-goblin: a goose GDK app that checks a document's numeric claims against run receipts.
//!
//! Each claim runs through a `goose-agent` state machine with three steps, in priority order:
//!
//! 1. `CertifyOperation` applies once the model has answered. It reads the packet leaf the model
//!    linked (`{"leaf": "L07"}` or `NONE`), asks once for a repair when the answer names no packet
//!    leaf, and otherwise has the server compare the claimed number with that leaf's value.
//! 2. `ReceiptPacketOperation` contributes the instructions and the claim's receipt packet as prompt
//!    parts.
//! 3. `InferenceRunner` sends the conversation to an OpenAI-compatible endpoint (`goose-providers`).
//!
//! The Receipt Goblin MCP server (`python -m MCP.receipt_goblin.server`) extracts claims, builds the
//! packets and certifies links. Output is one JSON receipt per claim (JSONL).

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use goblin_gdk::mcp::McpClient;
use goose_agent::inference::{InferenceEffect, InferenceRunner};
use goose_agent::machine::{EffectHandler, MachineSession, SessionLoader, StateMachine, Step};
use goose_agent::operation::{applied, not_applicable, yielded, Emitter, MachineEffect, Operation, OperationResult};
use goose_provider_types::conversation::message::Message;
use goose_provider_types::conversation::token_usage::ProviderUsage;
use goose_provider_types::conversation::{effective_role, Conversation, EffectiveRole};
use goose_provider_types::model::ModelConfig;
use goose_providers::api_client::{ApiClient, AuthMethod};
use goose_providers::openai_compatible::OpenAiCompatibleProvider;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const INSTRUCTIONS: &str = "You check one numeric claim from a research write-up against the run receipts \
it was written from. The claim's number is marked like \u{27e6}this\u{27e7}. The packet below lists receipt \
leaves, one per line: an id, the receipt file, the path inside it and (unless withheld) the value. A path \
segment in [brackets] names the cell a list element describes (arm, policy, run and so on).\n\n\
Choose the ONE leaf that records the quantity the marked number describes: the same metric, for the same \
run, arm, model and setting the claim names. Decide by meaning, not by whether the value equals the marked \
number: numbers get copied from the wrong cell, and a later step compares the values. If no leaf records \
that quantity, answer NONE.\n\nReply with JSON only: {\"leaf\": \"L07\"} or {\"leaf\": \"NONE\"}.";

const REPAIR: &str = "Your answer did not name a leaf id from the packet. Reply with JSON only, for example \
{\"leaf\": \"L07\"}, using an id that appears in the packet, or {\"leaf\": \"NONE\"}.";

// ---- GDK session, effects and runtime ----

#[derive(Clone)]
struct AppSession {
    id: String,
    conversation: Conversation,
}

impl MachineSession for AppSession {
    fn id(&self) -> &str {
        &self.id
    }
    fn conversation(&self) -> Option<&Conversation> {
        Some(&self.conversation)
    }
}

enum AppEffect {
    Append(Message),
    Usage(ProviderUsage),
}

impl MachineEffect for AppEffect {
    fn ensure_message_ids(&mut self) {
        if let AppEffect::Append(message) = self {
            if message.id.is_none() {
                message.id = Some(format!("msg_{}", uuid::Uuid::new_v4()));
            }
        }
    }
}

impl From<Message> for AppEffect {
    fn from(message: Message) -> Self {
        AppEffect::Append(message)
    }
}

impl InferenceEffect for AppEffect {
    fn record_usage(usage: ProviderUsage) -> Self {
        AppEffect::Usage(usage)
    }
}

struct Runtime {
    session: Mutex<AppSession>,
    usage: Mutex<Vec<ProviderUsage>>,
}

#[async_trait]
impl SessionLoader<AppSession> for Runtime {
    async fn load(&self, _session_id: &str) -> Result<AppSession> {
        Ok(self.session.lock().unwrap().clone())
    }
}

#[async_trait]
impl EffectHandler<AppSession, AppEffect> for Runtime {
    async fn apply_effects(&self, _session: &AppSession, effects: &mut [AppEffect], _emit: &Emitter) -> Result<()> {
        for effect in effects.iter() {
            match effect {
                AppEffect::Append(message) => self.session.lock().unwrap().conversation.push(message.clone()),
                AppEffect::Usage(usage) => self.usage.lock().unwrap().push(usage.clone()),
            }
        }
        Ok(())
    }
}

// ---- operations ----

struct ReceiptPacketOperation {
    packet: String,
}

#[async_trait]
impl Operation<AppSession, AppEffect> for ReceiptPacketOperation {
    fn name(&self) -> &'static str {
        "receipt_packet"
    }

    async fn prompt_parts(&self, _session: &AppSession, _conversation: &Conversation) -> Result<Vec<(String, String)>> {
        Ok(vec![
            ("receipt_goblin_instructions".to_string(), INSTRUCTIONS.to_string()),
            ("receipt_packet".to_string(), self.packet.clone()),
        ])
    }
}

/// Reads the linked leaf from an answer: the `leaf` field of the first JSON object, else the first
/// `L<digits>` or `NONE` token. Ids are normalised to the packet's zero-padded form.
fn parse_leaf(answer: &str, ids: &HashSet<String>) -> Option<String> {
    let normalise = |raw: &str| -> Option<String> {
        let raw = raw.trim().to_uppercase();
        if raw == "NONE" {
            return Some(raw);
        }
        let digits = raw.strip_prefix('L')?;
        let number: u32 = digits.parse().ok()?;
        let id = format!("L{number:02}");
        ids.contains(&id).then_some(id)
    };
    if let (Some(start), Some(end)) = (answer.find('{'), answer.rfind('}')) {
        if start < end {
            if let Ok(value) = serde_json::from_str::<Value>(&answer[start..=end]) {
                if let Some(leaf) = value.get("leaf").and_then(Value::as_str) {
                    return normalise(leaf);
                }
            }
        }
    }
    answer
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|token| token.eq_ignore_ascii_case("none") || (token.starts_with('L') && token.len() > 1))
        .find_map(normalise)
}

struct CertifyOperation {
    mcp: Arc<tokio::sync::Mutex<McpClient>>,
    packet_id: String,
    leaf_ids: HashSet<String>,
    max_repairs: u32,
    repairs: Mutex<u32>,
    outcome: Mutex<Option<Value>>,
}

#[async_trait]
impl Operation<AppSession, AppEffect> for CertifyOperation {
    fn name(&self) -> &'static str {
        "certify"
    }

    async fn run(&self, _session: &AppSession, conversation: &Conversation, _emit: &Emitter) -> Result<OperationResult<AppEffect>> {
        let Some(last) = conversation.last() else {
            return not_applicable();
        };
        if effective_role(last) != EffectiveRole::Assistant {
            return not_applicable();
        }
        if self.outcome.lock().unwrap().is_some() {
            return yielded();
        }
        if let Some(kind) = last.error_kind() {
            *self.outcome.lock().unwrap() = Some(json!({"verdict": "error", "error": format!("{kind:?}")}));
            return yielded();
        }
        match parse_leaf(&last.as_concat_text(), &self.leaf_ids) {
            Some(leaf) => {
                let result = self
                    .mcp
                    .lock()
                    .await
                    .call_tool("certify", json!({"packet_id": self.packet_id, "leaf": leaf}))
                    .await?;
                *self.outcome.lock().unwrap() = Some(result);
                yielded()
            }
            None => {
                let mut repairs = self.repairs.lock().unwrap();
                if *repairs < self.max_repairs {
                    *repairs += 1;
                    applied([AppEffect::Append(Message::user().with_text(REPAIR))])
                } else {
                    *self.outcome.lock().unwrap() = Some(json!({"verdict": "unparsed"}));
                    yielded()
                }
            }
        }
    }
}

// ---- command line ----

struct Args {
    bench: Option<String>,
    doc: Option<String>,
    receipts: Option<String>,
    out: Option<String>,
    arm: String,
    base_url: String,
    model: String,
    python: String,
    server_cwd: String,
    budget: u64,
    k_lexical: u64,
    k_neighbours: u64,
    blind: bool,
    max_tokens: Option<i32>,
    max_repairs: u32,
    timeout_secs: u64,
    offset: usize,
    limit: Option<usize>,
}

const USAGE: &str = "usage: receipt-goblin (--bench BENCH.jsonl | --doc DOC.md --receipts DIR) [--out RECEIPTS.jsonl] \
[--arm NAME] [--base-url URL] [--model ID] [--python PY] [--server-cwd REPO] [--budget N] [--k-lexical N] \
[--k-neighbours N] [--blind] [--max-tokens N] [--max-repairs N] [--timeout-secs N] [--offset N] [--limit N]";

fn parse_args() -> Result<Args> {
    let mut args = Args {
        bench: None,
        doc: None,
        receipts: None,
        out: None,
        arm: "packet".into(),
        base_url: "http://127.0.0.1:18090/v1/".into(),
        model: "inkling-2-mini".into(),
        python: "python3".into(),
        server_cwd: ".".into(),
        budget: 3000,
        k_lexical: 40,
        k_neighbours: 8,
        blind: false,
        max_tokens: Some(2048),
        max_repairs: 1,
        timeout_secs: 600,
        offset: 0,
        limit: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| anyhow!("{flag} needs a value"));
        match flag.as_str() {
            "--bench" => args.bench = Some(value()?),
            "--doc" => args.doc = Some(value()?),
            "--receipts" => args.receipts = Some(value()?),
            "--out" => args.out = Some(value()?),
            "--arm" => args.arm = value()?,
            "--base-url" => args.base_url = value()?,
            "--model" => args.model = value()?,
            "--python" => args.python = value()?,
            "--server-cwd" => args.server_cwd = value()?,
            "--budget" => args.budget = value()?.parse()?,
            "--k-lexical" => args.k_lexical = value()?.parse()?,
            "--k-neighbours" => args.k_neighbours = value()?.parse()?,
            "--blind" => args.blind = true,
            "--max-tokens" => args.max_tokens = Some(value()?.parse()?),
            "--max-repairs" => args.max_repairs = value()?.parse()?,
            "--timeout-secs" => args.timeout_secs = value()?.parse()?,
            "--offset" => args.offset = value()?.parse()?,
            "--limit" => args.limit = Some(value()?.parse()?),
            "--help" | "-h" => bail!("{USAGE}"),
            other => bail!("unknown flag {other}\n{USAGE}"),
        }
    }
    if args.bench.is_none() && (args.doc.is_none() || args.receipts.is_none()) {
        bail!("{USAGE}");
    }
    if !args.base_url.ends_with('/') {
        args.base_url.push('/');
    }
    Ok(args)
}

/// Claims to check: benchmark items, or every number the server extracts from `--doc`.
async fn load_claims(args: &Args, mcp: &tokio::sync::Mutex<McpClient>) -> Result<Vec<Value>> {
    if let Some(bench) = &args.bench {
        return std::fs::read_to_string(bench)?
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| Ok(serde_json::from_str(line)?))
            .collect();
    }
    let doc = args.doc.clone().unwrap_or_default();
    let receipts = args.receipts.clone().unwrap_or_default();
    let listed = mcp.lock().await.call_tool("extract_claims", json!({"doc_path": doc})).await?;
    let claims = listed["claims"].as_array().cloned().unwrap_or_default();
    Ok(claims
        .into_iter()
        .map(|mut claim| {
            claim["id"] = claim["cid"].clone();
            claim["receipts"] = json!(receipts);
            claim
        })
        .collect())
}

fn done_ids(out: &Option<String>, arm: &str) -> HashSet<String> {
    let Some(path) = out else { return HashSet::new() };
    let Ok(text) = std::fs::read_to_string(path) else { return HashSet::new() };
    text.lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|row| row["arm"] == arm)
        .filter_map(|row| row["id"].as_str().map(str::to_string))
        .collect()
}

async fn check_claim(
    args: &Args,
    claim: &Value,
    mcp: &Arc<tokio::sync::Mutex<McpClient>>,
    provider: &Arc<OpenAiCompatibleProvider>,
) -> Result<Value> {
    let started = Instant::now();
    let packet = mcp
        .lock()
        .await
        .call_tool(
            "claim_packet",
            json!({"text": claim["text"], "start": claim["start"], "end": claim["end"],
                   "receipts_root": claim["receipts"], "budget_tokens": args.budget,
                   "k_lexical": args.k_lexical, "k_neighbours": args.k_neighbours, "blind": args.blind}),
        )
        .await?;
    let packet_seconds = started.elapsed().as_secs_f64();
    let packet_id = packet["packet_id"].as_str().ok_or_else(|| anyhow!("packet has no id"))?.to_string();
    let packet_text = packet["packet"].as_str().unwrap_or_default().to_string();
    let leaf_ids: HashSet<String> = packet_text
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .filter(|token| token.starts_with('L') && token[1..].chars().all(|c| c.is_ascii_digit()))
        .map(str::to_string)
        .collect();

    let certify = Arc::new(CertifyOperation {
        mcp: mcp.clone(),
        packet_id: packet_id.clone(),
        leaf_ids,
        max_repairs: args.max_repairs,
        repairs: Mutex::new(0),
        outcome: Mutex::new(None),
    });
    let model = ModelConfig::new(&args.model)
        .with_max_tokens(args.max_tokens)
        .with_temperature(Some(0.0))
        .with_context_limit(Some(16384));
    let machine = StateMachine::new(
        vec![
            Step::Operation(certify.clone() as Arc<dyn Operation<AppSession, AppEffect>>),
            Step::Operation(Arc::new(ReceiptPacketOperation { packet: packet_text })),
            Step::Inference(Arc::new(InferenceRunner::new(provider.clone(), model))),
        ],
        CancellationToken::new(),
    );
    let question = format!(
        "Claim: {}\n\nWhich packet leaf records the quantity that the marked number \u{27e6}{}\u{27e7} describes?",
        packet["claim"].as_str().unwrap_or_default(),
        packet["target"].as_str().unwrap_or_default()
    );
    let runtime = Runtime {
        session: Mutex::new(AppSession {
            id: "receipt".into(),
            conversation: Conversation::new_unvalidated([Message::user().with_text(question)]),
        }),
        usage: Mutex::new(Vec::new()),
    };
    let (tx, mut rx) = mpsc::unbounded_channel();
    let emit = Emitter::new(tx, CancellationToken::new());
    tokio::spawn(async move { while rx.recv().await.is_some() {} });
    let inference_started = Instant::now();
    let session = machine.run(&runtime, "receipt", &emit).await?;
    let answers: Vec<String> = session
        .conversation
        .iter()
        .filter(|m| effective_role(m) == EffectiveRole::Assistant)
        .map(|m| m.as_concat_text())
        .collect();
    let usage = runtime.usage.lock().unwrap().clone();
    let outcome = certify.outcome.lock().unwrap().clone().unwrap_or_else(|| json!({"verdict": "no_answer"}));
    let repairs = *certify.repairs.lock().unwrap();
    Ok(json!({
        "app": "receipt-goblin", "gdk": "goose-agent 0.1.0-alpha.14", "arm": args.arm, "model": args.model,
        "id": claim["id"], "variant": claim.get("variant").cloned().unwrap_or(json!("doc")),
        "label": claim.get("label").cloned().unwrap_or(Value::Null),
        "gold_leaves": claim.get("gold_leaves").cloned().unwrap_or(Value::Null),
        "target": packet["target"], "verdict": outcome["verdict"], "leaf_id": outcome.get("leaf_id").cloned().unwrap_or(Value::Null),
        "leaf": outcome.get("leaf").cloned().unwrap_or(Value::Null),
        "leaf_value": outcome.get("leaf_value").cloned().unwrap_or(Value::Null),
        "outcome": outcome, "answers": answers, "repairs": repairs,
        "packet_id": packet_id, "packet_tokens": packet["packet_tokens"], "packet_leaf_paths": packet["leaf_paths"],
        "number_matches": packet["number_matches"], "blind": args.blind,
        "budget": args.budget, "k_lexical": args.k_lexical, "k_neighbours": args.k_neighbours,
        "provider_input_tokens": usage.iter().filter_map(|u| u.usage.input_tokens).sum::<i64>(),
        "provider_output_tokens": usage.iter().filter_map(|u| u.usage.output_tokens).sum::<i64>(),
        "inference_calls": usage.len(), "packet_seconds": packet_seconds,
        "inference_seconds": inference_started.elapsed().as_secs_f64(),
    }))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse_args()?;
    let mcp = Arc::new(tokio::sync::Mutex::new(
        McpClient::spawn(&args.python, &["-m", "MCP.receipt_goblin.server"], &args.server_cwd, "receipt-goblin").await?,
    ));
    let claims = load_claims(&args, &mcp).await?;
    let done = done_ids(&args.out, &args.arm);
    let api = ApiClient::with_timeout_and_tls(
        args.base_url.clone(),
        AuthMethod::NoAuth,
        std::time::Duration::from_secs(args.timeout_secs),
        None,
    )?;
    let provider = Arc::new(OpenAiCompatibleProvider::new("local".into(), api, String::new()).with_supports_streaming(false));
    let mut sink: Box<dyn Write> = match &args.out {
        Some(path) => Box::new(std::fs::OpenOptions::new().create(true).append(true).open(path)?),
        None => Box::new(std::io::stdout()),
    };
    let selected = claims.iter().skip(args.offset).take(args.limit.unwrap_or(usize::MAX));
    let mut counts: HashMap<String, u64> = HashMap::new();
    for claim in selected {
        let id = claim["id"].as_str().unwrap_or_default().to_string();
        if done.contains(&id) {
            continue;
        }
        let row = match check_claim(&args, claim, &mcp, &provider).await {
            Ok(row) => row,
            Err(error) => json!({"app": "receipt-goblin", "arm": args.arm, "model": args.model, "id": id,
                                 "variant": claim.get("variant").cloned().unwrap_or(json!("doc")),
                                 "label": claim.get("label").cloned().unwrap_or(Value::Null),
                                 "gold_leaves": claim.get("gold_leaves").cloned().unwrap_or(Value::Null),
                                 "verdict": "error", "error": format!("{error:#}")}),
        };
        *counts.entry(row["verdict"].as_str().unwrap_or("?").to_string()).or_default() += 1;
        writeln!(sink, "{}", serde_json::to_string(&row)?)?;
        sink.flush()?;
    }
    eprintln!("{}", json!({"arm": args.arm, "checked": counts.values().sum::<u64>(), "verdicts": counts}));
    Ok(())
}
