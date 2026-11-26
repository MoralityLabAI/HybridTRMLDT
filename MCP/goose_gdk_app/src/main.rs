//! goblin-gdk: a Goose GDK app built on the `goose-agent` state machine.
//!
//! The app assembles two GDK steps. `CompactContextOperation` contributes a prompt part holding a
//! compact code-context packet. `InferenceRunner` then calls an OpenAI-compatible provider from
//! `goose-providers`. The packet comes from the Data Goblin MCP server
//! (`python -m MCP.mcp03.goblin_server`), which runs a small local router over the repository's
//! code MCP surface and returns only a token-budgeted packet. With `--no-packet` the same machine
//! runs without the operation's contribution, so the two runs differ only in the packet.
//!
//! The app prints one JSON receipt: answer, provider-reported input/output tokens, packet
//! accounting from the goblin server, and timings. With `--max-agent-iterations` it runs the
//! multi-turn Pocket Goblin repair session instead (see `session.rs`). With `--acp` it serves the
//! Agent Client Protocol on stdio, which is how TitanAgent squads on OpenAB run their agents (see
//! `acp.rs`); the repair-session flags make each ACP prompt a repair session.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use goose_agent::inference::{InferenceEffect, InferenceRunner};
use goose_agent::machine::{EffectHandler, MachineSession, SessionLoader, StateMachine, Step};
use goose_agent::operation::{Emitter, MachineEffect, Operation};
use goose_provider_types::conversation::message::Message;
use goose_provider_types::conversation::token_usage::ProviderUsage;
use goose_provider_types::conversation::Conversation;
use goose_provider_types::model::ModelConfig;
use goose_providers::api_client::{ApiClient, AuthMethod};
use goose_providers::openai_compatible::OpenAiCompatibleProvider;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

mod acp;
mod session;

// ---- minimal MCP stdio client: newline-delimited JSON-RPC 2.0, protocol 2025-06-18 ----

struct McpClient {
    _child: Child,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
    next_id: i64,
}

impl McpClient {
    async fn spawn(program: &str, args: &[&str], cwd: &str) -> Result<Self> {
        let mut child = Command::new(program)
            .args(args)
            .current_dir(cwd)
            .env("PYTHONPATH", cwd)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("spawning {program}"))?;
        let stdin = child.stdin.take().ok_or_else(|| anyhow!("goblin server has no stdin"))?;
        let stdout = BufReader::new(child.stdout.take().ok_or_else(|| anyhow!("goblin server has no stdout"))?).lines();
        let mut client = Self { _child: child, stdin, stdout, next_id: 0 };
        client
            .request(
                "initialize",
                json!({"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "goblin-gdk", "version": env!("CARGO_PKG_VERSION")}}),
            )
            .await?;
        client.send(json!({"jsonrpc": "2.0", "method": "notifications/initialized"})).await?;
        Ok(client)
    }

    async fn send(&mut self, message: Value) -> Result<()> {
        let mut line = serde_json::to_vec(&message)?;
        line.push(b'\n');
        self.stdin.write_all(&line).await?;
        self.stdin.flush().await?;
        Ok(())
    }

    async fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        self.next_id += 1;
        let id = self.next_id;
        self.send(json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})).await?;
        while let Some(line) = self.stdout.next_line().await? {
            let message: Value = serde_json::from_str(&line)?;
            if message.get("id").and_then(Value::as_i64) == Some(id) {
                if let Some(error) = message.get("error") {
                    bail!("MCP error from {method}: {error}");
                }
                return Ok(message["result"].clone());
            }
        }
        bail!("goblin server closed its output")
    }

    async fn call_tool(&mut self, name: &str, arguments: Value) -> Result<Value> {
        let result = self.request("tools/call", json!({"name": name, "arguments": arguments})).await?;
        if result.get("isError").and_then(Value::as_bool) == Some(true) {
            bail!("tool {name} failed: {result}");
        }
        Ok(result.get("structuredContent").cloned().unwrap_or(result))
    }
}

// ---- GDK session, effects and runtime ----

#[derive(Clone)]
struct AppSession {
    id: String,
    conversation: Conversation,
    repair: Option<session::RepairState>,
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
    NormalizeLast(Message),
    Usage(ProviderUsage),
    Repair(session::RepairState),
}

impl MachineEffect for AppEffect {
    fn ensure_message_ids(&mut self) {
        if let AppEffect::Append(message) | AppEffect::NormalizeLast(message) = self {
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
    checkpoint: Option<std::path::PathBuf>,
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
                AppEffect::NormalizeLast(message) => {
                    let mut session = self.session.lock().unwrap();
                    let len = session.conversation.len();
                    session.conversation = Conversation::new_unvalidated(
                        session.conversation.iter().take(len.saturating_sub(1)).cloned()
                            .chain([message.clone()]),
                    );
                }
                AppEffect::Usage(usage) => {
                    self.usage.lock().unwrap().push(usage.clone());
                    if let Some(state) = &mut self.session.lock().unwrap().repair {
                        state.iterations += 1;
                        state.manager_steps = 0;
                    }
                }
                AppEffect::Repair(repair) => self.session.lock().unwrap().repair = Some(repair.clone()),
            }
        }
        if let Some(path) = &self.checkpoint {
            let session = self.session.lock().unwrap();
            let temp = path.with_extension("tmp");
            std::fs::write(&temp, serde_json::to_vec(&json!({
                "conversation": session.conversation.messages(),
                "session": session.repair.as_ref().map(session::RepairState::receipt),
            }))?)?;
            std::fs::rename(&temp, path)?;
        }
        Ok(())
    }
}

// ---- the operation that grounds inference in the compact packet ----

struct CompactContextOperation {
    packet: String,
}

#[async_trait]
impl Operation<AppSession, AppEffect> for CompactContextOperation {
    fn name(&self) -> &'static str {
        "compact_context"
    }

    async fn prompt_parts(&self, _session: &AppSession, _conversation: &Conversation) -> Result<Vec<(String, String)>> {
        if self.packet.is_empty() {
            return Ok(Vec::new());
        }
        Ok(vec![("retrieved_code_context".to_string(), self.packet.clone())])
    }
}

// ---- one GDK turn: the goblin packet, then inference ----

/// Ask the goblin server for the compact packet for `task` over `repo`. Returns the packet text and
/// the server's accounting without the packet.
async fn fetch_packet(goblin: &mut McpClient, args: &Args, task: &str, repo: &str) -> Result<(String, Value)> {
    let mut result = goblin
        .call_tool(
            "compact_context",
            json!({"task": task, "repo_path": repo, "budget_tokens": args.budget,
                   "max_tool_calls": args.max_tool_calls, "arm": args.arm}),
        )
        .await?;
    let packet = result["packet"].as_str().unwrap_or_default().to_string();
    if let Some(object) = result.as_object_mut() {
        object.remove("packet");
    }
    Ok((packet, result))
}

/// Run the two-step machine over `conversation`, whose last message is the user's turn, and return
/// the final conversation with the usage of each inference call. Cancelling `cancel` stops the
/// machine at its next step boundary or during inference.
async fn infer(
    args: &Args,
    conversation: Conversation,
    packet: String,
    cancel: CancellationToken,
) -> Result<(Conversation, Vec<ProviderUsage>)> {
    let api = ApiClient::with_timeout_and_tls(
        args.base_url.clone(),
        AuthMethod::NoAuth,
        std::time::Duration::from_secs(args.timeout_secs),
        None,
    )?;
    let provider = Arc::new(OpenAiCompatibleProvider::new("local".into(), api, String::new()).with_supports_streaming(false));
    let machine = StateMachine::new(
        vec![
            Step::Operation(Arc::new(CompactContextOperation { packet })),
            Step::Inference(Arc::new(InferenceRunner::new(
                provider,
                ModelConfig::new(&args.model).with_max_tokens(args.max_tokens),
            ))),
        ],
        cancel.clone(),
    );
    let runtime = Runtime {
        session: Mutex::new(AppSession { id: "goblin".into(), conversation, repair: None }),
        usage: Mutex::new(Vec::new()),
        checkpoint: None,
    };
    let (tx, mut rx) = mpsc::unbounded_channel();
    let emit = Emitter::new(tx, cancel);
    tokio::spawn(async move { while rx.recv().await.is_some() {} });
    let session = machine.run(&runtime, "goblin", &emit).await?;
    let usage = runtime.usage.into_inner().unwrap();
    Ok((session.conversation, usage))
}

/// Provider-reported (input, output) tokens summed over inference calls.
fn token_totals(usage: &[ProviderUsage]) -> (i64, i64) {
    (
        usage.iter().filter_map(|u| u.usage.input_tokens).sum(),
        usage.iter().filter_map(|u| u.usage.output_tokens).sum(),
    )
}

// ---- command line ----

const USAGE: &str = "usage: goblin-gdk --task TEXT|--task-file PATH --repo PATH [--base-url URL] [--model ID] \
[--budget N] [--no-packet] [--max-agent-iterations N --verifier PATH ...]\n       goblin-gdk --acp [--repo PATH] \
[--agent-name NAME] [--audit-log PATH] [--history-turns N] [--base-url URL] [--model ID] [--budget N] [--no-packet] \
[--max-agent-iterations N --verifier PATH ...]";

struct Args {
    repo: String,
    task: String,
    base_url: String,
    model: String,
    budget: u64,
    max_tool_calls: u64,
    arm: String,
    python: String,
    goblin_cwd: String,
    no_packet: bool,
    max_tokens: Option<i32>,
    timeout_secs: u64,
    session: session::Options,
    acp: bool,
    agent_name: String,
    audit_log: Option<std::path::PathBuf>,
    history_turns: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            repo: String::new(),
            task: String::new(),
            base_url: "http://127.0.0.1:8093/v1/".into(),
            model: "qwen2.5-0.5b-instruct".into(),
            budget: 2048,
            max_tool_calls: 24,
            arm: "hybrid".into(),
            python: "python3".into(),
            goblin_cwd: ".".into(),
            no_packet: false,
            max_tokens: None,
            timeout_secs: 600,
            session: session::Options::default(),
            acp: false,
            agent_name: "pocket-goblin".into(),
            audit_log: None,
            history_turns: 6,
        }
    }
}

fn parse_args(argv: impl IntoIterator<Item = String>) -> Result<Args> {
    let mut args = Args::default();
    let mut acp_only = false;
    let mut it = argv.into_iter();
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| anyhow!("{flag} needs a value"));
        match flag.as_str() {
            "--repo" => args.repo = value()?,
            "--task" => args.task = value()?,
            "--task-file" => args.task = std::fs::read_to_string(value()?)?,
            "--base-url" => args.base_url = value()?,
            "--model" => args.model = value()?,
            "--budget" => args.budget = value()?.parse()?,
            "--max-tool-calls" => args.max_tool_calls = value()?.parse()?,
            "--arm" => args.arm = value()?,
            "--python" => args.python = value()?,
            "--goblin-cwd" => args.goblin_cwd = value()?,
            "--no-packet" => args.no_packet = true,
            "--max-tokens" => args.max_tokens = Some(value()?.parse()?),
            "--timeout-secs" => args.timeout_secs = value()?.parse()?,
            "--acp" => args.acp = true,
            "--agent-name" => (args.agent_name, acp_only) = (value()?, true),
            "--audit-log" => (args.audit_log, acp_only) = (Some(value()?.into()), true),
            "--history-turns" => (args.history_turns, acp_only) = (value()?.parse()?, true),
            other => args.session.parse(other, &mut value).map_err(|error| anyhow!("{error}\n{USAGE}"))?,
        }
    }
    if args.session.max_iterations.is_some() {
        args.session.settings(&args)?;
    } else {
        args.session.check_legacy()?;
    }
    if args.acp {
        if !args.task.is_empty() {
            bail!("--acp reads each task from session/prompt; drop --task and --task-file");
        }
        args.session.check_acp()?;
    } else if acp_only {
        bail!("--agent-name, --audit-log and --history-turns need --acp");
    } else if args.task.is_empty() || (!args.no_packet && args.repo.is_empty()) {
        bail!("{USAGE}");
    }
    if !args.base_url.ends_with('/') {
        args.base_url.push('/');
    }
    Ok(args)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse_args(std::env::args().skip(1))?;
    if args.acp {
        return acp::serve(args).await;
    }
    if args.session.max_iterations.is_some() {
        let receipt = session::run(&args, &args.task, &args.repo, None, CancellationToken::new()).await?;
        println!("{}", serde_json::to_string_pretty(&receipt)?);
        return Ok(());
    }
    let started = Instant::now();
    let (packet, packet_stats) = if args.no_packet {
        (String::new(), Value::Null)
    } else {
        let mut goblin = McpClient::spawn(&args.python, &["-m", "MCP.mcp03.goblin_server"], &args.goblin_cwd).await?;
        fetch_packet(&mut goblin, &args, &args.task, &args.repo).await?
    };
    let retrieval_seconds = started.elapsed().as_secs_f64();

    let inference_started = Instant::now();
    let packet_chars = packet.len();
    let conversation = Conversation::new_unvalidated([Message::user().with_text(args.task.clone())]);
    let (conversation, usage) = infer(&args, conversation, packet, CancellationToken::new()).await?;
    let answer = conversation.last().map(|m| m.as_concat_text()).unwrap_or_default();
    let (input_tokens, output_tokens) = token_totals(&usage);
    let receipt = json!({
        "app": "goblin-gdk", "gdk": "goose-agent 0.1.0-alpha.14", "model": args.model,
        "used_packet": !args.no_packet, "packet_stats": packet_stats, "packet_chars": packet_chars,
        "provider_input_tokens": input_tokens, "provider_output_tokens": output_tokens,
        "inference_calls": usage.len(), "max_tokens": args.max_tokens, "timeout_secs": args.timeout_secs,
        "retrieval_seconds": retrieval_seconds,
        "inference_seconds": inference_started.elapsed().as_secs_f64(), "answer": answer,
    });
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(argv: &[&str]) -> Result<Args> {
        parse_args(argv.iter().map(|arg| arg.to_string()))
    }

    #[test]
    fn acp_mode_needs_no_task_and_takes_its_own_flags() {
        let args = parse(&["--acp", "--agent-name", "goblin", "--history-turns", "3", "--audit-log", "audit.jsonl"]).unwrap();
        assert!(args.acp && args.repo.is_empty() && args.task.is_empty());
        assert_eq!((args.agent_name.as_str(), args.history_turns), ("goblin", 3));
        assert_eq!(args.audit_log.as_deref(), Some(std::path::Path::new("audit.jsonl")));
    }

    #[test]
    fn acp_mode_rejects_a_fixed_task() {
        assert!(parse(&["--acp", "--task", "fix it"]).is_err());
    }

    #[test]
    fn acp_flags_without_acp_are_rejected() {
        assert!(parse(&["--task", "fix it", "--repo", ".", "--audit-log", "a.jsonl"]).is_err());
        assert!(parse(&["--task", "fix it", "--repo", ".", "--history-turns", "2"]).is_err());
    }

    #[test]
    fn acp_mode_takes_the_repair_session_flags() {
        let args = parse(&["--acp", "--max-agent-iterations", "8", "--verifier", "v.json", "--memory-mode", "typed"]).unwrap();
        assert!(args.acp && args.session.max_iterations == Some(8));
        // Validated at startup, not at the first prompt.
        assert!(parse(&["--acp", "--max-agent-iterations", "0", "--verifier", "v.json"]).is_err());
        assert!(parse(&["--acp", "--max-agent-iterations", "8", "--memory-mode", "full"]).is_err());
        // Repair flags still need an iteration budget, and one checkpoint file cannot serve ACP sessions.
        assert!(parse(&["--acp", "--verifier", "v.json"]).is_err());
        assert!(parse(&["--acp", "--max-agent-iterations", "8", "--session-file", "s.json"]).is_err());
        assert!(parse(&["--task", "t", "--repo", ".", "--max-agent-iterations", "8", "--session-file", "s.json"]).is_ok());
    }

    #[test]
    fn one_shot_mode_still_needs_a_task_and_a_repo() {
        assert!(parse(&["--repo", "."]).is_err());
        assert!(parse(&["--task", "fix it"]).is_err());
        let args = parse(&["--task", "fix it", "--no-packet", "--base-url", "http://h:1/v1"]).unwrap();
        assert_eq!(args.base_url, "http://h:1/v1/");
    }
}
