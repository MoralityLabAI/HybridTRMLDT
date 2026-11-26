//! Titan port: Pocket Goblin as an Agent Client Protocol (ACP) agent.
//!
//! TitanAgent, Blake Hung's agent squad, runs every agent as an ACP command behind OpenAB. OpenAB
//! spawns the command, speaks newline-delimited JSON-RPC 2.0 (ACP protocol version 1) on its stdio,
//! and relays prompts and replies to Discord, Slack or LINE. `goblin-gdk --acp` is that command.
//!
//! - Requests served: `initialize`, `authenticate`, `session/new`, `session/prompt`, and
//!   `session/close` (alias `session/end`). The `session/cancel` notification stops a running turn.
//!   Any other request gets JSON-RPC error -32601.
//! - Each `session/prompt` runs one GDK turn over the session's conversation: the Data Goblin
//!   compact packet for the session repository (the `session/new` cwd unless `--repo` pins one),
//!   reported as a `search` tool call, then inference, reported as one `agent_message_chunk`. The
//!   result carries `stopReason`, provider `usage`, and the turn receipt under `_meta.pocketGoblin`.
//! - Sessions keep their last `--history-turns` completed turns. A cancelled turn is dropped.
//! - `--audit-log PATH` appends one JSON line per retrieval and inference (timestamp, agent,
//!   session, op, path or model, status), the shape of a TitanAgent shared audit log.
//! - With the repair-session flags (`--max-agent-iterations`, `--verifier`, ...) each
//!   `session/prompt` runs one Pocket Goblin repair session (`session.rs`) over the session
//!   repository instead. Every repair tool call is a `tool_call`/`tool_call_update` pair. Each
//!   write (`apply_edit`, `create_file`) first sends `session/request_permission`, TitanAgent's
//!   approval gate, which OpenAB answers; a rejected write becomes a tool error. The audit log gets
//!   an `op: write` line with the edited path per write, an `op: verify` line per verifier run and
//!   one `op: repair` line per turn.
//! - Errors carry a stable `data.category`. Stdout carries protocol frames only; the goblin
//!   server's stderr passes through to ours.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use goose_provider_types::conversation::message::Message;
use goose_provider_types::conversation::Conversation;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{mpsc, oneshot, Mutex as AsyncMutex};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use crate::session::{self as repair, tool_kind, Observer, ToolCall};
use crate::{fetch_packet, infer, token_totals, Args, McpClient};

const PROTOCOL_VERSION: u64 = 1;
/// The permission options offered for each write. There is no `allow_always`: every edit asks.
const ALLOW: &str = "allow-once";
const REJECT: &str = "reject-once";

/// A JSON-RPC error with a stable `data.category` that clients can branch on.
#[derive(Debug)]
struct RpcError {
    code: i64,
    message: String,
    category: &'static str,
}

impl RpcError {
    fn new(code: i64, message: impl Into<String>, category: &'static str) -> Self {
        Self { code, message: message.into(), category }
    }

    fn invalid(message: impl Into<String>) -> Self {
        Self::new(-32602, message, "invalid_params")
    }

    fn internal(message: impl Into<String>, category: &'static str) -> Self {
        Self::new(-32603, message, category)
    }
}

struct Session {
    repo: String,
    /// Completed turns, oldest first: the user message and what the machine appended after it.
    turns: Vec<Vec<Message>>,
    prompts: u64,
}

/// A finished turn, committed and reported only if it was not cancelled.
struct Turn {
    messages: Vec<Message>,
    answer: String,
    input_tokens: i64,
    output_tokens: i64,
    receipt: Value,
}

/// Our end of the stdio connection: frames to stdout, and the requests this agent sent
/// (`session/request_permission`) that await the client's answer.
struct Client {
    out: mpsc::UnboundedSender<Value>,
    next_id: AtomicU64,
    pending: Mutex<HashMap<u64, oneshot::Sender<Value>>>,
}

impl Client {
    fn new(out: mpsc::UnboundedSender<Value>) -> Self {
        Self { out, next_id: AtomicU64::new(0), pending: Mutex::default() }
    }

    fn send(&self, frame: Value) {
        let _ = self.out.send(frame);
    }

    fn update(&self, session: &str, update: Value) {
        self.send(json!({"jsonrpc": "2.0", "method": "session/update",
                         "params": {"sessionId": session, "update": update}}));
    }

    /// Send a request and wait for the client's result. An error response or a closed connection
    /// is an `Err`; dropping the future (a cancelled turn) abandons the request.
    async fn request(&self, method: &str, params: Value) -> Result<Value, String> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        let (tx, rx) = oneshot::channel();
        self.pending.lock().unwrap().insert(id, tx);
        let _forget = Forget { client: self, id };
        self.send(json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        let reply = rx.await.map_err(|_| "the client connection closed".to_owned())?;
        match reply.get("error") {
            Some(error) => Err(format!("the client answered {method} with an error: {error}")),
            None => Ok(reply.get("result").cloned().unwrap_or(Value::Null)),
        }
    }

    /// Hand a response frame to the request awaiting it. A response to nothing pending is dropped.
    fn resolve(&self, frame: Value) {
        let waiting = frame.get("id").and_then(Value::as_u64).and_then(|id| self.pending.lock().unwrap().remove(&id));
        if let Some(waiting) = waiting {
            let _ = waiting.send(frame);
        }
    }

    /// The connection closed: every request still waiting fails.
    fn close(&self) {
        self.pending.lock().unwrap().clear();
    }
}

/// Removes a request's pending entry when its wait ends, answered or abandoned.
struct Forget<'a> {
    client: &'a Client,
    id: u64,
}

impl Drop for Forget<'_> {
    fn drop(&mut self) {
        self.client.pending.lock().unwrap().remove(&self.id);
    }
}

/// The squad's shared audit log: one JSON line per operation, tagged with time, agent and session.
#[derive(Clone)]
struct AuditLog {
    path: Option<PathBuf>,
    agent: String,
}

impl AuditLog {
    fn write(&self, session: &str, entry: Value) -> Result<(), RpcError> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |elapsed| elapsed.as_secs_f64());
        let mut line = json!({"ts": ts, "agent": self.agent, "session": session});
        if let (Some(line), Value::Object(entry)) = (line.as_object_mut(), entry) {
            line.extend(entry);
        }
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .and_then(|mut file| file.write_all(format!("{line}\n").as_bytes()))
            .map_err(|error| RpcError::internal(format!("audit log {}: {error}", path.display()), "audit_log_error"))
    }
}

struct Agent {
    args: Args,
    client: Arc<Client>,
    log: AuditLog,
    /// One Data Goblin server per agent process, shared by its sessions.
    goblin: AsyncMutex<Option<McpClient>>,
    sessions: Mutex<HashMap<String, Arc<AsyncMutex<Session>>>>,
    cancels: Mutex<HashMap<String, CancellationToken>>,
}

pub(crate) async fn serve(args: Args) -> anyhow::Result<()> {
    let (out, mut frames) = mpsc::unbounded_channel::<Value>();
    let writer = tokio::spawn(async move {
        let mut stdout = tokio::io::stdout();
        while let Some(frame) = frames.recv().await {
            let mut line = frame.to_string().into_bytes();
            line.push(b'\n');
            if stdout.write_all(&line).await.is_err() || stdout.flush().await.is_err() {
                break;
            }
        }
    });
    let agent = Arc::new(Agent::new(args, out));
    let mut prompts = JoinSet::new();
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    while let Some(line) = lines.next_line().await? {
        while prompts.try_join_next().is_some() {}
        if line.trim().is_empty() {
            continue;
        }
        let message: Value = match serde_json::from_str(&line) {
            Ok(message) => message,
            Err(error) => {
                agent.reply(Value::Null, Err(RpcError::new(-32700, format!("parse error: {error}"), "parse_error")));
                continue;
            }
        };
        let params = message.get("params").cloned().unwrap_or(Value::Null);
        match (message.get("method").and_then(Value::as_str), message.get("id").cloned()) {
            (Some("session/prompt"), Some(id)) => {
                let agent = agent.clone();
                prompts.spawn(async move {
                    let result = agent.prompt(&params).await;
                    agent.reply(id, result);
                });
            }
            (Some(method), Some(id)) => agent.reply(id, agent.handle(method, &params)),
            (Some("session/cancel"), None) => agent.cancel(&params),
            // A response to one of our requests (a permission answer).
            (None, Some(_)) => agent.client.resolve(message.clone()),
            // Other notifications.
            _ => {}
        }
    }
    // The client closed our stdin: stop running turns, let them answer, then flush and exit.
    for token in agent.cancels.lock().unwrap().values() {
        token.cancel();
    }
    agent.client.close();
    while prompts.join_next().await.is_some() {}
    drop(agent);
    writer.await?;
    Ok(())
}

impl Agent {
    fn new(args: Args, out: mpsc::UnboundedSender<Value>) -> Self {
        let log = AuditLog { path: args.audit_log.clone(), agent: args.agent_name.clone() };
        Self {
            args,
            client: Arc::new(Client::new(out)),
            log,
            goblin: AsyncMutex::new(None),
            sessions: Mutex::default(),
            cancels: Mutex::default(),
        }
    }

    fn reply(&self, id: Value, result: Result<Value, RpcError>) {
        let frame = match result {
            Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
            Err(error) => json!({"jsonrpc": "2.0", "id": id, "error": {
                "code": error.code, "message": error.message, "data": {"category": error.category}}}),
        };
        self.client.send(frame);
    }

    fn update(&self, session: &str, update: Value) {
        self.client.update(session, update);
    }

    fn repairs(&self) -> bool {
        self.args.session.max_iterations.is_some()
    }

    fn handle(&self, method: &str, params: &Value) -> Result<Value, RpcError> {
        match method {
            "initialize" => Ok(json!({
                "protocolVersion": PROTOCOL_VERSION,
                "agentCapabilities": {
                    "loadSession": false,
                    "promptCapabilities": {"image": false, "audio": false, "embeddedContext": true},
                    "mcpCapabilities": {"http": false, "sse": false},
                },
                "authMethods": [],
                "agentInfo": {"name": self.args.agent_name, "title": "Pocket Goblin",
                              "version": env!("CARGO_PKG_VERSION")},
            })),
            "authenticate" => Ok(json!({})),
            "session/new" => self.new_session(params),
            "session/close" | "session/end" => {
                let id = session_id(params)?;
                if let Some(token) = self.cancels.lock().unwrap().get(id) {
                    token.cancel();
                }
                self.sessions.lock().unwrap().remove(id).map(|_| json!({})).ok_or_else(|| unknown_session(id))
            }
            _ => Err(RpcError::new(-32601, format!("method not found: {method}"), "method_not_found")),
        }
    }

    fn new_session(&self, params: &Value) -> Result<Value, RpcError> {
        let cwd = params
            .get("cwd")
            .and_then(Value::as_str)
            .filter(|cwd| Path::new(cwd).is_absolute())
            .ok_or_else(|| RpcError::invalid("session/new needs an absolute cwd"))?;
        let repo = if self.args.repo.is_empty() { cwd } else { self.args.repo.as_str() };
        if (!self.args.no_packet || self.repairs()) && !Path::new(repo).is_dir() {
            return Err(RpcError::invalid(format!("session repository {repo} is not a directory")));
        }
        if params.get("mcpServers").and_then(Value::as_array).is_some_and(|servers| !servers.is_empty()) {
            eprintln!("{}: client MCP servers are not attached; context comes from the Data Goblin server",
                      self.args.agent_name);
        }
        let id = format!("{}-{}", self.args.agent_name, uuid::Uuid::new_v4());
        let session = Session { repo: repo.to_owned(), turns: Vec::new(), prompts: 0 };
        self.sessions.lock().unwrap().insert(id.clone(), Arc::new(AsyncMutex::new(session)));
        Ok(json!({"sessionId": id}))
    }

    fn cancel(&self, params: &Value) {
        if let Ok(id) = session_id(params) {
            if let Some(token) = self.cancels.lock().unwrap().get(id) {
                token.cancel();
            }
        }
    }

    async fn prompt(&self, params: &Value) -> Result<Value, RpcError> {
        let id = session_id(params)?.to_owned();
        let task = prompt_text(params.get("prompt"))
            .ok_or_else(|| RpcError::invalid("session/prompt needs a non-empty text prompt"))?;
        let session = self.sessions.lock().unwrap().get(&id).cloned().ok_or_else(|| unknown_session(&id))?;
        let mut session = session.lock().await;
        let cancel = CancellationToken::new();
        self.cancels.lock().unwrap().insert(id.clone(), cancel.clone());
        let outcome = if self.repairs() {
            self.repair_turn(&id, &mut session, task, &cancel).await
        } else {
            self.turn(&id, &mut session, task, &cancel).await
        };
        // Once the token leaves the map no cancel can land, so the stop reason below is final.
        self.cancels.lock().unwrap().remove(&id);
        if cancel.is_cancelled() {
            return Ok(json!({"stopReason": "cancelled"}));
        }
        let turn = outcome?;
        if !turn.messages.is_empty() {
            session.turns.push(turn.messages);
            let excess = session.turns.len().saturating_sub(self.args.history_turns);
            session.turns.drain(..excess);
        }
        if !turn.answer.is_empty() {
            self.update(&id, json!({"sessionUpdate": "agent_message_chunk",
                                    "content": {"type": "text", "text": turn.answer}}));
        }
        Ok(json!({
            "stopReason": "end_turn",
            "usage": {"inputTokens": turn.input_tokens, "outputTokens": turn.output_tokens,
                      "totalTokens": turn.input_tokens + turn.output_tokens},
            "_meta": {"pocketGoblin": turn.receipt},
        }))
    }

    async fn turn(&self, id: &str, session: &mut Session, task: String, cancel: &CancellationToken)
        -> Result<Turn, RpcError> {
        let args = &self.args;
        session.prompts += 1;
        let started = Instant::now();
        let (packet, packet_stats) = if args.no_packet {
            (String::new(), Value::Null)
        } else {
            let call = format!("{id}-packet-{}", session.prompts);
            self.update(id, json!({
                "sessionUpdate": "tool_call", "toolCallId": call, "kind": "search", "status": "in_progress",
                "title": format!("Data Goblin packet ({} token budget)", args.budget),
                "rawInput": {"repo_path": session.repo, "budget_tokens": args.budget,
                             "max_tool_calls": args.max_tool_calls, "arm": args.arm},
            }));
            let fetched = tokio::select! {
                fetched = self.retrieve(&task, &session.repo) => fetched,
                _ = cancel.cancelled() => Err(anyhow::anyhow!("cancelled by the client")),
            };
            match fetched {
                Ok((packet, stats)) => {
                    self.update(id, json!({
                        "sessionUpdate": "tool_call_update", "toolCallId": call, "status": "completed",
                        "content": [{"type": "content", "content": {"type": "text", "text": packet_summary(&stats)}}],
                        "rawOutput": stats,
                    }));
                    self.audit(id, json!({"op": "retrieve", "path": session.repo, "status": "ok",
                                          "packet_tokens": stats["packet_tokens"], "tool_calls": stats["tool_calls"],
                                          "router": stats["router"]}))?;
                    (packet, stats)
                }
                Err(error) => {
                    // A cancelled or failed call can leave the server mid-reply; the next turn starts a fresh one.
                    *self.goblin.lock().await = None;
                    let message = format!("{error:#}");
                    self.update(id, json!({
                        "sessionUpdate": "tool_call_update", "toolCallId": call, "status": "failed",
                        "content": [{"type": "content", "content": {"type": "text", "text": message}}],
                    }));
                    let status = if cancel.is_cancelled() { "cancelled" } else { "failed" };
                    self.audit(id, json!({"op": "retrieve", "path": session.repo, "status": status, "error": message}))?;
                    return Err(RpcError::internal(format!("Data Goblin retrieval failed: {message}"), "retrieval_error"));
                }
            }
        };
        let retrieval_seconds = started.elapsed().as_secs_f64();

        let mut messages: Vec<Message> = session.turns.iter().flatten().cloned().collect();
        let prior = messages.len();
        messages.push(Message::user().with_text(task));
        let packet_chars = packet.len();
        let inference_started = Instant::now();
        let (conversation, usage) =
            match infer(args, Conversation::new_unvalidated(messages), packet, cancel.clone()).await {
                Ok(done) => done,
                Err(error) => {
                    let message = format!("{error:#}");
                    self.audit(id, json!({"op": "infer", "model": args.model, "status": "failed", "error": message}))?;
                    return Err(RpcError::internal(format!("inference failed: {message}"), "provider_error"));
                }
            };
        let (input_tokens, output_tokens) = token_totals(&usage);
        let status = if cancel.is_cancelled() { "cancelled" } else { "ok" };
        self.audit(id, json!({"op": "infer", "model": args.model, "status": status,
                              "input_tokens": input_tokens, "output_tokens": output_tokens}))?;
        let messages: Vec<Message> = conversation.iter().skip(prior).cloned().collect();
        let answer = messages
            .iter()
            .skip(1)
            .map(Message::as_concat_text)
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
        let receipt = json!({
            "app": "goblin-gdk", "gdk": "goose-agent 0.1.0-alpha.14", "model": args.model,
            "used_packet": !args.no_packet, "packet_stats": packet_stats, "packet_chars": packet_chars,
            "history_messages": prior, "provider_input_tokens": input_tokens,
            "provider_output_tokens": output_tokens, "inference_calls": usage.len(),
            "retrieval_seconds": retrieval_seconds,
            "inference_seconds": inference_started.elapsed().as_secs_f64(),
        });
        Ok(Turn { messages, answer, input_tokens, output_tokens, receipt })
    }

    /// One repair session for `task` over the session repository, reported to the client while it
    /// runs. The session's own conversation is multi-turn; ACP prompts do not share history.
    async fn repair_turn(&self, id: &str, session: &mut Session, task: String, cancel: &CancellationToken)
        -> Result<Turn, RpcError> {
        session.prompts += 1;
        let observer = Arc::new(RepairObserver {
            client: self.client.clone(),
            log: self.log.clone(),
            session: id.to_owned(),
            calls: format!("{id}-repair-{}", session.prompts),
            workspace: std::path::absolute(&session.repo).unwrap_or_else(|_| PathBuf::from(&session.repo)),
            open: Mutex::new(None),
            audit_error: Mutex::new(None),
        });
        let run = repair::run(&self.args, &task, &session.repo, Some(observer.clone()), cancel.clone());
        // Dropping the run on cancel stops a tool call in flight too: the repair service and the
        // preflight relay are killed with it.
        let outcome = tokio::select! {
            outcome = run => outcome,
            _ = cancel.cancelled() => Err(anyhow::anyhow!("cancelled by the client")),
        };
        if cancel.is_cancelled() {
            observer.cancel_open();
            self.audit(id, json!({"op": "repair", "model": self.args.model, "status": "cancelled"}))?;
            return Err(RpcError::internal("cancelled by the client", "cancelled"));
        }
        let receipt = match outcome {
            Ok(receipt) => receipt,
            Err(error) => {
                let message = format!("{error:#}");
                self.audit(id, json!({"op": "repair", "model": self.args.model, "status": "failed", "error": message}))?;
                return Err(RpcError::internal(format!("repair session failed: {message}"), "repair_error"));
            }
        };
        if let Some(message) = observer.audit_error.lock().unwrap().take() {
            return Err(RpcError::internal(message, "audit_log_error"));
        }
        let input_tokens = receipt["provider_input_tokens"].as_i64().unwrap_or(0);
        let output_tokens = receipt["provider_output_tokens"].as_i64().unwrap_or(0);
        let error = receipt["error"].as_str().map(str::to_owned);
        self.audit(id, json!({
            "op": "repair", "model": self.args.model, "status": if error.is_some() { "failed" } else { "ok" },
            "verified": receipt["state"]["verified"], "input_tokens": input_tokens, "output_tokens": output_tokens,
            "inference_calls": receipt["inference_calls"], "tool_calls": receipt["session"]["tool_calls"],
        }))?;
        if let Some(error) = error {
            return Err(RpcError::internal(format!("repair session failed: {error}"), "repair_error"));
        }
        let answer = receipt["answer"].as_str().unwrap_or_default().to_owned();
        Ok(Turn { messages: Vec::new(), answer, input_tokens, output_tokens, receipt })
    }

    async fn retrieve(&self, task: &str, repo: &str) -> anyhow::Result<(String, Value)> {
        let mut goblin = self.goblin.lock().await;
        if goblin.is_none() {
            let server = McpClient::spawn(&self.args.python, &["-m", "MCP.mcp03.goblin_server"], &self.args.goblin_cwd);
            *goblin = Some(server.await?);
        }
        fetch_packet(goblin.as_mut().unwrap(), &self.args, task, repo).await
    }

    fn audit(&self, session: &str, entry: Value) -> Result<(), RpcError> {
        self.log.write(session, entry)
    }
}

/// Reports one repair turn: each tool call to the client and the writes and verifier runs to the
/// audit log, and asks the client to approve each write.
struct RepairObserver {
    client: Arc<Client>,
    log: AuditLog,
    session: String,
    /// The prefix of this turn's tool call ids.
    calls: String,
    /// The session repository as an absolute path, for ACP `locations` and diffs.
    workspace: PathBuf,
    /// The call in flight, which a cancelled turn closes.
    open: Mutex<Option<ToolCall>>,
    /// The first audit-log failure. The turn fails with it once the run returns.
    audit_error: Mutex<Option<String>>,
}

impl RepairObserver {
    fn call_id(&self, call: &ToolCall) -> String {
        format!("{}-{}", self.calls, call.index)
    }

    /// The ACP description of a call: id, title, kind, raw input, and for a write its file and diff.
    fn describe(&self, call: &ToolCall) -> Value {
        let mut fields = json!({"toolCallId": self.call_id(call), "title": title(call),
                                "kind": tool_kind(&call.tool), "rawInput": call.arguments});
        if let Some(path) = &call.path {
            let path = self.workspace.join(path).display().to_string();
            let (old, new) = match call.tool.as_str() {
                "apply_edit" => (call.arguments["old"].clone(), call.arguments["new"].clone()),
                _ => (Value::Null, call.arguments["content"].clone()),
            };
            fields["locations"] = json!([{"path": path}]);
            fields["content"] = json!([{"type": "diff", "path": path, "oldText": old, "newText": new}]);
        }
        fields
    }

    fn record(&self, entry: Value) {
        if let Err(error) = self.log.write(&self.session, entry) {
            self.audit_error.lock().unwrap().get_or_insert(error.message);
        }
    }

    /// The turn was cancelled with a call in flight: close it for the client and the audit log.
    fn cancel_open(&self) {
        let Some(call) = self.open.lock().unwrap().take() else {
            return;
        };
        self.client.update(&self.session, json!({
            "sessionUpdate": "tool_call_update", "toolCallId": self.call_id(&call), "status": "failed",
            "content": text_content("cancelled by the client"),
        }));
        if call.is_write() {
            self.record(json!({"op": "write", "tool": call.tool, "path": call.path, "status": "cancelled",
                               "origin": call.origin}));
        }
    }
}

#[async_trait]
impl Observer for RepairObserver {
    fn started(&self, call: &ToolCall) {
        let mut update = self.describe(call);
        update["sessionUpdate"] = json!("tool_call");
        update["status"] = json!(if call.is_write() { "pending" } else { "in_progress" });
        self.client.update(&self.session, update);
        *self.open.lock().unwrap() = Some(call.clone());
    }

    async fn permit(&self, call: &ToolCall) -> bool {
        let mut tool_call = self.describe(call);
        tool_call["status"] = json!("pending");
        let params = json!({
            "sessionId": self.session, "toolCall": tool_call,
            "options": [
                {"optionId": ALLOW, "name": "Allow this edit", "kind": "allow_once"},
                {"optionId": REJECT, "name": "Reject this edit", "kind": "reject_once"},
            ],
        });
        match self.client.request("session/request_permission", params).await {
            Ok(result) => result["outcome"]["outcome"] == "selected" && result["outcome"]["optionId"] == ALLOW,
            Err(error) => {
                eprintln!("{}: write not approved: {error}", self.log.agent);
                false
            }
        }
    }

    fn finished(&self, call: &ToolCall, output: &Value, error: bool) {
        self.open.lock().unwrap().take();
        self.client.update(&self.session, json!({
            "sessionUpdate": "tool_call_update", "toolCallId": self.call_id(call),
            "status": if error { "failed" } else { "completed" },
            "content": text_content(&summary(call, output, error)), "rawOutput": output,
        }));
        if call.is_write() {
            let status = match (call.permission, error) {
                (Some(false), _) => "rejected",
                (_, true) => "failed",
                _ => "ok",
            };
            let mut entry = json!({"op": "write", "tool": call.tool, "status": status, "origin": call.origin,
                                   "path": output["path"].as_str().or(call.path.as_deref())});
            if error {
                entry["error"] = output["error"].clone();
            } else {
                entry["before"] = output["before"].clone();
                entry["after"] = output["after"].clone();
            }
            self.record(entry);
        } else if call.tool == "run_verifier" {
            self.record(json!({"op": "verify", "status": if error { "failed" } else { "ok" },
                               "origin": call.origin, "passed": output["passed"], "exit_code": output["exit_code"]}));
        }
    }
}

fn title(call: &ToolCall) -> String {
    let argument = |key: &str| call.arguments[key].as_str().unwrap_or("?");
    match call.tool.as_str() {
        "context_query" => format!("Search: {}", argument("query")),
        "context_expand" => format!("Expand {}", argument("node_id")),
        "context_read" => format!("Read {}", call.arguments["uri"].as_str().unwrap_or_else(|| argument("handle"))),
        "context_state" => "Inspect the context state".into(),
        "apply_edit" => format!("Edit {}", call.path.as_deref().unwrap_or("source")),
        "create_file" => format!("Create {}", call.path.as_deref().unwrap_or("file")),
        "inspect_edit" => "Inspect the patch".into(),
        "run_verifier" => "Run the native verifier".into(),
        other => other.to_owned(),
    }
}

fn summary(call: &ToolCall, output: &Value, error: bool) -> String {
    if error {
        return output["error"].as_str().unwrap_or("failed").to_owned();
    }
    match call.tool.as_str() {
        "run_verifier" => format!("the native verifier {} (exit code {})",
                                  if output["passed"] == true { "passed" } else { "failed" }, output["exit_code"]),
        "apply_edit" | "create_file" => format!("wrote {}", output["path"].as_str().unwrap_or("?")),
        "context_read" => format!("read {}", output["path"].as_str().unwrap_or("?")),
        _ => output.to_string().chars().take(300).collect(),
    }
}

fn text_content(text: &str) -> Value {
    json!([{"type": "content", "content": {"type": "text", "text": text}}])
}

fn session_id(params: &Value) -> Result<&str, RpcError> {
    params.get("sessionId").and_then(Value::as_str).ok_or_else(|| RpcError::invalid("missing sessionId"))
}

fn unknown_session(id: &str) -> RpcError {
    RpcError::new(-32602, format!("unknown session {id}"), "session_not_found")
}

/// The task text of an ACP prompt: text blocks, embedded text resources (preceded by their URI)
/// and resource links, in order. `None` when nothing textual remains.
fn prompt_text(prompt: Option<&Value>) -> Option<String> {
    let parts: Vec<String> = prompt?
        .as_array()?
        .iter()
        .filter_map(|block| match block.get("type")?.as_str()? {
            "text" => block.get("text")?.as_str().map(str::to_owned),
            "resource" => {
                let resource = block.get("resource")?;
                let text = resource.get("text")?.as_str()?;
                Some(match resource.get("uri").and_then(Value::as_str) {
                    Some(uri) => format!("{uri}\n{text}"),
                    None => text.to_owned(),
                })
            }
            "resource_link" => {
                let uri = block.get("uri")?.as_str()?;
                Some(format!("[{}]({uri})", block.get("name").and_then(Value::as_str).unwrap_or(uri)))
            }
            _ => None,
        })
        .collect();
    let text = parts.join("\n\n");
    (!text.trim().is_empty()).then_some(text)
}

fn packet_summary(stats: &Value) -> String {
    format!(
        "{} packet tokens from {} tool calls; the {} router read {} tokens locally",
        stats["packet_tokens"],
        stats["tool_calls"],
        stats["router"].as_str().unwrap_or("goblin"),
        stats["local_result_tokens"],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent(args: Args) -> (Agent, mpsc::UnboundedReceiver<Value>) {
        let (out, frames) = mpsc::unbounded_channel();
        (Agent::new(args, out), frames)
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("pocket-goblin-{name}-{}", uuid::Uuid::new_v4()))
    }

    #[test]
    fn prompt_text_keeps_text_resources_and_links_in_order() {
        let prompt = json!([
            {"type": "text", "text": "Why does refine drop unknowns?"},
            {"type": "image", "data": "AAAA", "mimeType": "image/png"},
            {"type": "resource", "resource": {"uri": "file:///repo/env.py", "text": "def refine(): ..."}},
            {"type": "resource_link", "uri": "file:///repo/world.py", "name": "world.py"},
        ]);
        assert_eq!(
            prompt_text(Some(&prompt)).unwrap(),
            "Why does refine drop unknowns?\n\nfile:///repo/env.py\ndef refine(): ...\n\n[world.py](file:///repo/world.py)"
        );
    }

    #[test]
    fn prompt_text_rejects_blank_media_only_and_malformed_prompts() {
        assert_eq!(prompt_text(Some(&json!([{"type": "text", "text": "  \n"}]))), None);
        assert_eq!(prompt_text(Some(&json!([{"type": "image", "data": "AAAA"}]))), None);
        assert_eq!(prompt_text(Some(&json!("not a block list"))), None);
        assert_eq!(prompt_text(None), None);
    }

    #[test]
    fn initialize_advertises_protocol_one_without_session_loading() {
        let (agent, _frames) = agent(Args { agent_name: "goblin".into(), ..Args::default() });
        let result = agent.handle("initialize", &json!({"protocolVersion": 1, "clientCapabilities": {}})).unwrap();
        assert_eq!(result["protocolVersion"], 1);
        assert_eq!(result["agentCapabilities"]["loadSession"], false);
        assert_eq!(result["agentInfo"]["name"], "goblin");
        assert_eq!(result["authMethods"], json!([]));
    }

    #[test]
    fn session_new_needs_an_absolute_existing_directory() {
        let (agent, _frames) = agent(Args::default());
        let relative = agent.handle("session/new", &json!({"cwd": "relative/dir", "mcpServers": []}));
        assert_eq!(relative.unwrap_err().category, "invalid_params");
        let missing = temp_path("missing");
        let missing = agent.handle("session/new", &json!({"cwd": missing.to_str().unwrap(), "mcpServers": []}));
        assert_eq!(missing.unwrap_err().code, -32602);
        let cwd = std::env::temp_dir();
        let created = agent.handle("session/new", &json!({"cwd": cwd.to_str().unwrap(), "mcpServers": []})).unwrap();
        let id = created["sessionId"].as_str().unwrap();
        assert!(id.starts_with("pocket-goblin-"));
        assert_eq!(agent.sessions.lock().unwrap()[id].try_lock().unwrap().repo, cwd.to_str().unwrap());
    }

    #[test]
    fn pinned_repo_overrides_the_session_cwd() {
        let repo = std::env::temp_dir();
        let (agent, _frames) = agent(Args { repo: repo.to_str().unwrap().into(), ..Args::default() });
        let cwd = temp_path("unused-cwd");
        let created = agent.handle("session/new", &json!({"cwd": cwd.to_str().unwrap()})).unwrap();
        let sessions = agent.sessions.lock().unwrap();
        let session = sessions[created["sessionId"].as_str().unwrap()].try_lock().unwrap();
        assert_eq!(session.repo, repo.to_str().unwrap());
    }

    #[test]
    fn unsupported_methods_and_unknown_sessions_are_typed_errors() {
        let (agent, _frames) = agent(Args::default());
        let load = agent.handle("session/load", &json!({"sessionId": "x", "cwd": "/", "mcpServers": []}));
        assert_eq!(load.unwrap_err().code, -32601);
        let close = agent.handle("session/close", &json!({"sessionId": "nope"}));
        assert_eq!(close.unwrap_err().category, "session_not_found");
    }

    #[tokio::test]
    async fn prompt_errors_before_any_update_for_bad_requests() {
        let (agent, mut frames) = agent(Args::default());
        let unknown = agent.prompt(&json!({"sessionId": "nope", "prompt": [{"type": "text", "text": "hi"}]})).await;
        assert_eq!(unknown.unwrap_err().category, "session_not_found");
        let blank = agent.prompt(&json!({"sessionId": "nope", "prompt": []})).await;
        assert_eq!(blank.unwrap_err().category, "invalid_params");
        assert!(frames.try_recv().is_err());
    }

    /// Answer the next permission request in `frames` with `outcome`, as OpenAB would, and return it.
    async fn answer(frames: &mut mpsc::UnboundedReceiver<Value>, client: &Client, outcome: Value) -> Value {
        let request = frames.recv().await.unwrap();
        client.resolve(json!({"jsonrpc": "2.0", "id": request["id"], "result": {"outcome": outcome}}));
        request
    }

    #[tokio::test]
    async fn repair_writes_ask_permission_once_each_and_audit_the_edited_path() {
        let (out, mut frames) = mpsc::unbounded_channel();
        let client = Arc::new(Client::new(out));
        let path = temp_path("repair-audit.jsonl");
        let observer = RepairObserver {
            client: client.clone(), log: AuditLog { path: Some(path.clone()), agent: "fixer".into() },
            session: "s-1".into(), calls: "s-1-repair-1".into(), workspace: std::env::temp_dir(),
            open: Mutex::new(None), audit_error: Mutex::new(None),
        };
        let edit = ToolCall {
            index: 2, origin: "solver".into(), tool: "apply_edit".into(),
            arguments: json!({"handle": "source:a", "old": "speed - gravity", "new": "speed + gravity"}),
            path: Some("engine/physics.py".into()), permission: None,
        };

        observer.started(&edit);
        let started = frames.try_recv().unwrap()["params"]["update"].clone();
        assert_eq!((started["sessionUpdate"].as_str(), started["kind"].as_str()), (Some("tool_call"), Some("edit")));
        assert_eq!(started["status"], "pending");
        assert!(started["locations"][0]["path"].as_str().unwrap().ends_with("physics.py"));
        assert_eq!(started["content"][0]["newText"], "speed + gravity");

        let rejected = json!({"outcome": "selected", "optionId": REJECT});
        let (approved, asked) = tokio::join!(observer.permit(&edit), answer(&mut frames, &client, rejected));
        assert!(!approved);
        assert_eq!(asked["method"], "session/request_permission");
        assert_eq!(asked["params"]["sessionId"], "s-1");
        assert_eq!(asked["params"]["toolCall"]["toolCallId"], started["toolCallId"]);
        let kinds: Vec<_> = asked["params"]["options"].as_array().unwrap().iter().map(|o| o["kind"].clone()).collect();
        assert_eq!(kinds, [json!("allow_once"), json!("reject_once")]);
        let error = json!({"error": "The client rejected this apply_edit call, so the workspace is unchanged."});
        observer.finished(&ToolCall { permission: Some(false), ..edit.clone() }, &error, true);
        let update = frames.try_recv().unwrap()["params"]["update"].clone();
        assert_eq!((update["toolCallId"].clone(), update["status"].as_str()), (started["toolCallId"].clone(), Some("failed")));

        let allowed = json!({"outcome": "selected", "optionId": ALLOW});
        let (approved, _) = tokio::join!(observer.permit(&edit), answer(&mut frames, &client, allowed));
        assert!(approved);
        let receipt = json!({"path": "engine/physics.py", "before": "b", "after": "a"});
        observer.finished(&ToolCall { permission: Some(true), ..edit.clone() }, &receipt, false);
        assert_eq!(frames.try_recv().unwrap()["params"]["update"]["status"], "completed");

        // A cancelled outcome and a closed connection both reject the write.
        let cancelled = json!({"outcome": "cancelled"});
        assert!(!tokio::join!(observer.permit(&edit), answer(&mut frames, &client, cancelled)).0);
        let closed = async {
            frames.recv().await.unwrap();
            client.close();
        };
        assert!(!tokio::join!(observer.permit(&edit), closed).0);
        assert!(client.pending.lock().unwrap().is_empty());

        let lines: Vec<Value> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        std::fs::remove_file(&path).unwrap();
        let writes: Vec<_> = lines.iter().map(|l| (l["op"].as_str(), l["status"].as_str(), l["path"].as_str())).collect();
        let path = Some("engine/physics.py");
        assert_eq!(writes, [(Some("write"), Some("rejected"), path), (Some("write"), Some("ok"), path)]);
        assert_eq!((lines[1]["agent"].as_str(), lines[1]["session"].as_str()), (Some("fixer"), Some("s-1")));
        assert!(observer.audit_error.lock().unwrap().is_none());
    }

    #[test]
    fn audit_lines_carry_agent_session_and_operation() {
        let path = temp_path("audit.jsonl");
        let (agent, _frames) = agent(Args { agent_name: "goblin".into(), audit_log: Some(path.clone()), ..Args::default() });
        agent.audit("s-1", json!({"op": "retrieve", "path": "/repo", "status": "ok"})).unwrap();
        agent.audit("s-1", json!({"op": "infer", "model": "m", "status": "ok"})).unwrap();
        let lines: Vec<Value> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(lines.len(), 2);
        assert_eq!((lines[0]["agent"].as_str(), lines[0]["session"].as_str()), (Some("goblin"), Some("s-1")));
        assert_eq!((lines[0]["op"].as_str(), lines[1]["op"].as_str()), (Some("retrieve"), Some("infer")));
        assert!(lines[1]["ts"].as_f64().unwrap() > 0.0);
    }
}
