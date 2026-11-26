//! Minimal MCP stdio client: newline-delimited JSON-RPC 2.0, protocol 2025-06-18.

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

pub struct McpClient {
    _child: Child,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
    next_id: i64,
}

impl McpClient {
    /// Start `program args` in `cwd` (also its PYTHONPATH) and complete the initialize handshake.
    pub async fn spawn(program: &str, args: &[&str], cwd: &str, client_name: &str) -> Result<Self> {
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
        let stdin = child.stdin.take().ok_or_else(|| anyhow!("MCP server has no stdin"))?;
        let stdout = BufReader::new(child.stdout.take().ok_or_else(|| anyhow!("MCP server has no stdout"))?).lines();
        let mut client = Self { _child: child, stdin, stdout, next_id: 0 };
        client
            .request(
                "initialize",
                json!({"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": client_name, "version": env!("CARGO_PKG_VERSION")}}),
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

    pub async fn request(&mut self, method: &str, params: Value) -> Result<Value> {
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
        bail!("MCP server closed its output")
    }

    /// Call a tool; an `isError` result becomes an error carrying the tool's payload.
    pub async fn call_tool(&mut self, name: &str, arguments: Value) -> Result<Value> {
        let result = self.request("tools/call", json!({"name": name, "arguments": arguments})).await?;
        if result.get("isError").and_then(Value::as_bool) == Some(true) {
            bail!("tool {name} failed: {}", result.get("structuredContent").unwrap_or(&result));
        }
        Ok(result.get("structuredContent").cloned().unwrap_or(result))
    }
}
