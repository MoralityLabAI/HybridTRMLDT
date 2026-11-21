"""Verify framing and a real subprocess tool round trip, with no inference."""

from __future__ import annotations

import io
import json
import sys
import time
import unittest

from .serve import Session, serve
from .transport import JsonRpcError, PROTOCOL_VERSION, StdioClient, TransportError, validate_tools


class EchoEnv:
    task = {"request": "Echo a public value", "scope": "synthetic-test"}
    private_gold = "private-answer-never-on-wire"

    def tools(self):
        return [{"name": "echo", "description": "Echo a string", "inputSchema": {
            "type": "object", "properties": {"value": {"type": "string"}},
            "required": ["value"], "additionalProperties": False,
        }}]

    def call(self, name, arguments):
        if arguments["value"] == "raise":
            raise RuntimeError(self.private_gold)
        return {"content": [{"type": "text", "text": arguments["value"]}], "isError": False}


def initialize(request_id=0, version=PROTOCOL_VERSION):
    return {"jsonrpc": "2.0", "id": request_id, "method": "initialize", "params": {
        "protocolVersion": version, "capabilities": {}, "clientInfo": {"name": "tests", "version": "1"},
    }}


def initialized():
    return {"jsonrpc": "2.0", "method": "notifications/initialized"}


def ready_session():
    session = Session(EchoEnv())
    session.dispatch(initialize())
    session.dispatch(initialized())
    return session


class TransportTests(unittest.TestCase):
    def test_lifecycle_id_zero_and_public_only_metadata(self):
        session = Session(EchoEnv())
        early = session.dispatch({"jsonrpc": "2.0", "id": 0, "method": "tools/list"})
        self.assertEqual(early["error"]["code"], -32002)
        reply = session.dispatch(initialize())
        self.assertEqual(reply["id"], 0)
        self.assertEqual(reply["result"]["protocolVersion"], PROTOCOL_VERSION)
        self.assertEqual(reply["result"]["_meta"]["publicTask"], EchoEnv.task)
        self.assertNotIn(EchoEnv.private_gold, json.dumps(reply))
        self.assertIsNone(session.dispatch(initialized()))
        result = session.dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        self.assertEqual(result["result"]["tools"], EchoEnv().tools())

    def test_unknown_methods_tools_and_bad_arguments(self):
        session = ready_session()
        reply = session.dispatch({"jsonrpc": "2.0", "id": 2, "method": "sampling/createMessage"})
        self.assertEqual(reply["error"]["code"], -32601)
        for params in ({"name": "missing", "arguments": {}}, {"name": "echo", "arguments": []},
                       {"name": "echo", "arguments": {}}, {"name": "echo", "arguments": {"value": True}},
                       {"name": "echo", "arguments": {"value": "ok", "extra": 1}}):
            reply = session.dispatch({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": params})
            self.assertEqual(reply["error"]["code"], -32602)

    def test_execution_errors_are_tool_results_and_do_not_expose_internal_text(self):
        reply = ready_session().dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                          "params": {"name": "echo", "arguments": {"value": "raise"}}})
        self.assertTrue(reply["result"]["isError"])
        self.assertNotIn(EchoEnv.private_gold, json.dumps(reply))

    def test_newline_framing_parse_error_recovery_and_no_notification_reply(self):
        messages = [initialize(), initialized(), {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                    {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                     "params": {"name": "echo", "arguments": {"value": "line one\nline two"}}}]
        source = io.BytesIO(b"not-json\n" + b"".join(json.dumps(m).encode() + b"\n" for m in messages))
        output = io.BytesIO()
        serve(EchoEnv(), source, output)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 4)
        replies = [json.loads(line) for line in lines]
        self.assertEqual(replies[0]["error"]["code"], -32700)
        self.assertEqual([r["id"] for r in replies[1:]], [0, 1, 2])
        self.assertEqual(replies[-1]["result"]["content"][0]["text"], "line one\nline two")

    def test_duplicate_keys_and_invalid_utf8_are_parse_errors(self):
        source = io.BytesIO(b'{"jsonrpc":"2.0","id":0,"id":1,"method":"x"}\n\xff\n')
        output = io.BytesIO()
        serve(EchoEnv(), source, output)
        self.assertEqual([json.loads(line)["error"]["code"] for line in output.getvalue().splitlines()], [-32700, -32700])

    def test_tool_schema_assertions(self):
        with self.assertRaises(TransportError):
            validate_tools([{"name": "bad", "inputSchema": {"type": "array"}}])
        with self.assertRaises(TransportError):
            validate_tools(EchoEnv().tools() * 2)

    def test_explicit_null_and_boolean_request_ids_are_invalid(self):
        for request_id in (None, True, False):
            reply = Session(EchoEnv()).dispatch(initialize(request_id))
            self.assertEqual(reply["error"]["code"], -32600)

    def test_client_checks_response_ids_and_closes_process(self):
        script = 'import sys,json;sys.stdin.readline();print(json.dumps({"jsonrpc":"2.0","id":999,"result":{}}),flush=True)'
        with StdioClient(command=[sys.executable, "-u", "-c", script]) as client:
            with self.assertRaisesRegex(TransportError, "ID mismatch"):
                client.initialize()
            self.assertIsNotNone(client.process.poll())

    def test_client_rejects_unsupported_version(self):
        script = ('import sys,json;r=json.loads(sys.stdin.readline());'
                  'print(json.dumps({"jsonrpc":"2.0","id":r["id"],"result":'
                  '{"protocolVersion":"unknown","capabilities":{"tools":{}}}}),flush=True)')
        with StdioClient(command=[sys.executable, "-u", "-c", script]) as client:
            with self.assertRaisesRegex(TransportError, "protocol version"):
                client.initialize()

    def test_client_timeout_cleans_up_process(self):
        script = 'import sys,time;sys.stdin.readline();time.sleep(30)'
        started = time.monotonic()
        with StdioClient(command=[sys.executable, "-u", "-c", script], timeout=0.05) as client:
            with self.assertRaisesRegex(TransportError, "timed out"):
                client.initialize()
            self.assertIsNotNone(client.process.poll())
        self.assertLess(time.monotonic() - started, 2)

    def test_client_timeout_also_bounds_a_blocked_stdin_write(self):
        script = 'import time;time.sleep(30)'
        started = time.monotonic()
        with StdioClient(command=[sys.executable, "-u", "-c", script], timeout=0.05) as client:
            with self.assertRaisesRegex(TransportError, "timed out writing"):
                client.request("large-request", {"payload": "x" * 262144})
            self.assertIsNotNone(client.process.poll())
        self.assertLess(time.monotonic() - started, 2)

    def test_client_rejects_stdout_noise(self):
        script = 'import sys;sys.stdin.readline();print("server log on stdout",flush=True)'
        with StdioClient(command=[sys.executable, "-u", "-c", script]) as client:
            with self.assertRaisesRegex(TransportError, "invalid JSON"):
                client.initialize()

    def test_real_stdio_handshake_and_observed_tool_execution(self):
        from .environment import RetrievalEnv, choose_reference_action, make_cases
        cases = make_cases(n=1, catalog_size=8, depth=3)
        for architecture in ("flat", "search", "layered", "layered_ldt"):
            with self.subTest(architecture=architecture):
                env = RetrievalEnv(cases[0], architecture, catalog_size=8, depth=3)
                history = []
                with StdioClient(architecture=architecture, catalog_size=8, depth=3, n=1) as client:
                    init = client.initialize()
                    self.assertEqual(init, client.initialize())
                    self.assertEqual(init["_meta"]["publicTask"], env.task)
                    tools = client.tools()
                    self.assertEqual(tools, env.tools())
                    observed = False
                    for _ in range(24):
                        action = choose_reference_action(env.task, tools, history)
                        if action["tool"] == "finish":
                            break
                        result = client.call(action["tool"], action["arguments"])
                        expected = env.call(action["tool"], action["arguments"])
                        self.assertEqual(result, expected)
                        self.assertFalse(result.get("isError", False))
                        observed = True
                        history.append({**action, "result": result})
                    else:
                        self.fail("reference retrieval failed to finish within step budget")
                    self.assertTrue(observed)
                    self.assertEqual(action["tool"], "finish")
                    with self.assertRaises(JsonRpcError) as caught:
                        client.call("nonexistent-tool", {})
                    self.assertEqual(caught.exception.code, -32602)
                    self.assertEqual(client.tools(), tools)
                self.assertIsNotNone(client.process.poll())


if __name__ == "__main__":
    unittest.main()
