"""MCP-03: compact context packets from real code-repository MCP catalogs.

Each task is a SWE-bench Lite or SWE-bench instance: an issue text and a
repository snapshot at the instance's base commit. A local router explores the
repository through MCP tools (BM25 code search, a layered package/module/symbol
index, symbol reads) and hands a frontier agent a token-budgeted packet. The
gold patch's files and functions are the hidden evaluation target.
"""

GYM_VERSION = "mcp-03"
