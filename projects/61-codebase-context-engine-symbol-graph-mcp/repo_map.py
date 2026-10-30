"""Codebase context engine for coding agents.

Parses a repository with tree-sitter into a symbol graph (definitions and
call/import edges), ranks symbols with personalized PageRank seeded from the
files the agent is touching, and packs the highest-ranked signatures into a
token budget. Exposed as an MCP tool so any coding agent can request a
"repo map" instead of reading whole files.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import tiktoken
from mcp.server.fastmcp import FastMCP
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

PY = Language(tspython.language())
parser = Parser(PY)
enc = tiktoken.get_encoding("cl100k_base")
mcp = FastMCP("codebase-context")

DEF_QUERY = PY.query("""
(function_definition name: (identifier) @def.func) @node
(class_definition name: (identifier) @def.class) @node
""")
REF_QUERY = PY.query("(call function: [(identifier) @ref (attribute attribute: (identifier) @ref)])")


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str
    file: str
    line: int
    signature: str


class SymbolGraph:
    def __init__(self, root: Path):
        self.root = root
        self.symbols: dict[str, list[Symbol]] = defaultdict(list)
        self.refs: dict[str, set[str]] = defaultdict(set)   # file -> referenced names
        self.graph = nx.DiGraph()

    def index(self) -> "SymbolGraph":
        for path in self.root.rglob("*.py"):
            if any(p in path.parts for p in (".venv", "node_modules", ".git")):
                continue
            src = path.read_bytes()
            tree = parser.parse(src)
            rel = str(path.relative_to(self.root))
            for node, caps in DEF_QUERY.matches(tree.root_node):
                name_node = (caps.get("def.func") or caps.get("def.class"))[0]
                body = caps["node"][0]
                header = src[body.start_byte: body.end_byte].split(b":\n", 1)[0].decode(errors="ignore")
                kind = "class" if "def.class" in caps else "func"
                self.symbols[name_node.text.decode()].append(
                    Symbol(name_node.text.decode(), kind, rel, body.start_point[0] + 1, header.strip()[:160]))
            for node in REF_QUERY.captures(tree.root_node).get("ref", []):
                self.refs[rel].add(node.text.decode())
        self._build_edges()
        return self

    def _build_edges(self) -> None:
        for src_file, names in self.refs.items():
            for name in names:
                for sym in self.symbols.get(name, []):
                    if sym.file != src_file:
                        w = self.graph.get_edge_data(src_file, sym.file, {"weight": 0})["weight"]
                        self.graph.add_edge(src_file, sym.file, weight=w + 1)
        self.graph.add_nodes_from({s.file for syms in self.symbols.values() for s in syms})

    def rank_files(self, focus: list[str]) -> dict[str, float]:
        if not self.graph:
            return {}
        seeds = {f: 1.0 for f in focus if f in self.graph}
        personalization = seeds or None
        return nx.pagerank(self.graph, alpha=0.85, personalization=personalization, weight="weight")

    def repo_map(self, focus: list[str], budget_tokens: int = 2000) -> str:
        ranks = self.rank_files(focus)
        by_file: dict[str, list[Symbol]] = defaultdict(list)
        for syms in self.symbols.values():
            for s in syms:
                by_file[s.file].append(s)
        out, used = [], 0
        for file in sorted(by_file, key=lambda f: -ranks.get(f, 0.0)):
            block = [f"{file}:"] + [f"  L{s.line}: {s.signature}" for s in sorted(by_file[file], key=lambda s: s.line)]
            text = "\n".join(block)
            cost = len(enc.encode(text))
            if used + cost > budget_tokens:
                if used == 0:   # always include at least a truncated top file
                    out.append(enc.decode(enc.encode(text)[:budget_tokens]))
                break
            out.append(text)
            used += cost
        return "\n".join(out)


_graphs: dict[str, SymbolGraph] = {}


@mcp.tool()
def repo_map(root: str, focus_files: list[str], budget_tokens: int = 2000) -> str:
    """Return the most relevant symbol signatures for the files in focus, within a token budget."""
    if root not in _graphs:
        _graphs[root] = SymbolGraph(Path(root)).index()
    return _graphs[root].repo_map(focus_files, budget_tokens)


@mcp.tool()
def find_symbol(root: str, name: str) -> list[dict]:
    """Locate definitions of a symbol by name."""
    g = _graphs.setdefault(root, SymbolGraph(Path(root)).index())
    return [s.__dict__ for s in g.symbols.get(name, [])]


if __name__ == "__main__":
    mcp.run()
