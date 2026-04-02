"""Trace Tree - Hierarchical tree structure built from a DistilledTrace.

Organizes trace spans into a tree using parent-child relationships,
with children ordered chronologically (left to right by timestamp).
Span IDs are used as node identifiers throughout.

Usage:

    tree = TraceTree(distilled_trace)

    # Iterate all nodes in pre-order (top-down, left-to-right)
    for node in tree.iter_preorder():
        print(node.depth, node.span.span_name)

    # Get spans excluding bottom-layer LLM and Tool calls
    high_level_spans = tree.get_spans_excluding_leaf_llm_tool()

    # Get direct children of a span
    children = tree.get_children("some_span_id")

    # Get all descendants under a span
    descendants = tree.get_descendants("some_span_id")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from src.graph.constants import SpanKind
from src.graph.distilled_trace import DistilledSpan, DistilledTrace


@dataclass
class TreeNode:
    """A node in the trace tree, identified by its span_id.

    Attributes:
        span: The underlying span data.
        children: Ordered list of child span_ids (sorted by timestamp, left to right).
        depth: Depth in the tree (0 = root).
    """

    span: DistilledSpan
    children: list[str] = field(default_factory=list)  # child span_ids
    depth: int = 0

    @property
    def span_id(self) -> str:
        return self.span.span_id

    @property
    def is_leaf(self) -> bool:
        """True if this node has no children."""
        return len(self.children) == 0

    def is_bottom_layer_llm_or_tool(self, max_depth: int) -> bool:
        """True if this node is at the deepest layer and is LLM or TOOL."""
        return self.depth == max_depth and self.span.span_kind in (SpanKind.LLM, SpanKind.TOOL)

    def __repr__(self) -> str:
        return (
            f"TreeNode(span_id={self.span_id!r}, "
            f"kind={self.span.span_kind.value}, "
            f"depth={self.depth}, "
            f"children={self.children})"
        )


class TraceTree:
    """Tree structure built from a DistilledTrace using parent-child span hierarchy.

    Span IDs serve as node identifiers. The internal node map (``_node_map``)
    maps each span_id to its ``TreeNode``. Children lists and root lists all
    store span_ids; use ``get_node(span_id)`` to retrieve the actual node.

    Children are ordered by timestamp (ascending), representing left-to-right
    execution order. Spans whose parent is absent or not in the trace become
    root nodes.

    Key extraction methods:

    1. ``get_spans_excluding_leaf_llm_tool()``
       Returns spans in pre-order excluding LLM/TOOL nodes at the deepest
       layer (``max_depth``). Mid-level LLM/TOOL nodes are retained.

    2. ``get_children(span_id)`` / ``get_descendants(span_id)``
       Extract direct children or the full subtree under a given span.
    """

    def __init__(self, trace: DistilledTrace) -> None:
        self.trace = trace
        # span_id -> TreeNode
        self._node_map: dict[str, TreeNode] = {}
        # span_ids of root nodes (no valid parent), sorted by timestamp
        self.root_node_ids: list[str] = []
        self._build()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Build the tree from the trace's flat span dictionary."""
        # 1. Create a TreeNode for every span
        for span_id, span in self.trace.spans.items():
            self._node_map[span_id] = TreeNode(span=span)

        # 2. Wire parent → child (store child span_ids)
        for span_id, node in self._node_map.items():
            parent_id = node.span.parent_span_id
            if parent_id and parent_id in self._node_map:
                self._node_map[parent_id].children.append(span_id)
            else:
                self.root_node_ids.append(span_id)

        # 3. Sort children by timestamp (left-to-right chronological order)
        for node in self._node_map.values():
            node.children.sort(
                key=lambda sid: self._node_map[sid].span.timestamp or ""
            )

        # 4. Assign depths via BFS from roots
        queue: list[tuple[str, int]] = [(sid, 0) for sid in self.root_node_ids]
        while queue:
            span_id, depth = queue.pop(0)
            node = self._node_map[span_id]
            node.depth = depth
            for child_id in node.children:
                queue.append((child_id, depth + 1))

        # 5. Sort root node ids by timestamp
        self.root_node_ids.sort(
            key=lambda sid: self._node_map[sid].span.timestamp or ""
        )

    # ------------------------------------------------------------------
    # Node access
    # ------------------------------------------------------------------

    def get_node(self, span_id: str) -> TreeNode | None:
        """Return the TreeNode for a given span_id, or None if not found."""
        return self._node_map.get(span_id)

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def iter_preorder(self, root_id: str | None = None) -> Iterator[TreeNode]:
        """Yield all nodes in pre-order (parent before children, left to right).

        Args:
            root_id: span_id of the starting node. If None, iterates from all
                root nodes.
        """
        start_ids = [root_id] if root_id is not None else self.root_node_ids
        stack = list(reversed(start_ids))  # reversed so leftmost is popped first
        while stack:
            sid = stack.pop()
            node = self._node_map.get(sid)
            if node is None:
                continue
            yield node
            stack.extend(reversed(node.children))

    # ------------------------------------------------------------------
    # Extraction method 1: exclude bottom-layer LLM / Tool spans
    # ------------------------------------------------------------------

    @property
    def max_depth(self) -> int:
        """The maximum depth across all nodes in the tree (0-indexed)."""
        if not self._node_map:
            return 0
        return max(node.depth for node in self._node_map.values())

    def get_spans_excluding_leaf_llm_tool(
        self,
        root_span_id: str | None = None,
    ) -> list[DistilledSpan]:
        """Return spans in pre-order, excluding bottom-layer LLM and Tool spans.

        "Bottom-layer" means depth == max depth of the traversed subtree.
        Only LLM/TOOL spans at that exact depth are excluded. LLM or TOOL
        spans at intermediate depths are retained even without children.

        Args:
            root_span_id: If given, restrict traversal to the subtree rooted
                at this span (uses that subtree's own max depth).

        Returns:
            List of DistilledSpan in pre-order, without bottom-layer LLM/Tool spans.
        """
        nodes = list(self.iter_preorder(root_id=root_span_id))
        if not nodes:
            return []
        subtree_max_depth = max(n.depth for n in nodes)
        return [
            node.span
            for node in nodes
            if not node.is_bottom_layer_llm_or_tool(subtree_max_depth)
        ]

    # ------------------------------------------------------------------
    # Extraction method 2: children / descendants by span_id
    # ------------------------------------------------------------------

    def get_children(self, span_id: str) -> list[DistilledSpan]:
        """Return the direct child spans of the given span, in timestamp order.

        Args:
            span_id: The parent span's ID.

        Returns:
            Ordered list of direct child DistilledSpan objects.
            Empty list if the span is not found or has no children.
        """
        node = self._node_map.get(span_id)
        if node is None:
            return []
        return [self._node_map[cid].span for cid in node.children]

    def get_descendants(
        self,
        span_id: str,
        *,
        include_root: bool = False,
    ) -> list[DistilledSpan]:
        """Return all descendant spans under the given span (full subtree).

        Spans are returned in pre-order (parent before its children,
        left to right by timestamp).

        Args:
            span_id: The ancestor span's ID.
            include_root: If True, the span itself is included as the first
                element. Defaults to False (only descendants).

        Returns:
            Ordered list of descendant DistilledSpan objects.
            Empty list if the span is not found.
        """
        nodes = list(self.iter_preorder(root_id=span_id))
        if not include_root:
            nodes = nodes[1:]
        return [n.span for n in nodes]

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def all_spans_preorder(self) -> list[DistilledSpan]:
        """All spans in the tree in pre-order (no filtering)."""
        return [node.span for node in self.iter_preorder()]

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot(
        self,
        figsize: tuple[int, int] = (20, 10),
        max_label_len: int = 30,
        level_gap: float = 2.0,
        title: str | None = None,
    ):
        """Plot the trace tree using matplotlib.

        Layout: x-axis = left-to-right execution order (timestamp),
        y-axis = tree depth (root at top).  All nodes at the same depth are
        pinned to the same horizontal level (y = -depth * level_gap).
        Nodes are colour-coded by SpanKind.

        Args:
            figsize: Matplotlib figure size (width, height) in inches.
            max_label_len: Maximum characters for the span_name label.
            level_gap: Vertical distance between consecutive depth levels.
            title: Figure title.  Defaults to the trace_id.

        Returns:
            (fig, ax) tuple so the caller can further customise or save.
        """
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        KIND_COLORS: dict[SpanKind, str] = {
            SpanKind.AGENT:     "#8172B2",  # purple
            SpanKind.CHAIN:     "#55A868",  # green
            SpanKind.LLM:       "#4C72B0",  # blue
            SpanKind.TOOL:      "#DD8452",  # orange
            SpanKind.RETRIEVER: "#C44E52",  # red
            SpanKind.EMBEDDING: "#937860",  # brown
            SpanKind.UNKNOWN:   "#AAAAAA",  # grey
        }

        # ----------------------------------------------------------
        # 1. Compute (x, y) positions
        #    x: leaf nodes get consecutive integers; parents are
        #       centred over their children.
        #    y: strictly -depth * level_gap — guaranteed same y for
        #       every node at the same depth, regardless of subtree shape.
        # ----------------------------------------------------------
        positions: dict[str, tuple[float, float]] = {}

        def _assign(span_id: str, x_start: float, depth: int) -> float:
            """Assign positions recursively; return next free x cursor."""
            y = -depth * level_gap          # y depends only on depth
            node = self._node_map[span_id]
            if node.is_leaf:
                positions[span_id] = (x_start + 0.5, y)
                return x_start + 1.0

            x = x_start
            child_xs: list[float] = []
            for cid in node.children:
                x = _assign(cid, x, depth + 1)
                child_xs.append(positions[cid][0])

            # Centre parent over its children
            positions[span_id] = (sum(child_xs) / len(child_xs), y)
            return x

        x_cursor = 0.0
        for root_id in self.root_node_ids:
            x_cursor = _assign(root_id, x_cursor, 0)

        total_width = x_cursor

        # ----------------------------------------------------------
        # 2. Draw
        # ----------------------------------------------------------
        fig, ax = plt.subplots(figsize=figsize)

        # Horizontal reference lines — one per depth level
        all_depths = sorted({node.depth for node in self._node_map.values()})
        for depth in all_depths:
            y_level = -depth * level_gap
            ax.axhline(y=y_level, color="#E8E8E8", lw=0.8, zorder=0)
            ax.text(
                -0.4, y_level, f"depth {depth}",
                va="center", ha="right", fontsize=7, color="#999999",
            )

        # Edges
        for span_id, node in self._node_map.items():
            if span_id not in positions:
                continue
            x1, y1 = positions[span_id]
            for cid in node.children:
                if cid not in positions:
                    continue
                x2, y2 = positions[cid]
                ax.plot([x1, x2], [y1, y2], color="#CCCCCC", lw=1.0, zorder=1)

        # Nodes + labels
        for span_id, node in self._node_map.items():
            if span_id not in positions:
                continue
            x, y = positions[span_id]
            color = KIND_COLORS.get(node.span.span_kind, "#AAAAAA")

            ax.scatter(x, y, c=color, s=350, zorder=3,
                       edgecolors="white", linewidths=1.2)

            if node.span.span_kind == SpanKind.LLM:
                name = "LLM"
            else:
                name = node.span.span_name
                if len(name) > max_label_len:
                    name = name[:max_label_len - 1] + "…"
            short_id = span_id[:6]
            label = f"{name}\n{short_id}"
            ax.annotate(
                label, (x, y),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=6.5, color="#222222",
            )

        # Legend
        legend_handles = [
            mpatches.Patch(color=color, label=kind.value)
            for kind, color in KIND_COLORS.items()
        ]
        ax.legend(handles=legend_handles, loc="upper right",
                  fontsize=8, framealpha=0.8)

        ax.set_title(title or f"Trace Tree: {self.trace.trace_id}", fontsize=11)
        ax.set_xlim(-0.8, total_width + 0.2)
        ax.axis("off")
        plt.tight_layout()
        return fig, ax

    def __len__(self) -> int:
        return len(self._node_map)

    def __repr__(self) -> str:
        return (
            f"TraceTree(trace_id={self.trace.trace_id!r}, "
            f"spans={len(self)}, roots={len(self.root_node_ids)})"
        )


# ---------------------------------------------------------------------------
# Helpers to build a DistilledTrace from a raw GAIA JSON file
# ---------------------------------------------------------------------------

def _flatten_raw_spans(
    raw_spans: list[dict],
    parent_id: str | None = None,
) -> list[dict]:
    """Recursively flatten a hierarchical span list into a flat list."""
    result = []
    for span in raw_spans:
        flat = {k: v for k, v in span.items() if k != "child_spans"}
        flat["parent_span_id"] = parent_id
        result.append(flat)
        result.extend(_flatten_raw_spans(span.get("child_spans", []), span["span_id"]))
    return result



def distilled_trace_from_gaia(raw: dict) -> "DistilledTrace":
    """Build a DistilledTrace from a raw GAIA JSON trace dict."""
    from src.graph.distilled_trace import DistilledEvent, DistilledSpan, DistilledTrace, SourceFormat
    from src.utils.trace_utils import _KEEP_ATTR_KEYS

    flat_spans = _flatten_raw_spans(raw.get("spans", []))
    spans: dict[str, DistilledSpan] = {}

    for s in flat_spans:
        span_id = s.get("span_id", "")
        if not span_id:
            continue

        attrs = s.get("span_attributes", {})
        filtered_attrs = {k: v for k, v in attrs.items() if k in _KEEP_ATTR_KEYS}
        kind = SpanKind.from_string(filtered_attrs.get("openinference.span.kind"))
        if "openinference.span.kind" not in filtered_attrs:
            filtered_attrs["openinference.span.kind"] = kind.value

        # Events
        events = [
            DistilledEvent(
                name=e.get("Name", ""),
                timestamp=e.get("Timestamp"),
                attributes=e.get("Attributes", {}),
            )
            for e in s.get("events", [])
        ]

        span_obj = DistilledSpan(
            span_id=span_id,
            span_name=s.get("span_name", ""),
            span_kind=kind,
            status_code=s.get("status_code", "Unset"),
            status_message=s.get("status_message") or None,
            parent_span_id=s.get("parent_span_id"),
            span_attributes=filtered_attrs,
            duration=s.get("duration"),
            events=events,
        )
        span_obj._timestamp = s.get("timestamp")
        spans[span_id] = span_obj

    return DistilledTrace(
        trace_id=raw.get("trace_id", "unknown"),
        spans=spans,
        source_format=SourceFormat.GAIA,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys
    import matplotlib.pyplot as plt
    from pathlib import Path

    DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "GAIA"

    # Accept an optional trace file path as CLI argument
    if len(sys.argv) > 1:
        trace_path = Path(sys.argv[1])
    else:
        # Default: first file in GAIA directory
        candidates = sorted(DATA_DIR.glob("*.json"))
        if not candidates:
            print(f"No trace files found in {DATA_DIR}")
            sys.exit(1)
        trace_path = candidates[20]

    print(f"Loading trace: {trace_path.name}")
    raw = json.loads(trace_path.read_text())

    trace = distilled_trace_from_gaia(raw)
    tree = TraceTree(trace)

    print(tree)
    print(f"  max_depth : {tree.max_depth}")
    print(f"  root ids  : {tree.root_node_ids}")

    # --- method 1: spans excluding bottom-layer LLM/Tool ---
    high_level = tree.get_spans_excluding_leaf_llm_tool()
    print(f"\nSpans excluding bottom-layer LLM/Tool ({len(high_level)}/{len(tree)}):")
    for sp in high_level:
        node = tree.get_node(sp.span_id)
        print(f"  [depth={node.depth}] {sp.span_kind.value:10s}  {sp.span_name}")

    # --- method 2: children of the first root ---
    if tree.root_node_ids:
        root_id = tree.root_node_ids[0]
        children = tree.get_children(root_id)
        print(f"\nDirect children of root '{root_id[:8]}…' ({len(children)}):")
        for sp in children:
            print(f"  {sp.span_kind.value:10s}  {sp.span_name}")

    # --- plot ---
    fig, ax = tree.plot(figsize=(24, 12))
    out_path = Path(f"trace_tree_{trace.trace_id[:8]}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nTree plot saved to: {out_path}")
    plt.show()
