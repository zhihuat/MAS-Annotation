
"""Graph builder for converting traces to NetworkX graphs.

Domain service that builds graph representations from raw traces.
Uses strict computational methods for data flow detection.
"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any

import networkx as nx

from src.graph.constants import (
    EdgeType,
    SpanAttributes,
    SpanKind,
)

logger = logging.getLogger(__name__)


# Regex patterns for call ID extraction
CALL_ID_PATTERN = re.compile(r"Call id: (call_\d+)")
TOOL_CALL_ID_PATTERN = re.compile(r"['\"]id['\"]:\s*['\"]?(call_\d+)['\"]?")


class GraphBuilder:
    """Builds NetworkX graphs from raw OpenTelemetry traces."""

    def __init__(self, llm_verify: bool = False, llm_provider = None, max_history_check: int = 0):
        """Initialize graph builder.

        Args:
            llm_verify: Enable LLM verification for data flow edges (hybrid mode)
            llm_provider: LLM interface for verification (uses OpenAIClient if None)
            max_history_check: Number of recent messages to check for data flow (default 0)
        """
        self.llm_verify = llm_verify
        self.llm_provider = llm_provider
        self.max_history_check = max_history_check
        self._llm_cache: dict[str, bool] = {}  # Cache LLM verification results

    def build_from_trace(
        self,
        trace: dict[str, Any],
        include_hierarchy: bool = True,
        include_data_flow: bool = True
    ) -> tuple[nx.DiGraph, dict[str, int]]:
        """Build NetworkX graph from trace structure.

        Args:
            trace: Raw trace with hierarchical spans
            include_hierarchy: Include parent-child edges
            include_data_flow: Include data flow edges (heuristic-based)

        Returns:
            Tuple of (NetworkX DiGraph, execution order map)
        """
        logger.info("Building graph from trace")

        # Flatten spans and collect timestamps
        span_map = self._flatten_trace(trace)
        spans_with_time = self._collect_spans_with_time(span_map)

        # Build order map
        G = nx.DiGraph()
        order_map = {}

        for order, (span_id, _, span) in enumerate(spans_with_time):
            order_map[span_id] = order

            # Reduce verbosity: If LLM messages exist, remove redundant input/output values
            attrs = span.get("span_attributes", {})
            if "llm.input_messages" in attrs:
                attrs.pop(SpanAttributes.INPUT_VALUE, None)
            if "llm.output_messages" in attrs:
                attrs.pop(SpanAttributes.OUTPUT_VALUE, None)

            # Add node with span data
            G.add_node(span_id, **span)

        # Add edges
        if include_hierarchy:
            self._add_hierarchy_edges(G, spans_with_time)

        if include_data_flow:
            # Check if this is a tracegen trace with explicit data sources
            if self._is_tracegen_trace(span_map):
                # Use explicit data sources - no heuristics needed
                logger.info("Detected tracegen format - using explicit input_data_sources")
                self._add_tracegen_data_edges(G, span_map)
            else:
                # Fall back to heuristic detection for other formats
                self._add_data_flow_edges(G, spans_with_time, order_map)
                # Deduplicate: keep only one data edge per node type
                self._deduplicate_data_flows_by_type(G, span_map)
                # # Postorder: detect bubble-up data flows (child output → parent output)
                # self._add_postorder_bubble_up_edges(G, span_map)
                # # Transitive bubble-up: deep descendant LLM → ancestor AGENT
                # self._add_transitive_bubble_up_edges(G, span_map)
                # # Forward flow: AGENT output → next consumer (LLM/TOOL)
                # self._add_agent_forward_flow_edges(G, span_map, spans_with_time)
                # # Delegation flow: LLM output → sibling AGENT/TOOL input (flow down)
                # self._add_delegation_flow_edges(G, span_map, spans_with_time)
                # # Sequence flow: Link siblings chronologically (Child 1 -> Child 2)
                # self._add_sequence_flow_edges(G, span_map, spans_with_time)

        logger.info(f"Graph built: {len(G.nodes)} nodes, {len(G.edges)} edges")
        return G, order_map

    def _add_sequence_flow_edges(
        self,
        G: nx.DiGraph,
        span_map: dict[str, Any],
        spans_with_time: list[tuple[str, datetime | None, dict[str, Any]]]
    ):
        """Add sequence edges between chronological siblings.

        This links children of the same parent in the order they occurred.
        Essential for visualization to show the 'flow' of steps or calls
        left-to-right (or top-to-bottom) within a parent container.
        
        Enhancement: connecting 'bridge' nodes (deepest descendants) enforces
        layout constraints even for container/combo nodes.
        """
        # Group children by parent
        parent_children = {}
        for span_id, _, span in spans_with_time:
            parent_id = span.get("parent_span_id")
            if parent_id:
                if parent_id not in parent_children:
                    parent_children[parent_id] = []
                parent_children[parent_id].append((span_id, span))

        # Sort key helper
        def get_sort_key(item):
            sid, s = item
            ts = self._parse_timestamp(s.get("timestamp"))
            if ts:
                return (ts.timestamp(), sid)
            return (0, sid)

        # Pre-sort all children lists for consistent navigation
        for pid in parent_children:
            parent_children[pid].sort(key=get_sort_key)

        count = 0
        for parent_id, children in parent_children.items():
            if len(children) < 2:
                continue

            # Link adjacent siblings
            for i in range(len(children) - 1):
                u_id, _ = children[i]
                v_id, _ = children[i + 1]

                # High-level structural edge
                if not G.has_edge(u_id, v_id):
                    # No edge exists - create new sequence edge
                    G.add_edge(
                        u_id,
                        v_id,
                        type="sequence",
                        optional={
                            "edge_type": "sequence",
                            "is_sequence": True
                        }
                    )
                    count += 1
                else:
                    # Edge already exists (likely data/hierarchy edge)
                    # Mark it as also being a sequence edge for layout purposes
                    edge_data = G[u_id][v_id]
                    if "optional" not in edge_data:
                        edge_data["optional"] = {}
                    edge_data["optional"]["is_sequence"] = True
                    logger.debug(f"Marked existing edge as sequence: {u_id} → {v_id} (type: {edge_data.get('type')})")
        
        if count > 0:
            logger.info(f"Added {count} sequence flow edges")

    def _flatten_trace(self, trace: dict[str, Any]) -> dict[str, Any]:
        """Flatten hierarchical trace into span_id -> span dict."""
        span_map = {}

        def recurse(span_raw, parent_id):
            # Clean copy without children
            clean_span = {k: v for k, v in span_raw.items() if k != 'child_spans'}
            clean_span['parent_span_id'] = parent_id
            span_id = clean_span.get("span_id")

            if span_id:
                span_map[span_id] = clean_span

            for child in span_raw.get("child_spans", []):
                recurse(child, span_id)

        for root in trace.get("spans", []):
            recurse(root, None)

        return span_map

    def _is_tracegen_trace(self, span_map: dict[str, Any]) -> bool:
        """Check if trace has tracegen's input_data_sources attributes."""
        for span_data in span_map.values():
            attrs = span_data.get("span_attributes", {})
            if SpanAttributes.INPUT_DATA_SOURCES in attrs:
                return True
        return False

    def _add_tracegen_data_edges(
        self,
        G: nx.DiGraph,
        span_map: dict[str, Any]
    ) -> None:
        """Build data flow edges from tracegen's input_data_sources attribute.

        Tracegen spans have explicit input_data_sources JSON attribute:
        [{"span_id": "abc", "edge_type": "tool_result", "data": "..."}]

        This provides deterministic data flow edges without heuristics.
        """
        tracegen_edge_count = 0

        for span_id, span_data in span_map.items():
            attrs = span_data.get("span_attributes", {})
            sources_json = attrs.get(SpanAttributes.INPUT_DATA_SOURCES)

            if not sources_json:
                continue

            # Parse the JSON array
            try:
                if isinstance(sources_json, str):
                    sources = json.loads(sources_json)
                else:
                    sources = sources_json
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Failed to parse input_data_sources for span {span_id}")
                continue

            if not isinstance(sources, list):
                continue

            # Create edges from each source
            for source in sources:
                if not isinstance(source, dict):
                    continue

                source_span_id = source.get("span_id")
                edge_type = source.get("edge_type", "data")
                data_content = source.get("data", "")

                # Only create edge if source span exists in graph
                if source_span_id and source_span_id in span_map:
                    # Map tracegen edge types to our EdgeType enum
                    mapped_type = self._map_tracegen_edge_type(edge_type)

                    G.add_edge(
                        source_span_id,
                        span_id,
                        type=mapped_type,
                        has_data_flow=True,
                        data_content=data_content[:500] if data_content else None,
                        tracegen_edge_type=edge_type,
                    )
                    tracegen_edge_count += 1
                    logger.debug(
                        f"Tracegen data edge: {source_span_id[:8]}... -> {span_id[:8]}... ({edge_type})"
                    )

        if tracegen_edge_count > 0:
            logger.info(f"Added {tracegen_edge_count} data edges from tracegen input_data_sources")

    def _map_tracegen_edge_type(self, tracegen_type: str) -> str:
        """Map tracegen edge type to our EdgeType enum values."""
        mapping = {
            "delegation": EdgeType.DELEGATION_FLOW.value,
            "conversation": EdgeType.DATA.value,
            "tool_call": EdgeType.DATA.value,
            "tool_result": EdgeType.DATA.value,
        }
        return mapping.get(tracegen_type, EdgeType.DATA.value)

    def _collect_spans_with_time(
        self,
        span_map: dict[str, Any]
    ) -> list[tuple[str, datetime | None, dict[str, Any]]]:
        """Collect spans with parsed timestamps and sort by time."""
        spans_with_time = []

        for span_id, span in span_map.items():
            timestamp = self._parse_timestamp(span.get("timestamp"))
            spans_with_time.append((span_id, timestamp, span))

        # Sort by timestamp
        spans_with_time.sort(key=lambda x: x[1] if x[1] else datetime.min)

        return spans_with_time

    def _parse_timestamp(self, ts: str | None) -> datetime | None:
        """Parse ISO timestamp string."""
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return None

    def _parse_duration(self, duration_str: str | None) -> float:
        """Parse ISO 8601 duration string to seconds.

        Handles formats like:
        - PT8.002459S (8.002459 seconds)
        - PT3M1.140869S (3 minutes 1.140869 seconds)
        """
        if not duration_str:
            return 0.0

        # Match seconds only: PT8.002459S
        match = re.match(r'PT(\d+(?:\.\d+)?)S', duration_str)
        if match:
            return float(match.group(1))

        # Match minutes and seconds: PT3M1.140869S
        match = re.match(r'PT(\d+)M(\d+(?:\.\d+)?)S', duration_str)
        if match:
            minutes = int(match.group(1))
            seconds = float(match.group(2))
            return minutes * 60 + seconds

        return 0.0

    def _compute_end_time(self, span: dict[str, Any]) -> datetime | None:
        """Compute span end time from timestamp + duration."""
        start = self._parse_timestamp(span.get("timestamp"))
        if not start:
            return None

        duration_secs = self._parse_duration(span.get("duration"))
        if duration_secs > 0:
            return start + timedelta(seconds=duration_secs)

        return start  # No duration, assume instant

    def _add_hierarchy_edges(
        self,
        G: nx.DiGraph,
        spans_with_time: list[tuple[str, datetime | None, dict[str, Any]]]
    ):
        """Add hierarchical parent-child edges."""
        for span_id, _, span in spans_with_time:
            parent_id = span.get("parent_span_id")
            if parent_id and parent_id in G:
                G.add_edge(parent_id, span_id, type=EdgeType.HIERARCHY.value)

    def _add_data_flow_edges(
        self,
        G: nx.DiGraph,
        spans_with_time: list[tuple[str, datetime | None, dict[str, Any]]],
        order_map: dict[str, int]
    ):
        """Add data flow edges using hybrid approach.

        Uses multiple detection methods:
        1. Call ID matching (LLM ↔ Tool via call_id)
        2. Sibling temporal flow (adjacent siblings under same parent)
        3. Log-based function I/O (function.arguments/output in logs)
        4. Tool call matching (for LLM → TOOL flows)
        5. LLM conversational flow (assistant message → message history)
        6. JSON structural matching (handles wrappers, transformations)
        7. String containment (fallback for string data)
        """
        # Build sibling map for temporal flow detection
        sibling_map = self._get_sibling_spans(spans_with_time)

        # Iterate current node (producer)
        for i, (producer_id, _, producer_span) in enumerate(spans_with_time):
            # Check future nodes (consumers) for data flow
            # We look forward at recent future to see if current output flows into subsequent input

            lookahead_limit = 50
            end_index = min(len(spans_with_time), i + 1 + lookahead_limit)
            possible_consumers = spans_with_time[i+1:end_index]

            for consumer_id, _, consumer_span in possible_consumers:

                # Hybrid data flow check
                flows = self._data_flows_to(
                    producer_span,
                    consumer_span,
                    sibling_map
                )

                # If heuristic says yes and LLM verify is enabled, double-check with LLM
                if flows and self.llm_verify:
                    flows = self._llm_verify_data_flow(producer_span, consumer_span)

                if flows:
                    # If hierarchy edge already exists, mark it as having data flow
                    if G.has_edge(producer_id, consumer_id):
                        G[producer_id][consumer_id]['has_data_flow'] = True
                        logger.debug(f"Marked hierarchy edge as data flow: {producer_id} → {consumer_id}")
                    else:
                        # Create new data edge (not in hierarchy)
                        G.add_edge(producer_id, consumer_id, type=EdgeType.DATA.value)
                        logger.debug(f"New data flow edge: {producer_id} → {consumer_id}")

    def _deduplicate_data_flows_by_type(
        self,
        G: nx.DiGraph,
        span_map: dict[str, Any]
    ):
        """Ensure each node has at most one outgoing data edge per target type.

        This prevents branching where one node's output goes to multiple nodes
        of the same type (e.g., one LLM → multiple LLMs). Keeps data flow linear.

        For each (source, target_type) pair with multiple edges, we keep only
        the first edge by temporal order.
        """
        edges_to_remove = []

        # Group outgoing data edges by (source, target_type)
        source_to_targets: dict[str, dict[str, list[str]]] = {}

        for source, target, edge_data in G.edges(data=True):
            if edge_data.get('type') != EdgeType.DATA.value:
                continue

            # Get target node type
            target_node = G.nodes[target]
            target_attrs = target_node.get('span_attributes', {})
            target_type = target_attrs.get(SpanAttributes.KIND, SpanKind.UNKNOWN.value)

            if source not in source_to_targets:
                source_to_targets[source] = {}
            if target_type not in source_to_targets[source]:
                source_to_targets[source][target_type] = []

            source_to_targets[source][target_type].append(target)

        # For each source with multiple edges to same target type, keep only first
        for source, targets_by_type in source_to_targets.items():
            for target_type, targets in targets_by_type.items():
                if len(targets) > 1:
                    # Keep first by temporal order, remove rest
                    targets_sorted = sorted(targets, key=lambda t: G.nodes[t].get('timestamp', ''))
                    for target in targets_sorted[1:]:
                        edges_to_remove.append((source, target))
                        logger.debug(
                            f"Removing duplicate data edge: {source} → {target} "
                            f"(keeping first {target_type} target: {targets_sorted[0]})"
                        )

        # Remove duplicate edges
        for source, target in edges_to_remove:
            G.remove_edge(source, target)

        if edges_to_remove:
            logger.info(f"Removed {len(edges_to_remove)} duplicate data flow edges")

    def _add_postorder_bubble_up_edges(
        self,
        G: nx.DiGraph,
        span_map: dict[str, Any]
    ):
        """Add bubble-up data flow markers for parent-child edges.

        In agent traces, results flow UPWARD: children produce output that
        gets aggregated into parent's output. This is a postorder pattern
        where children finish before parents.

        For each hierarchy edge (parent → child), check if child's output
        appears in parent's output. If yes, mark as having upward data flow.
        """
        bubble_up_count = 0

        for parent_id, child_id, edge_data in list(G.edges(data=True)):
            # Only process hierarchy edges
            if edge_data.get('type') != EdgeType.HIERARCHY.value:
                continue

            # Already marked? Skip
            if edge_data.get('has_data_flow'):
                continue

            parent_span = span_map.get(parent_id, {})
            child_span = span_map.get(child_id, {})

            # Get end times to verify postorder (child ends before parent)
            parent_end = self._compute_end_time(parent_span)
            child_end = self._compute_end_time(child_span)

            # Only check bubble-up if child ends before parent (postorder)
            if parent_end and child_end and child_end > parent_end:
                continue  # Not postorder - skip

            # Check if child's output appears in parent's output (bubble-up)
            child_output = self._get_output_data(child_span)
            parent_output = self._get_output_data(parent_span)

            if child_output and parent_output:
                if self._content_flows_up(child_output, parent_output):
                    G[parent_id][child_id]['has_data_flow'] = True
                    G[parent_id][child_id]['data_flow_direction'] = 'bubble_up'
                    bubble_up_count += 1
                    logger.debug(
                        f"Bubble-up data flow: {child_id[:8]}... → {parent_id[:8]}..."
                    )

        if bubble_up_count > 0:
            logger.info(f"Detected {bubble_up_count} bubble-up data flows")

    def _add_transitive_bubble_up_edges(
        self,
        G: nx.DiGraph,
        span_map: dict[str, Any]
    ):
        """Add transitive bubble-up edges from deep descendant LLM to ancestor AGENT.

        In agent frameworks like smolagents, the LAST LLM in an agent's subtree
        produces the content that becomes the agent's final answer. This content
        may traverse through intermediate nodes (Steps, Chains) that don't have
        output themselves, but ultimately appears in the AGENT's output.

        For each AGENT node, we:
        1. Find all descendants in its subtree
        2. Identify the LAST LLM (by end time / postorder position)
        3. Check if that LLM's output appears in the AGENT's output
        4. If yes, create a transitive bubble-up edge (LLM → AGENT)
        """
        transitive_count = 0

        # Find all AGENT nodes
        agent_nodes = []
        for node_id, _node_data in G.nodes(data=True):
            span = span_map.get(node_id, {})
            kind = span.get("span_attributes", {}).get(SpanAttributes.KIND)
            if kind == SpanKind.AGENT.value:
                agent_nodes.append(node_id)

        for agent_id in agent_nodes:
            agent_span = span_map.get(agent_id, {})
            agent_output = self._get_output_data(agent_span)

            if not agent_output:
                continue

            # Get all descendants of this agent
            descendants = self._get_all_descendants(G, agent_id)

            if not descendants:
                continue

            # Find the last LLM descendant by end time
            last_llm_id = None
            last_llm_end = None

            for desc_id in descendants:
                desc_span = span_map.get(desc_id, {})
                desc_kind = desc_span.get("span_attributes", {}).get(SpanAttributes.KIND)

                if desc_kind != SpanKind.LLM.value:
                    continue

                desc_end = self._compute_end_time(desc_span)
                if desc_end:
                    if last_llm_end is None or desc_end > last_llm_end:
                        last_llm_id = desc_id
                        last_llm_end = desc_end

            if not last_llm_id:
                continue

            # Skip if already directly connected (direct parent-child)
            # We want to detect TRANSITIVE flows (across N levels)
            parent_id = span_map.get(last_llm_id, {}).get("parent_span_id")
            if parent_id == agent_id:
                continue  # Direct child, handled by regular bubble-up

            # Check if last LLM's output appears in AGENT's output
            llm_span = span_map.get(last_llm_id, {})
            llm_output = self._get_output_data(llm_span)

            if not llm_output:
                continue

            if self._content_flows_up(llm_output, agent_output):
                # Create transitive bubble-up edge (add to graph)
                # This is a NEW edge type: transitive_bubble_up
                if not G.has_edge(last_llm_id, agent_id):
                    G.add_edge(
                        last_llm_id,
                        agent_id,
                        type=EdgeType.TRANSITIVE_BUBBLE_UP.value,
                        has_data_flow=True,
                        data_flow_direction="bubble_up",
                        is_transitive=True
                    )
                    transitive_count += 1
                    logger.info(
                        f"Transitive bubble-up: {last_llm_id[:8]}... (LLM) → {agent_id[:8]}... (AGENT)"
                    )

        if transitive_count > 0:
            logger.info(f"Detected {transitive_count} transitive bubble-up data flows")

    def _add_agent_forward_flow_edges(
        self,
        G: nx.DiGraph,
        span_map: dict[str, Any],
        spans_with_time: list[tuple[str, datetime | None, dict[str, Any]]]
    ):
        """Add forward data flow edges from AGENT nodes to their consumers.

        When an AGENT completes (like a recursive call returning), its output
        flows forward to the next LLM or TOOL that consumes it. This completes
        the data flow picture:

        1. Bubble-up: deep LLM → AGENT (data aggregates up)
        2. Forward flow: AGENT → next consumer (agent's answer flows forward)

        This typically happens at the last chain/step of an agent's execution.
        """
        forward_count = 0

        # Find all AGENT nodes with output
        agent_spans = []
        for span_id, _timestamp, span in spans_with_time:
            kind = span.get("span_attributes", {}).get(SpanAttributes.KIND)
            if kind == SpanKind.AGENT.value:
                output = self._get_output_data(span)
                if output:
                    end_time = self._compute_end_time(span)
                    agent_spans.append((span_id, end_time, span))

        # For each agent, find the next consumer of its output
        for agent_id, agent_end, agent_span in agent_spans:
            agent_output = self._get_output_data(agent_span)
            if not agent_output:
                continue

            # Look for LLM/TOOL nodes that:
            # 1. Start after this agent ends (or overlap in time)
            # 2. Are NOT descendants of this agent (already covered by hierarchy)
            # 3. Have the agent's output in their input
            descendants = self._get_all_descendants(G, agent_id)

            best_consumer = None
            best_consumer_start = None

            for consumer_id, consumer_start, consumer_span in spans_with_time:
                # Skip if same node or descendant
                if consumer_id == agent_id or consumer_id in descendants:
                    continue

                # Only check LLM and TOOL nodes as consumers
                consumer_kind = consumer_span.get("span_attributes", {}).get(SpanAttributes.KIND)
                if consumer_kind not in (SpanKind.LLM.value, SpanKind.TOOL.value):
                    continue

                # Check temporal order: consumer should start after agent ends (or close to it)
                if agent_end and consumer_start:
                    # Allow some overlap (agent might still be "ending" when consumer starts)
                    if consumer_start < agent_end:
                        continue

                # Check if agent's output appears in consumer's input
                consumer_input = self._get_input_data(consumer_span)
                if not consumer_input:
                    continue

                if self._content_flows_forward(agent_output, consumer_input):
                    # Found a consumer - keep the earliest one
                    if best_consumer_start is None or (consumer_start and consumer_start < best_consumer_start):
                        best_consumer = consumer_id
                        best_consumer_start = consumer_start

            # Create forward flow edge if consumer found
            if best_consumer and not G.has_edge(agent_id, best_consumer):
                G.add_edge(
                    agent_id,
                    best_consumer,
                    type=EdgeType.AGENT_FORWARD_FLOW.value,
                    has_data_flow=True,
                    data_flow_direction="forward",
                    is_agent_output=True
                )
                forward_count += 1
                logger.info(
                    f"Agent forward flow: {agent_id[:8]}... (AGENT) → {best_consumer[:8]}... "
                    f"({span_map.get(best_consumer, {}).get('span_attributes', {}).get(SpanAttributes.KIND, '?')})"
                )

        if forward_count > 0:
            logger.info(f"Detected {forward_count} agent forward flow edges")

    def _add_delegation_flow_edges(
        self,
        G: nx.DiGraph,
        span_map: dict[str, Any],
        spans_with_time: list[tuple[str, datetime | None, dict[str, Any]]]
    ):
        """Add delegation flow edges from LLM to sibling AGENT/TOOL nodes.

        This detects the "flow down" pattern where an LLM decides to call an agent
        or tool, and the LLM's output content (the task/parameters) gets passed
        to the sibling agent/tool as input.

        Pattern:
        - Parent CHAIN orchestrates execution
          - LLM child: outputs "call agent X with task Y"
          - AGENT/TOOL child: receives task Y as input

        This is important for error propagation analysis because mistakes in the
        LLM's delegation instructions will propagate to the called agent/tool.
        """
        delegation_count = 0

        # Group spans by parent
        children_by_parent: dict[str, list[tuple[str, datetime | None, dict]]] = {}
        for span_id, timestamp, span in spans_with_time:
            parent_id = span.get('parent_span_id')
            if parent_id:
                if parent_id not in children_by_parent:
                    children_by_parent[parent_id] = []
                children_by_parent[parent_id].append((span_id, timestamp, span))

        # For each parent, look for LLM → AGENT/TOOL delegation patterns
        for _parent_id, children in children_by_parent.items():
            # Separate LLM children from AGENT/TOOL children
            llm_children = []
            agent_tool_children = []

            for span_id, timestamp, span in children:
                kind = span.get('span_attributes', {}).get(SpanAttributes.KIND, '')
                if kind == SpanKind.LLM.value:
                    end_time = self._compute_end_time(span)
                    llm_children.append((span_id, timestamp, end_time, span))
                elif kind in (SpanKind.AGENT.value, SpanKind.TOOL.value):
                    agent_tool_children.append((span_id, timestamp, span))

            # For each LLM, check if its output flows to a subsequent AGENT/TOOL
            for llm_id, _llm_start, llm_end, llm_span in llm_children:
                llm_output = self._get_output_data(llm_span)
                if not llm_output:
                    continue

                for target_id, target_start, target_span in agent_tool_children:
                    # Target should start after LLM (or at least not before)
                    if llm_end and target_start and target_start < llm_end:
                        continue

                    # Check if LLM output appears in target input
                    target_input = self._get_input_data(target_span)
                    if not target_input:
                        continue

                    if self._content_flows_forward(llm_output, target_input):
                        # Create delegation flow edge
                        if not G.has_edge(llm_id, target_id):
                            target_kind = target_span.get('span_attributes', {}).get(
                                SpanAttributes.KIND, '?'
                            )
                            G.add_edge(
                                llm_id,
                                target_id,
                                type=EdgeType.DELEGATION_FLOW.value,
                                has_data_flow=True,
                                data_flow_direction="down",
                                is_delegation=True
                            )
                            delegation_count += 1
                            logger.info(
                                f"Delegation flow: {llm_id[:8]}... (LLM) → "
                                f"{target_id[:8]}... ({target_kind})"
                            )

        if delegation_count > 0:
            logger.info(f"Detected {delegation_count} delegation flow edges")

    def _content_flows_forward(self, producer_output: Any, consumer_input: Any) -> bool:
        """Check if producer's output content appears in consumer's input.

        Similar to _content_flows_up but for forward flow detection.
        Uses the same normalization for consistency.

        For delegation patterns, the consumer input may be a subset of the
        producer output (LLM generates task, agent receives task portion).
        """
        # Extract core content from both
        producer_content = self._extract_core_content(producer_output)
        consumer_content = self._extract_core_content(consumer_input)

        if not producer_content or not consumer_content:
            return False

        # Normalize for comparison
        producer_norm = self._normalize_for_comparison(producer_content)
        consumer_norm = self._normalize_for_comparison(consumer_content)

        # Check direct containment (either direction)
        if producer_norm in consumer_norm:
            return True
        if consumer_norm in producer_norm:
            return True

        # Check significant overlap with prefixes
        producer_prefix = producer_norm[:100] if len(producer_norm) > 100 else producer_norm
        consumer_prefix = consumer_norm[:100] if len(consumer_norm) > 100 else consumer_norm

        if len(producer_prefix) > 20 and producer_prefix in consumer_norm:
            return True
        if len(consumer_prefix) > 20 and consumer_prefix in producer_norm:
            return True

        return False

    def _get_all_descendants(self, G: nx.DiGraph, node_id: str) -> set[str]:
        """Get all descendants of a node in the hierarchy tree.

        Uses BFS to traverse the tree downward following ONLY hierarchy edges.
        Data flow edges are excluded to avoid crossing hierarchy boundaries.
        """
        descendants = set()
        queue = [(node_id, child) for child in G.successors(node_id)]

        while queue:
            parent, current = queue.pop(0)
            if current in descendants:
                continue

            # Only follow hierarchy edges (skip data flow edges)
            edge_data = G.get_edge_data(parent, current)
            if edge_data and edge_data.get('type') != EdgeType.HIERARCHY.value:
                continue  # Skip non-hierarchy edges

            descendants.add(current)

            # Add children of current node
            for child in G.successors(current):
                queue.append((current, child))

        return descendants

    def _content_flows_up(self, child_output: Any, parent_output: Any) -> bool:
        """Check if child's output content appears in parent's output.

        This detects the bubble-up pattern where child results are
        aggregated into parent's final output.
        """
        # Extract core content from both
        child_content = self._extract_core_content(child_output)
        parent_content = self._extract_core_content(parent_output)

        if not child_content or not parent_content:
            return False

        # Normalize for comparison:
        # 1. Escaped newlines/tabs to actual whitespace
        # 2. Curly quotes to straight quotes (Unicode normalization)
        # 3. Normalize multiple whitespace to single space
        child_norm = self._normalize_for_comparison(child_content)
        parent_norm = self._normalize_for_comparison(parent_content)

        # Direct containment (either direction - content may be subset)
        if child_norm in parent_norm:
            return True
        if parent_norm in child_norm:
            return True

        # Check significant overlap (first 100 meaningful chars)
        child_prefix = child_norm[:100] if len(child_norm) > 100 else child_norm
        parent_prefix = parent_norm[:100] if len(parent_norm) > 100 else parent_norm

        if child_prefix in parent_norm or parent_prefix in child_norm:
            return True

        return False

    def _normalize_for_comparison(self, text: str) -> str:
        """Normalize text for content comparison.

        Handles:
        - Escaped newlines/tabs
        - Escaped Unicode sequences (\\u2019 -> ')
        - Curly quotes to straight quotes
        - Python string concatenation artifacts (" ")
        - Multiple whitespace to single space
        """
        result = text

        # Unescape common sequences
        result = result.replace('\\n', '\n').replace('\\t', '\t')

        # Decode escaped Unicode sequences (\\uXXXX -> actual Unicode char)
        # Use regex to selectively decode only escaped sequences
        def decode_unicode_escape(match):
            code = int(match.group(1), 16)
            return chr(code)

        result = re.sub(r'\\u([0-9a-fA-F]{4})', decode_unicode_escape, result)

        # Normalize quotes (curly to straight)
        result = result.replace(''', "'").replace(''', "'")
        result = result.replace('"', '"').replace('"', '"')

        # Remove Python string concatenation artifacts like: " "  or ' '
        # These appear when multi-line strings are normalized
        result = re.sub(r'"\s*"', '', result)
        result = re.sub(r"'\s*'", '', result)

        # Normalize whitespace (multiple spaces/newlines to single space)
        result = ' '.join(result.split())
        return result

    def _extract_core_content(self, output: Any) -> str:
        """Extract core content from output, stripping wrappers.

        Handles:
        - JSON with 'content' or 'text' fields
        - Wrapper prefixes like 'Execution logs:', 'Last output:'
        - Function calls like final_answer(answer="...")
        - Raw strings
        """
        if not output:
            return ""

        output_str = str(output)

        # Try JSON extraction first
        try:
            if output_str.startswith('{'):
                data = json.loads(output_str)
                if isinstance(data, dict):
                    # Extract content field - try multiple common keys
                    # Order matters: more specific fields first
                    for key in ['content', 'text', 'message', 'task']:
                        content = data.get(key)
                        if content and isinstance(content, str):
                            # Recursively extract from content (may have nested wrappers)
                            return self._extract_core_content(content)
        except (json.JSONDecodeError, TypeError):
            pass

        # Strip common wrapper prefixes
        prefixes_to_strip = [
            'Execution logs:\n',
            'Last output from code snippet:\n',
            'Output:\n',
        ]
        result = output_str
        for prefix in prefixes_to_strip:
            if result.startswith(prefix):
                result = result[len(prefix):]

        # Extract task content from agent task format:
        # "...---\nTask:\n<actual task>\n---\n..."
        task_match = re.search(r'---\s*\nTask:\s*\n(.+?)\n---', result, re.DOTALL)
        if task_match:
            return task_match.group(1).strip()

        # Extract from function calls like final_answer(answer="...")
        answer_match = re.search(r'final_answer\s*\(\s*answer\s*=\s*["\'](.+?)["\'](?:\s*\)|\s*,)', result, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()

        # Also try extracting content after "### 1." or similar markdown headers
        # But only if it's the primary content (not just a template)
        if result.strip().startswith('###'):
            header_match = re.search(r'(###\s*\d+\..*)', result, re.DOTALL)
            if header_match:
                return header_match.group(1).strip()

        return result.strip()

    def _get_output_data(self, span: dict[str, Any]) -> Any | None:
        """Extract output data from span."""
        attrs = span.get("span_attributes", {})
        return attrs.get(SpanAttributes.OUTPUT_VALUE)

    def _get_input_data(self, span: dict[str, Any]) -> Any | None:
        """Extract input data from span."""
        attrs = span.get("span_attributes", {})
        return attrs.get(SpanAttributes.INPUT_VALUE)

    def _has_output(self, span: dict[str, Any]) -> bool:
        """Check if span has output value."""
        return self._get_output_data(span) is not None

    def _has_input(self, span: dict[str, Any]) -> bool:
        """Check if span has input value."""
        return self._get_input_data(span) is not None

    def _data_flows_to(
        self,
        producer_span: dict[str, Any],
        consumer_span: dict[str, Any],
        sibling_map: dict[str, list[tuple[str, datetime | None, dict[str, Any]]]]
    ) -> bool:
        """Hybrid check: does producer output flow to consumer input?

        Uses multiple detection methods in order of reliability:
        1. Call ID matching (explicit LLM ↔ Tool linking via call_id)
        2. Sibling temporal flow (adjacent siblings under same parent)
        3. Log-based function I/O (function.arguments/output from logs)
        4. Tool call matching (LLM tool_calls → Tool input)
        5. LLM conversational flow (assistant message → message history)
        6. JSON structural matching (handles wrappers)
        7. String containment (fallback)
        """
        # ONLY LLM and TOOL nodes process data
        # CHAIN, AGENT, RETRIEVER, EMBEDDING are just orchestrators/infrastructure
        producer_kind = producer_span.get("span_attributes", {}).get(SpanAttributes.KIND)
        consumer_kind = consumer_span.get("span_attributes", {}).get(SpanAttributes.KIND)

        if producer_kind not in (SpanKind.LLM.value, SpanKind.TOOL.value) or consumer_kind not in (SpanKind.LLM.value, SpanKind.TOOL.value):
            return False

        # Check 1: Call ID matching (most explicit link we can infer)
        if self._call_id_flows(producer_span, consumer_span):
            logger.debug("Matched via call_id flow")
            return True

        # Check 2: Sibling temporal flow (same parent, adjacent in time)
        if self._sibling_temporal_flows(producer_span, consumer_span, sibling_map):
            logger.debug("Matched via sibling temporal flow")
            return True

        # Check 3: Log-based function I/O
        if self._log_function_io_flows(producer_span, consumer_span):
            logger.debug("Matched via log function I/O")
            return True

        # Get span attribute-based I/O for remaining checks
        producer_output = self._get_output_data(producer_span)
        consumer_input = self._get_input_data(consumer_span)

        # Skip if no span attribute I/O available
        if producer_output is None or consumer_input is None:
            return False

        # Check 4a: Tool call matching (LLM → TOOL)
        if self._tool_calls_match(producer_output, consumer_input):
            logger.debug("Matched via tool_calls (LLM → TOOL)")
            return True

        # Check 4b: Code-based tool call matching (LLM → TOOL)
        if self._code_tool_call_match(producer_output, consumer_input, producer_span, consumer_span):
            logger.debug("Matched via code tool call (LLM → TOOL)")
            return True

        # Check 4c: Tool result matching (TOOL → LLM)
        if self._tool_result_flows_to_llm(producer_output, consumer_input, producer_span, consumer_span):
            logger.debug("Matched via tool result (TOOL → LLM)")
            return True

        # Check 5: LLM conversational flow (LLM → LLM)
        if self._llm_conversation_flows(producer_output, consumer_input):
            logger.debug("Matched via LLM conversation flow")
            return True

        # Check 6: JSON structural matching
        if self._json_structures_match(producer_output, consumer_input):
            logger.debug("Matched via JSON structure")
            return True

        # Check 7: Exact equality
        if producer_output == consumer_input:
            logger.debug("Matched via exact equality")
            return True

        # Check 8: String containment (with normalization)
        if self._string_contains(producer_output, consumer_input):
            logger.debug("Matched via string containment")
            return True

        return False

    def _tool_calls_match(self, producer_output: Any, consumer_input: Any) -> bool:
        """Check if producer's tool calls appear in consumer's input.

        For LLM → TOOL flows where:
        - LLM outputs tool_calls (producer)
        - TOOL span receives the arguments (consumer)
        """
        try:
            # Parse outputs as JSON if needed
            prod_data = self._parse_json(producer_output)
            cons_data = self._parse_json(consumer_input)

            # Extract tool calls from producer
            tool_calls = None
            if isinstance(prod_data, dict):
                tool_calls = prod_data.get("tool_calls")

            if not tool_calls or not isinstance(tool_calls, list):
                return False

            # Check each tool call
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue

                # Case 1: Direct match - consumer input has same tool_calls
                if isinstance(cons_data, dict):
                    if cons_data.get("tool_calls") == tool_calls:
                        return True

                # Case 2: Function arguments match
                func = tc.get("function")
                if func and isinstance(func, dict):
                    func_args = func.get("arguments")
                    func_name = func.get("name")

                    # Try parsing function arguments if it's a string
                    if isinstance(func_args, str):
                        try:
                            func_args = json.loads(func_args)
                        except (json.JSONDecodeError, TypeError):
                            pass

                    # Check if arguments appear in consumer input
                    if func_args:
                        # Direct arguments match
                        if cons_data == func_args:
                            return True
                        # Consumer input contains the arguments
                        if isinstance(cons_data, dict) and self._contains_structure(cons_data, func_args):
                            return True
                        # Consumer input wrapped with function name
                        if isinstance(cons_data, dict) and func_name:
                            cons_args = cons_data.get("arguments") or cons_data.get(func_name)
                            if cons_args and self._contains_structure(cons_args, func_args):
                                return True

                # Case 3: Tool call ID match
                call_id = tc.get("id")
                if call_id and isinstance(cons_data, str):
                    if call_id in cons_data:
                        return True

            return False
        except Exception:
            return False

    def _code_tool_call_match(
        self,
        producer_output: Any,
        consumer_input: Any,
        producer_span: dict[str, Any],
        consumer_span: dict[str, Any]
    ) -> bool:
        """Check if LLM output contains code calling a tool that matches consumer input.

        For agent frameworks (HuggingFace, LangChain) where tools are called via code:
        - LLM outputs code like: final_answer("33149") or search_tool(query="test")
        - TOOL span receives: {"args": ["33149"], "kwargs": {}} or similar

        Example:
            Producer (LLM): {"content": "```py\nfinal_answer('33149')\n```"}
            Consumer (TOOL): {"args": ["33149"], "kwargs": {}}
        """
        try:
            # Only check LLM → TOOL flows
            producer_kind = producer_span.get("span_attributes", {}).get(SpanAttributes.KIND)
            consumer_kind = consumer_span.get("span_attributes", {}).get(SpanAttributes.KIND)

            if producer_kind != SpanKind.LLM.value or consumer_kind != SpanKind.TOOL.value:
                return False

            # Parse producer output
            prod_data = self._parse_json(producer_output)
            if not isinstance(prod_data, dict):
                return False

            # Get content field from LLM output
            content = prod_data.get("content", "")
            if not content or not isinstance(content, str):
                return False

            # Parse consumer input
            cons_data = self._parse_json(consumer_input)
            if not isinstance(cons_data, dict):
                return False

            # Get tool name from consumer span attributes
            tool_attrs = consumer_span.get("span_attributes", {})
            tool_name = tool_attrs.get(SpanAttributes.TOOL_NAME) or consumer_span.get("span_name", "")

            # Look for tool calls in code blocks
            # Pattern: tool_name("arg1", "arg2") or tool_name(arg1="val1", arg2="val2")
            import re

            # Extract args from consumer
            consumer_args = cons_data.get("args", [])
            consumer_kwargs = cons_data.get("kwargs", {})

            if not consumer_args and not consumer_kwargs:
                return False

            # Search for function calls in content
            # Match: function_name(...) with any content inside
            if tool_name:
                # Pattern 1: Check if tool name appears followed by parentheses
                pattern = rf"{re.escape(tool_name)}\s*\("
                if not re.search(pattern, content):
                    return False

                # Pattern 2: Extract arguments from code
                # Match: tool_name("arg") or tool_name('arg') or tool_name(arg)
                for arg in consumer_args:
                    arg_str = str(arg)
                    # Look for this argument value in the content near the tool call
                    # Check both quoted and unquoted versions
                    if f'"{arg_str}"' in content or f"'{arg_str}'" in content or f"({arg_str})" in content:
                        logger.debug(f"Found code tool call: {tool_name}(...{arg_str}...)")
                        return True

                # Check kwargs
                for key, value in consumer_kwargs.items():
                    # Look for key=value pattern
                    value_str = str(value)
                    if f'{key}=' in content and (f'"{value_str}"' in content or f"'{value_str}'" in content):
                        logger.debug(f"Found code tool call: {tool_name}({key}={value_str})")
                        return True

            return False

        except Exception as e:
            logger.debug(f"Code tool call match failed: {e}")
            return False

    def _tool_result_flows_to_llm(
        self,
        producer_output: Any,
        consumer_input: Any,
        producer_span: dict[str, Any],
        consumer_span: dict[str, Any]
    ) -> bool:
        """Check if TOOL output flows to LLM input.

        Handles agent pattern where:
        - TOOL span produces output
        - LLM receives it in messages array as a tool/function message

        Example:
            Producer (TOOL): {"result": "42"}
            Consumer (LLM): {
                "messages": [
                    {"role": "assistant", "tool_calls": [...]},
                    {"role": "tool", "content": "42", "tool_call_id": "call_1"}
                ]
            }
        """
        try:
            # Check if consumer is an LLM span
            consumer_kind = consumer_span.get("span_attributes", {}).get(SpanAttributes.KIND)
            if consumer_kind != SpanKind.LLM.value:
                return False

            # Parse consumer input
            cons_data = self._parse_json(consumer_input)
            if not isinstance(cons_data, dict):
                return False

            # Check if consumer has messages array
            messages = cons_data.get("messages", [])
            if not isinstance(messages, list):
                return False

            # Normalize producer output to string for comparison
            prod_str = self._normalize_to_string(producer_output)
            if len(prod_str) < 5:  # Skip very short outputs
                return False

            # Look for tool/function messages in consumer's message history
            for msg in messages:
                if not isinstance(msg, dict):
                    continue

                msg_role = msg.get("role")

                # Check tool role messages
                if msg_role in ("tool", "function"):
                    msg_content = msg.get("content") or msg.get("tool_output") or msg.get("output")

                    if msg_content:
                        # Try exact match
                        if msg_content == producer_output:
                            return True

                        # Try string containment
                        msg_str = self._normalize_to_string(msg_content)
                        if prod_str in msg_str or msg_str in prod_str:
                            return True

                        # Try JSON structural match
                        if self._json_structures_match(producer_output, msg_content):
                            return True

            return False
        except Exception:
            return False

    def _llm_conversation_flows(self, producer_output: Any, consumer_input: Any) -> bool:
        """Check if LLM output appears in next LLM call's message history.

        For LLM traces where one model's output (assistant message) is embedded
        in the next model call's messages array.
        """
        try:
            # 1. Prepare Producer Content
            prod_data = self._parse_json(producer_output)
            prod_content = None

            if isinstance(prod_data, dict):
                # Typically producer is an assistant message or just partial output
                prod_role = prod_data.get("role")
                if prod_role == "assistant":
                    prod_content = prod_data.get("content")
                # Sometimes producer output is just the content string if not wrapped
                elif not prod_role:
                     # Check if it looks like content? 
                     prod_content = prod_data.get("content")
            elif isinstance(prod_data, str):
                prod_content = prod_data
            
            if not prod_content:
                return False

            # Normalize producer content (remove spaces/newlines)
            norm_prod_content = self._normalize_strict(prod_content)
            if not norm_prod_content:
                return False

            # 2. Prepare Consumer Messages
            cons_data = self._parse_json(consumer_input)
            if not isinstance(cons_data, dict):
                return False

            messages = cons_data.get("messages", [])
            if not isinstance(messages, list):
                return False

            # Parameterized history check
            check_count = self.max_history_check
            recent_messages = messages[-check_count:] if len(messages) >= check_count else messages

            # 3. Compare against Consumer Messages
            for msg in recent_messages:
                if not isinstance(msg, dict):
                    continue

                msg_content = msg.get("content")
                
                # Handle list of content blocks (e.g. OpenAI vision/multimodal)
                if isinstance(msg_content, list):
                    # Join text blocks to form the full message content
                    texts = []
                    for item in msg_content:
                        if isinstance(item, dict) and "text" in item:
                            texts.append(str(item["text"]))
                    msg_content = "".join(texts)

                norm_msg_content = self._normalize_strict(msg_content)
                
                if norm_msg_content and norm_prod_content in norm_msg_content:
                    return True

            return False
        except Exception:
            return False

    def _normalize_strict(self, text: Any) -> str:
        """Normalize text by removing all whitespace and newlines."""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        return re.sub(r'\s+', '', text)

    def _json_structures_match(self, producer_output: Any, consumer_input: Any) -> bool:
        """Check if producer output appears within consumer input structure.

        Handles cases where data is wrapped in objects or transformed.
        """
        try:
            prod_data = self._parse_json(producer_output)
            cons_data = self._parse_json(consumer_input)

            if prod_data is None or cons_data is None:
                return False

            # Check if producer data is contained in consumer data structure
            return self._contains_structure(cons_data, prod_data)
        except Exception:
            return False

    def _contains_structure(self, container: Any, target: Any) -> bool:
        """Recursively check if target structure appears in container."""
        # Direct match
        if container == target:
            return True

        # If container is dict, check values
        if isinstance(container, dict):
            for value in container.values():
                if self._contains_structure(value, target):
                    return True

        # If container is list, check elements
        if isinstance(container, list):
            for item in container:
                if self._contains_structure(item, target):
                    return True

        return False

    def _string_contains(self, producer_output: Any, consumer_input: Any) -> bool:
        """Check if producer output string is contained in consumer input.

        Fallback for string-based data flow.
        """
        try:
            prod_str = self._normalize_to_string(producer_output)
            cons_str = self._normalize_to_string(consumer_input)

            if not prod_str or not cons_str:
                return False

            # Minimum length threshold to avoid false positives
            if len(prod_str) < 10:
                return False

            return prod_str in cons_str
        except Exception:
            return False

    def _parse_json(self, data: Any) -> Any | None:
        """Parse data as JSON if it's a string, otherwise return as-is."""
        if isinstance(data, str):
            try:
                return json.loads(data)
            except (json.JSONDecodeError, TypeError):
                return None
        return data

    def _normalize_to_string(self, data: Any) -> str:
        """Normalize data to string for comparison."""
        if isinstance(data, str):
            return data.strip()
        if isinstance(data, (dict, list)):
            try:
                return json.dumps(data, sort_keys=True)
            except (TypeError, ValueError):
                return str(data)
        return str(data) if data is not None else ""

    # =========================================================================
    # HYBRID DATAFLOW METHODS
    # =========================================================================

    def _extract_call_ids_from_span(self, span: dict[str, Any]) -> set[str]:
        """Extract all call IDs referenced in a span's LLM messages.

        Looks for patterns like:
        - "Call id: call_2" in message content
        - "'id': 'call_2'" in tool call definitions
        """
        call_ids = set()
        attrs = span.get("span_attributes", {})

        # Search through all llm.input_messages and llm.output_messages
        for key, value in attrs.items():
            if "message.content" in key and isinstance(value, str):
                # Find "Call id: call_X" patterns
                for match in CALL_ID_PATTERN.findall(value):
                    call_ids.add(match)
                # Find tool call definitions with IDs
                for match in TOOL_CALL_ID_PATTERN.findall(value):
                    call_ids.add(match)

        # Also check input.value for call IDs
        input_val = attrs.get(SpanAttributes.INPUT_VALUE, "")
        if isinstance(input_val, str):
            for match in CALL_ID_PATTERN.findall(input_val):
                call_ids.add(match)
            for match in TOOL_CALL_ID_PATTERN.findall(input_val):
                call_ids.add(match)

        return call_ids

    def _extract_tool_call_from_output(self, span: dict[str, Any]) -> str | None:
        """Extract the call ID from an LLM's tool call output.

        When an LLM outputs a tool call, extract the call ID so we can
        link it to the tool execution span.
        """
        attrs = span.get("span_attributes", {})
        output_val = attrs.get(SpanAttributes.OUTPUT_VALUE, "")

        if isinstance(output_val, str):
            # Try parsing as JSON first
            try:
                data = json.loads(output_val)
                if isinstance(data, dict):
                    tool_calls = data.get("tool_calls")
                    if tool_calls and isinstance(tool_calls, list):
                        for tc in tool_calls:
                            if isinstance(tc, dict) and "id" in tc:
                                return tc["id"]
            except json.JSONDecodeError:
                pass

            # Fallback to regex
            match = TOOL_CALL_ID_PATTERN.search(output_val)
            if match:
                return match.group(1)

        return None

    def _call_id_flows(
        self,
        producer_span: dict[str, Any],
        consumer_span: dict[str, Any]
    ) -> bool:
        """Check if producer's tool call ID is referenced in consumer.

        Handles LLM → Tool flow where LLM outputs call_id and tool
        execution is tracked, then result flows back to next LLM call.
        """
        # Case 1: LLM outputs tool call → Tool span executes it
        producer_call_id = self._extract_tool_call_from_output(producer_span)
        if producer_call_id:
            consumer_call_ids = self._extract_call_ids_from_span(consumer_span)
            if producer_call_id in consumer_call_ids:
                return True

        # Case 2: Tool span result → LLM input references the call ID
        # Check if producer is a tool and consumer LLM references the same call context
        producer_kind = producer_span.get("span_attributes", {}).get(SpanAttributes.KIND)
        consumer_kind = consumer_span.get("span_attributes", {}).get(SpanAttributes.KIND)

        if producer_kind == SpanKind.TOOL.value and consumer_kind == SpanKind.LLM.value:
            # Check if producer's parent LLM call ID is referenced in consumer
            consumer_call_ids = self._extract_call_ids_from_span(consumer_span)
            # The tool result observation will reference the call ID
            if consumer_call_ids:
                return True  # Tool result likely flows to next LLM

        return False

    def _get_sibling_spans(
        self,
        spans_with_time: list[tuple[str, datetime | None, dict[str, Any]]]
    ) -> dict[str, list[tuple[str, datetime | None, dict[str, Any]]]]:
        """Group spans by parent, returning dict of parent_id -> [child spans]."""
        siblings: dict[str, list] = {}

        for span_id, timestamp, span in spans_with_time:
            parent_id = span.get("parent_span_id")
            if parent_id:
                if parent_id not in siblings:
                    siblings[parent_id] = []
                siblings[parent_id].append((span_id, timestamp, span))

        # Sort each group by timestamp
        for parent_id in siblings:
            siblings[parent_id].sort(key=lambda x: x[1] if x[1] else datetime.min)

        return siblings

    def _sibling_temporal_flows(
        self,
        producer_span: dict[str, Any],
        consumer_span: dict[str, Any],
        sibling_map: dict[str, list[tuple[str, datetime | None, dict[str, Any]]]]
    ) -> bool:
        """Check if producer and consumer are adjacent siblings (temporal flow).

        Siblings under the same parent, where producer immediately precedes
        consumer, implies dataflow.
        """
        producer_id = producer_span.get("span_id")
        consumer_id = consumer_span.get("span_id")
        producer_parent = producer_span.get("parent_span_id")
        consumer_parent = consumer_span.get("parent_span_id")

        # Must have same parent
        if not producer_parent or producer_parent != consumer_parent:
            return False

        siblings = sibling_map.get(producer_parent, [])

        # Find positions
        producer_idx = None
        consumer_idx = None
        for idx, (span_id, _, _) in enumerate(siblings):
            if span_id == producer_id:
                producer_idx = idx
            if span_id == consumer_id:
                consumer_idx = idx

        # Check if consumer immediately follows producer
        if producer_idx is not None and consumer_idx is not None:
            if consumer_idx == producer_idx + 1:
                return True

        return False

    def _log_function_io_flows(
        self,
        producer_span: dict[str, Any],
        consumer_span: dict[str, Any]
    ) -> bool:
        """Check dataflow via log-based function.arguments and function.output.

        Spans may have logs with:
        - body.function.arguments: inputs
        - body.function.output: outputs

        If producer's function.output appears in consumer's function.arguments,
        there's dataflow.
        """
        producer_output = self._get_log_function_output(producer_span)
        consumer_input = self._get_log_function_arguments(consumer_span)

        if producer_output is None or consumer_input is None:
            return False

        # Skip null outputs
        if producer_output == "<null>" or producer_output is None:
            return False

        # Check if output appears in input (structural or string match)
        if self._json_structures_match(producer_output, consumer_input):
            return True

        if self._string_contains(str(producer_output), str(consumer_input)):
            return True

        return False

    def _get_log_function_output(self, span: dict[str, Any]) -> Any | None:
        """Extract function.output from span's logs."""
        logs = span.get("logs", [])
        for log in logs:
            body = log.get("body", {})
            if isinstance(body, dict) and "function.output" in body:
                return body["function.output"]
        return None

    def _get_log_function_arguments(self, span: dict[str, Any]) -> Any | None:
        """Extract function.arguments from span's logs."""
        logs = span.get("logs", [])
        for log in logs:
            body = log.get("body", {})
            if isinstance(body, dict) and "function.arguments" in body:
                return body["function.arguments"]
        return None

    def _llm_verify_data_flow(
        self,
        producer_span: dict[str, Any],
        consumer_span: dict[str, Any]
    ) -> bool:
        """Use LLM to verify if data flows from producer to consumer.

        Based on archive's approach with caching for efficiency.
        """
        producer_output = self._get_output_data(producer_span)
        consumer_input = self._get_input_data(consumer_span)

        if producer_output is None or consumer_input is None:
            return False

        # Create cache key
        prod_str = str(producer_output)[:500]
        cons_str = str(consumer_input)[:500]
        cache_key = f"{hash(prod_str)}:{hash(cons_str)}"

        # Check cache
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        # Truncate for LLM
        prod_truncated = prod_str[:1000]
        cons_truncated = cons_str[:1000]

        prompt = f"""You must decide if Text A (producer output) is used as Text B (consumer input) as the SAME concrete data.

**Analysis Rules:**
- Treat wrappers like {{args:[]}}, {{kwargs:{{...}}}}, {{sanitize_inputs_outputs:true}} as NON-semantic wrappers.
- Consider paraphrases and JSON key renames, but DO NOT accept topical similarity alone.
- REQUIRE that specific literals (queries, responses, URLs, IDs, argument values) from A are preserved in B.
- If A = tool_calls[].function.arguments, compare those arguments with B's input arguments (possibly under kwargs).

Return STRICT JSON only:
{{ "same_data": true|false, "reasoning": "brief explanation" }}

**Text A (producer output):**
{prod_truncated}

**Text B (consumer input):**
{cons_truncated}
"""

        try:
            # Initialize LLM provider if needed
            if self.llm_provider is None:
                from src.llm import Message, OpenAIClient
                import os
                self.llm_provider = OpenAIClient(model=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"))

            from src.llm import Message

            result = self.llm_provider.complete_json(
                messages=[Message(role="user", content=prompt)],
                temperature=0.0,
                max_completion_tokens=200
            )

            same_data = bool(result.get("same_data", False))
            self._llm_cache[cache_key] = same_data

            if same_data:
                logger.debug(f"LLM confirmed data flow: {result.get('reasoning', '')}")
            else:
                logger.debug(f"LLM rejected data flow: {result.get('reasoning', '')}")

            return same_data

        except Exception as e:
            logger.warning(f"LLM verification failed: {e}, falling back to heuristic result")
            # On error, trust the heuristic
            return True
