"""
Graph preprocessing utilities for failure analysis.

Provides filtering and extraction operations on NetworkX graphs built from traces.
"""

import networkx as nx
from typing import Any
import logging

from src.graph.constants import EdgeType
from src.graph.graph_builder import GraphBuilder
from src.utils.trace_utils import build_span_map, extract_task_description

logger = logging.getLogger(__name__)


class GraphPreprocessor:
    """Preprocesses trace graphs for failure analysis."""

    def __init__(self, excluded_node_names: list[str] | None = None):
        self.excluded_node_names = set(excluded_node_names or [])
        self.data_flow_types = EdgeType.data_flow_types()
        self._graph_builder = GraphBuilder()

    def build_and_filter(
        self, trace: dict[str, Any],
    ) -> tuple[nx.DiGraph, dict[str, Any], list[tuple[str, dict[str, Any]]], str]:
        """
        Full pipeline: build graph → remove isolated → remove excluded → extract spans.

        Returns:
            (graph, span_map, nodes_and_spans, task_description)
        """
        span_map = build_span_map(trace)

        G, _ = self._graph_builder.build_from_trace(
            trace, include_hierarchy=False, include_data_flow=True,
        )
        logger.debug(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Remove isolated nodes (AGENT/CHAIN with no data-flow edges)
        isolated = [n for n in G.nodes() if G.degree(n) == 0]
        G.remove_nodes_from(isolated)
        logger.debug(f"Removed {len(isolated)} isolated nodes, {G.number_of_nodes()} remaining")

        G, span_map = self.remove_excluded_nodes(G, span_map)
        nodes_and_spans = self.extract_nodes_and_spans(G, span_map)
        task_desc = extract_task_description(span_map)

        return G, span_map, nodes_and_spans, task_desc

    def filter_to_data_flow_only(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        Create a new graph containing only data flow edges.

        Removes all hierarchy edges, keeping only edges that represent
        data dependencies between spans.

        Args:
            G: Original graph with all edge types

        Returns:
            New graph with only data flow edges
        """
        # Create new graph
        G_filtered = nx.DiGraph()

        # Copy all nodes with their attributes
        for node, data in G.nodes(data=True):
            G_filtered.add_node(node, **data)

        # Copy only data flow edges
        for u, v, data in G.edges(data=True):
            edge_type = data.get('type')

            # Check if this is a data flow edge
            if edge_type and self._is_data_flow_edge(edge_type):
                G_filtered.add_edge(u, v, **data)

        return G_filtered

    def _is_data_flow_edge(self, edge_type: str) -> bool:
        """
        Check if an edge type represents data flow.

        Args:
            edge_type: Edge type string

        Returns:
            True if edge represents data flow
        """
        try:
            edge_enum = EdgeType(edge_type)
            return edge_enum in self.data_flow_types
        except ValueError:
            # If not a recognized EdgeType, check for common data flow indicators
            return edge_type in {'data', 'delegation_flow', 'agent_forward_flow',
                               'transitive_bubble_up'}

    def remove_excluded_nodes(
        self,
        G: nx.DiGraph,
        span_map: dict[str, Any]
    ) -> tuple[nx.DiGraph, dict[str, Any]]:
        """
        Remove nodes with excluded span names from graph and span map.

        Args:
            G: Graph to filter
            span_map: Mapping from span_id to span data

        Returns:
            Tuple of (filtered_graph, filtered_span_map)
        """
        if not self.excluded_node_names:
            return G, span_map

        # Identify nodes to remove
        nodes_to_remove = set()
        for node_id in G.nodes():
            if node_id in span_map:
                span_name = span_map[node_id].get('span_name', '')
                if span_name in self.excluded_node_names:
                    nodes_to_remove.add(node_id)

        # Create filtered graph
        G_filtered = G.copy()
        G_filtered.remove_nodes_from(nodes_to_remove)

        # Create filtered span map
        span_map_filtered = {
            node_id: span_data
            for node_id, span_data in span_map.items()
            if node_id not in nodes_to_remove
        }

        return G_filtered, span_map_filtered

    def extract_nodes_and_spans(
        self,
        G: nx.DiGraph,
        span_map: dict[str, Any]
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        Extract list of (node_id, span_data) pairs for all nodes in graph.

        Args:
            G: Graph to extract from
            span_map: Mapping from span_id to span data

        Returns:
            List of (node_id, span_data) tuples
        """
        nodes_and_spans = []
        for node_id in G.nodes():
            if node_id in span_map:
                nodes_and_spans.append((node_id, span_map[node_id]))

        return nodes_and_spans

    def detect_leaf_nodes(self, G: nx.DiGraph) -> list[str]:
        """
        Detect leaf nodes (nodes with no outgoing edges).

        In data flow graphs, leaf nodes represent terminal operations
        that don't pass data to any other nodes. These are good starting
        points for backtracking analysis.

        Args:
            G: Graph to analyze

        Returns:
            List of node IDs with out_degree == 0
        """
        return [node for node in G.nodes() if G.out_degree(node) == 0]

    def detect_source_nodes(self, G: nx.DiGraph) -> list[str]:
        """
        Detect source nodes (nodes with no incoming edges).

        In data flow graphs, source nodes represent initial operations
        that don't receive data from any other nodes.

        Args:
            G: Graph to analyze

        Returns:
            List of node IDs with in_degree == 0
        """
        return [node for node in G.nodes() if G.in_degree(node) == 0]

    def get_predecessors(self, G: nx.DiGraph, node_id: str) -> list[str]:
        """
        Get all predecessor nodes (nodes with edges pointing to this node).

        Args:
            G: Graph to query
            node_id: Target node ID

        Returns:
            List of predecessor node IDs
        """
        return list(G.predecessors(node_id))

    def get_successors(self, G: nx.DiGraph, node_id: str) -> list[str]:
        """
        Get all successor nodes (nodes this node points to).

        Args:
            G: Graph to query
            node_id: Source node ID

        Returns:
            List of successor node IDs
        """
        return list(G.successors(node_id))
