"""
Plan DAG construction and validation.

Represents a finalized agent plan as a Directed Acyclic Graph (DAG) using NetworkX.
Nodes are plan steps; edges encode dependencies between steps.
"""

import logging
from typing import Any

import networkx as nx

from src.progress_monitor.plan_extractor import PlanStep

logger = logging.getLogger(__name__)


class PlanDAG:
    """DAG representation of a finalized agent plan.

    Each plan step is a node in the graph. Edges represent dependencies:
    an edge (A -> B) means step A must complete before step B can begin.

    Dependency Resolution:
    - If a step has explicit ``depends_on`` (from LLM extraction detecting
      parallelism keywords like "in parallel", "independently"), those are used.
    - Otherwise, defaults to sequential chain: step i depends on step i-1.

    Node attributes:
        description (str): Human-readable description of the step.
        completed (bool): Whether this step has been completed (used during tracking).
        progress_credit (float): Weight of this step = 1/N where N is total steps.
    """

    def __init__(self, steps: list[PlanStep]) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self._build(steps)

    def _build(self, steps: list[PlanStep]) -> None:
        """Build DAG from plan steps.

        Args:
            steps: Finalized plan steps with optional depends_on.
        """
        if not steps:
            return

        n = len(steps)
        for step in steps:
            self.graph.add_node(
                step.step_number,
                description=step.description,
                completed=False,
                progress_credit=1.0 / n,
            )

        for step in steps:
            if step.depends_on:
                # Explicit dependencies from LLM extraction
                for dep in step.depends_on:
                    if dep in self.graph:
                        self.graph.add_edge(dep, step.step_number)
                    else:
                        logger.warning(
                            f"Step {step.step_number} depends on nonexistent "
                            f"step {dep}, ignoring edge"
                        )
            else:
                # Default: sequential (step i depends on step i-1)
                if step.step_number > 1 and (step.step_number - 1) in self.graph:
                    self.graph.add_edge(step.step_number - 1, step.step_number)

    def validate(self) -> list[str]:
        """Check plan DAG for logical errors.

        Returns:
            List of validation issue strings. Empty list means the DAG is valid.

        Checks performed:
            - Cycles (loops): the graph must be a DAG
            - Disconnected components: all nodes should be reachable
            - Empty graph
        """
        issues: list[str] = []

        if self.graph.number_of_nodes() == 0:
            issues.append("Plan DAG is empty (no steps)")
            return issues

        if not nx.is_directed_acyclic_graph(self.graph):
            cycles = list(nx.simple_cycles(self.graph))
            issues.append(f"Plan contains cycles: {cycles}")

        # Check for disconnected components
        undirected = self.graph.to_undirected()
        components = list(nx.connected_components(undirected))
        if len(components) > 1:
            issues.append(
                f"Plan has {len(components)} disconnected components: {components}"
            )

        return issues

    @property
    def total_steps(self) -> int:
        """Total number of steps in the plan."""
        return self.graph.number_of_nodes()

    def get_step_descriptions(self) -> dict[int, str]:
        """Get mapping of step_number -> description for all steps."""
        return {
            node: data["description"]
            for node, data in self.graph.nodes(data=True)
        }

    def get_step_credit(self, step_number: int) -> float:
        """Get the progress credit for a specific step."""
        if step_number not in self.graph:
            return 0.0
        return self.graph.nodes[step_number]["progress_credit"]

    def get_ready_steps(self, completed: set[int]) -> list[int]:
        """Get steps whose dependencies are all satisfied.

        Args:
            completed: Set of step numbers already completed.

        Returns:
            List of step numbers that are ready to execute.
        """
        ready: list[int] = []
        for node in self.graph.nodes():
            if node in completed:
                continue
            predecessors = set(self.graph.predecessors(node))
            if predecessors.issubset(completed):
                ready.append(node)
        return sorted(ready)

    def get_topological_order(self) -> list[int]:
        """Get steps in topological order (respecting dependencies).

        Returns:
            List of step numbers in valid execution order.

        Raises:
            nx.NetworkXUnfeasible: If the graph contains a cycle.
        """
        return list(nx.topological_sort(self.graph))

    def get_ancestors(self, step_number: int) -> set[int]:
        """Get all transitive dependencies (ancestors) of a step.

        If step 3 depends on step 2 which depends on step 1,
        get_ancestors(3) returns {1, 2}.
        """
        if step_number not in self.graph:
            return set()
        return nx.ancestors(self.graph, step_number)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the DAG to a dictionary for JSON output."""
        nodes = []
        for node, data in self.graph.nodes(data=True):
            nodes.append(
                {
                    "step_number": node,
                    "description": data["description"],
                    "progress_credit": data["progress_credit"],
                    "predecessors": list(self.graph.predecessors(node)),
                    "successors": list(self.graph.successors(node)),
                }
            )
        return {
            "total_steps": self.total_steps,
            "nodes": nodes,
            "edges": list(self.graph.edges()),
        }
