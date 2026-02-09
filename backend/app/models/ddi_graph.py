from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, List, Sequence, Set, Tuple


@dataclass
class DDIGraph:
    """Very small in-memory DDI graph.

    In the work this class is the place where you would plug in a real
    drug-drug interaction knowledge base (e.g. TWOSIDES or a hospital-specific
    rules engine). For the backend demo we only keep a tiny list of risky pairs
    so that the API can illustrate how DDI checks are incorporated.
    """

    risky_pairs: Set[Tuple[str, str]]

    @classmethod
    def build_demo(cls, drugs: Sequence[str]) -> "DDIGraph":
        """Create a toy graph using a handful of the available drug names.

        We simply flag a few arbitrary pairs among the first 10 distinct drugs.
        The choice of pairs has no clinical meaning and is only for demonstration.
        """
        upper = [d.upper() for d in drugs[:10]]
        pairs: Set[Tuple[str, str]] = set()

        # Pair neighbours (0,1), (2,3), ... as "risky".
        for i in range(0, len(upper) - 1, 2):
            a, b = upper[i], upper[i + 1]
            if a == b:
                continue
            pair = tuple(sorted((a, b)))
            pairs.add(pair)

        return cls(risky_pairs=pairs)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def _normalise_pair(self, drug1: str, drug2: str) -> Tuple[str, str]:
        a, b = drug1.upper(), drug2.upper()
        return tuple(sorted((a, b)))

    def has_ddi(self, drug1: str, drug2: str) -> bool:
        return self._normalise_pair(drug1, drug2) in self.risky_pairs

    def check_combination(self, drugs: Iterable[str]) -> List[str]:
        """Return human-readable warnings about risky pairs in the combination."""
        warnings: List[str] = []
        drug_list = [d for d in (drugs or []) if d]
        for a, b in combinations(drug_list, 2):
            if self.has_ddi(a, b):
                warnings.append(
                    f"Potential DDI detected between {a} and {b} (toy demo graph, not clinically validated)."
                )
        return warnings
