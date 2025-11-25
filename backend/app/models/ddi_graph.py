from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, List, Set, Tuple


@dataclass
class DDIGraph:
    """Toy DDI graph using a set of (drug_a, drug_b) pairs.

    In your actual thesis implementation you would load a real DDI knowledge base
    (e.g., TWOSIDES) and possibly severity levels. Here we just use a tiny in-memory
    example so the demo is self-contained.
    """

    risky_pairs: Set[Tuple[str, str]]

    @classmethod
    def demo(cls) -> "DDIGraph":
        demo_pairs = {
            # These are **fake** codes for demo purposes only.
            ("DRUG_A", "DRUG_B"),
            ("DRUG_C", "DRUG_D"),
        }
        # Make them symmetric
        symmetric_pairs: Set[Tuple[str, str]] = set()
        for a, b in demo_pairs:
            symmetric_pairs.add((a, b))
            symmetric_pairs.add((b, a))
        return cls(risky_pairs=symmetric_pairs)

    def has_ddi(self, drug1: str, drug2: str) -> bool:
        key = (drug1.upper(), drug2.upper())
        return key in self.risky_pairs

    def check_combination(self, drugs: Iterable[str]) -> List[str]:
        """Return human-readable warnings about risky pairs in the given combination."""
        warnings: List[str] = []
        drug_list = [d.upper() for d in drugs]
        for a, b in combinations(drug_list, 2):
            if self.has_ddi(a, b):
                warnings.append(
                    f"Potential DDI detected between {a} and {b} (toy demo graph, not clinically validated)."
                )
        return warnings
