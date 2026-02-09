"""
Stakeholder Weight Profiles

Defines criterion weights for different stakeholder perspectives.
Weights are based on SCOR model criteria and represent typical priorities
for each organizational role.
"""

import numpy as np
from typing import Dict
from dataclasses import dataclass


@dataclass
class StakeholderProfile:
    """Weight profile for a stakeholder."""
    name: str
    weights: np.ndarray
    description: str
    
    def as_dict(self, criteria_names: list) -> Dict[str, float]:
        """Return weights as a dictionary."""
        return dict(zip(criteria_names, self.weights))


# Criterion order: [Cost, Reliability, Recovery, Financial Exposure, Asset Efficiency]
CRITERIA_NAMES = ["Cost", "Reliability", "Recovery", "FinExposure", "AssetEfficiency"]


# =============================================================================
# STAKEHOLDER PROFILES
# =============================================================================

STAKEHOLDERS = {
    "CFO": StakeholderProfile(
        name="CFO (Chief Financial Officer)",
        weights=np.array([0.40, 0.15, 0.10, 0.25, 0.10]),
        description="""
        Cost-focused perspective. The CFO prioritizes total cost to serve (40%)
        and financial exposure to disruption risk (25%). Reliability and recovery
        time receive lower weights because the CFO views these as operational
        concerns that should be optimized within a cost envelope. Asset efficiency
        matters for capital allocation but is secondary to direct costs.
        
        Typical question: "What's the most cost-effective configuration that keeps
        our risk exposure within acceptable bounds?"
        """
    ),
    
    "COO": StakeholderProfile(
        name="COO (Chief Operations Officer)",
        weights=np.array([0.15, 0.30, 0.25, 0.15, 0.15]),
        description="""
        Operations-focused perspective. The COO prioritizes service reliability (30%)
        and recovery speed (25%) because these directly affect day-to-day fulfillment
        and customer satisfaction. Cost matters (15%) but is seen as a constraint
        rather than the primary objective. The COO cares about maintaining operations
        through disruptions, not just minimizing their probability.
        
        Typical question: "Which configuration keeps our fulfillment running smoothly
        even when things go wrong?"
        """
    ),
    
    "CMO": StakeholderProfile(
        name="CMO (Chief Marketing Officer)",
        weights=np.array([0.10, 0.35, 0.15, 0.15, 0.25]),
        description="""
        Customer-focused perspective. The CMO weights reliability highest (35%)
        because stockouts and service failures directly damage brand reputation.
        Asset efficiency (25%) matters because efficient operations enable competitive
        pricing and market responsiveness. Recovery time (15%) is important for
        managing customer communications during disruptions. Cost is lowest priority
        because the CMO views it as a finance concern.
        
        Typical question: "Which configuration protects our customer experience and
        brand reputation?"
        """
    ),
    
    "CRO": StakeholderProfile(
        name="CRO (Chief Risk Officer)",
        weights=np.array([0.10, 0.15, 0.30, 0.35, 0.10]),
        description="""
        Risk-focused perspective. The CRO prioritizes financial exposure (35%) and
        recovery time (30%) because these represent tail risk that could threaten
        business continuity. Reliability (15%) is a means to reduce risk, not an
        end in itself. Cost (10%) is a secondary concern because the CRO's job is
        to ensure survival, not optimize profits.
        
        Typical question: "Which configuration minimizes our downside exposure in
        worst-case scenarios?"
        """
    )
}


def get_stakeholder(name: str) -> StakeholderProfile:
    """Get stakeholder profile by name."""
    if name not in STAKEHOLDERS:
        raise ValueError(f"Unknown stakeholder: {name}. Available: {list(STAKEHOLDERS.keys())}")
    return STAKEHOLDERS[name]


def get_all_stakeholders() -> Dict[str, StakeholderProfile]:
    """Get all stakeholder profiles."""
    return STAKEHOLDERS


def print_stakeholder_summary():
    """Print summary of all stakeholder profiles."""
    print("\nSTAKEHOLDER WEIGHT PROFILES")
    print("=" * 70)
    print(f"{'Stakeholder':<12}" + "".join(f"{c:>14}" for c in CRITERIA_NAMES))
    print("-" * 70)
    
    for name, profile in STAKEHOLDERS.items():
        row = f"{name:<12}" + "".join(f"{w:>14.0%}" for w in profile.weights)
        print(row)
    
    print("\n" + "=" * 70)
    print("JUSTIFICATIONS")
    print("=" * 70)
    for name, profile in STAKEHOLDERS.items():
        print(f"\n{profile.name}")
        print("-" * 40)
        print(profile.description.strip())


if __name__ == "__main__":
    print_stakeholder_summary()
