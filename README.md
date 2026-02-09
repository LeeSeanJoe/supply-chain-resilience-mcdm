# Multi-Criteria Supply Chain Resilience Under Cyber Disruption

MIT Operations Strategy PBL Project

## Research Question

How do multi-criteria rankings of supply chain configurations shift when cyber disruption risk is incorporated into the evaluation?

## Project Structure

```
supply_chain_mcdm/
├── models.py           # Network, Node, Edge classes
├── configurations.py   # C1-C8 supply chain configurations
├── flow_solver.py      # Min-cost flow optimization
├── disruptions.py      # Disruption scenario functions
├── evaluate.py         # Configuration evaluation pipeline
├── criteria.py         # SCOR-based criteria functions
├── mcdm.py             # WSM, WPM, TOPSIS, PROMETHEE II
├── stakeholders.py     # Stakeholder weight profiles
├── sensitivity.py      # Weight and probability sensitivity analysis
├── visualizations.py   # Charts and network diagrams
├── run_analysis.py     # Full analysis pipeline
├── main.py             # Entry point
└── README.md
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Team

- Alice (Director)
- Sean (Research Lead)
- Meiqi (Chief of Communications)
- Yuantong & Sekito (Research and Programming)

## Methods

### MCDM Methods Implemented
- **WSM** (Weighted Sum Model): Additive aggregation of normalized scores
- **WPM** (Weighted Product Model): Multiplicative aggregation using weighted powers
- **TOPSIS**: Distance to ideal and anti-ideal solutions
- **PROMETHEE II**: Pairwise outranking with preference functions

### Supply Chain Configurations (C1-C8)
| Config | Sourcing | Inventory | IT Redundancy | Strategy |
|--------|----------|-----------|---------------|----------|
| C1 | Single | Lean (JIT) | None | Cost-optimized |
| C2 | Single | Buffered | None | Inventory hedge |
| C3 | Dual | Lean | None | Supplier hedge |
| C4 | Dual | Buffered | None | Hedged, no cyber |
| C5 | Single | Lean | Full | Cyber-resilient |
| C6 | Dual | Buffered | Full | Max resilience |
| C7 | Multi | Lean | Partial | Diversified |
| C8 | Multi | Buffered | Partial | Balanced |

### Disruption Scenarios
- **Cyberattack**: Warehouse visibility loss (Asahi-style)
- **Supplier Failure**: Primary supplier offline
- **Demand Surge**: Bullwhip effect (30-100% spike)
- **Compound**: Cyber + supplier failure

### Evaluation Criteria (SCOR-based)
1. Reliability (Perfect Order Fulfillment)
2. Responsiveness (Order Fulfillment Cycle Time)
3. Agility (Recovery Time)
4. Cost (Total Cost to Serve)
5. Asset Efficiency (Return on SC Fixed Assets)
