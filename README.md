# quantum-circuit-de-solver

**Quantum-Circuit-Based Solution of Ordinary and Partial Differential Equations**
Mushrafi Munim Sushmit — course project, Physics 610: Quantum Optics and Quantum
Information Science, University of Tennessee, Knoxville, Fall 2026.

Code and saved results (`qoqpinn/`, `scripts/`, `results/`); the draft's LaTeX
source, figures, tables and diagrams are in `draft_paper/` .

```bash
pip install -e .
python scripts/optics_odes.py plot    # Figs. 3-5, Table 3 (drop "plot" to retrain)
python scripts/literature.py plot     # Figs. 6-8
python tests/test_core.py
cd draft_paper && latexmk -pdf draft.tex   # diagrams: cd draft_paper/diagrams && pdflatex <name>.tex
```
