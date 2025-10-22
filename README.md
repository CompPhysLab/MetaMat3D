# MetaMat3D

Numerical playground for studying quasi-static electromagnetic response of 3D metamaterials built from split-ring resonators (SRRs). The codebase focuses on explicit ring-level models, finite metastructure assembly, and direct solvers for induced currents and effective polarizabilities.

## Repository Layout

- `Quasistatic/Metamaterial.py` — ring primitives, SRR effective-medium model, Plotly helpers.
- `Quasistatic/Metastructure.py` — finite assemblies with configurable shape, geometry, and orientations.
- `Quasistatic/Solvers.py` — direct current solver, polarizability post-processing, solution I/O.
- `Quasistatic/Calc.py` — minimal example that builds a cubic structure and runs a sweep.
- `Quasistatic/Parameters.py` — default SRR parameters and frequency grids.
- `Quasistatic/Data*/` — sample current/polarizability outputs.
- `requirements.txt` — Python dependencies (numpy, plotly, colormaps, scipy, tqdm, matplotlib, etc.).

## Core Objects

### `Ring`
- Immutable description of a single SRR: center coordinates, orientation (`pos` ∈ {`x`,`y`,`z`}), radius, strip width, and RLC parameters.
- `__repr__`/`__str__` expose concise diagnostics for logging.

### `visualize_rings`
- Plotly helper that renders a list of `Ring` instances with orientation-aware coloring.
- Highlights high-resistance rings (treated as air) using alpha blending.

### `SRR_metamaterial`
- Encapsulates effective-medium parameters for a periodic SRR lattice.
- Constructor requires `(R, L, C, radius, (dz, dy, dx))` and optionally strip width.
- Key methods:
  - `set_period(period)` — validate and update lattice spacing.
  - `get_params()` — serialized view used by `MetaSolver` outputs.
  - `effective_mu(omega)` — Lorentz-type response including lattice sum correction.
  - `spherical_polarizability`, `disk_polarizability`, `needle_polarizability` — helper formulas for standard shapes.

### `Metastructure`
- Builds finite collections of rings on top of an `SRR_metamaterial`.
- Constructor arguments: `shape` (nz, ny, nx), `geometry` (`rectangle`, `ellipse`, `cylinder-x/y/z`), `Type` (`smooth`, `ragged`, `centred`, `anizotropic`), `orientations`, and optional `full_grid` flag.
- Validates orientation list, grid size, and geometry choice at construction time.
- Key API:
  - `set_shape(shape)` / `set_type(type)` / `set_geometry(geometry)` update configuration with safety checks.
  - `fill_grid()` populates `self.rings` with `Ring` instances, handling offsets for different boundary types and optional full-grid masking.
  - `visualize(...)` delegates to `visualize_rings` to render the populated structure.
  - `name` property returns a canonical identifier (`{type}_{geometry}_{nz}x{ny}x{nx}`) used in saved datasets.

### `MetaSolver`
- Accepts either a pre-built `rings` list or a `Metastructure` instance plus a frequency sweep `Omega`.
- Builds impedance diagonal, mutual inductance matrix (via `RingsInductance.py`), and stores metadata for later export.
- Main operations:
  - `direct_solve(voltages)` — solves for complex currents across the sweep.
  - `direct_polarizability(axes, save_currents=False)` — computes averaged currents per orientation and rescales to magnetic moment per unit cell.

### `MetaSolution`
- Light-weight container for solver output (`currents`, `polarizability`, `Omega`, metadata).
- `save_sol(folder)` writes JSON parameters plus NPZ archives for currents/polarizability.
- `open_sol(folder, struct, load_currents=True)` restores solution files, including object arrays stored with `allow_pickle=True`.

## Quick Start

1. Create a virtual environment and install requirements:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run the demo script:
   ```bash
   python Quasistatic/Calc.py
   ```
   The script constructs a `Metastructure`, runs `MetaSolver.direct_solve`, and stores results under `Quasistatic/Data/`.
3. Open `Quasistatic/ResultComparing.ipynb` to explore polarizability plots or compare saved datasets.

## Typical Workflow for periodic metamaterial
Examples of some structures are in `Calc.py` and `Result.Comparing.ipynb`
- Configure SRR properties in `Quasistatic/Parameters.py` or instantiate `SRR_metamaterial` directly and create another parameters file.
- Build a `Metastructure`, adjust `shape`, `geometry`, `Type`, and `orientations`, then call `fill_grid()` .
- Visualize placement with `Metastructure.visualize()` (returns a Plotly figure object).
- Instantiate `MetaSolver` with the structure and frequency sweep; compute currents or polarizability.
- Persist the results via `MetaSolution.save_sol()` and reload them later with `open_sol()` for post-processing.

## Typical Workflow for custom rings structure 
- Create a rings array for any radiuses, width and positions withou regular grid 
- Visualize rings if it is necessary 
- Use only `Rings` argument for a `MetaSolver` to find currents on each given frequency, 
but you should to configure accurately where and how it should be saved.

## Data Conventions

- Solution folders contain `Params.json`, `currents.npz`, and `polarizability.npz`.
- `polarizability.npz` stores a Python dict (`{'x': array, 'y': array, 'z': array}`) serialized with `allow_pickle=True`; call `.item()` after loading if NumPy wraps it in an object array.
- Ring ordering inside the solver follows the orientation list supplied to `Metastructure`, so excitation vectors must respect the same stacking convention.

