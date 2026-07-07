# tigris_tools

Standalone post-processing utilities for TIGRESS++ / Athena++ simulations.

Independent of the [tigris](https://github.com/PrincetonUniversity/tigris)
repository — these tools consume simulation outputs (restart files, HDF5
dumps, particle catalogs) but do not link against or require a source-tree
build of the simulation code.

## Tools

- **refine_restart** — read a TIGRESS++ restart checkpoint and produce a new
  checkpoint at a different mesh resolution or meshblock size, so that the
  simulation can restart from it without any code changes. See
  [`docs/refine_restart.md`](docs/refine_restart.md) for the full
  description.

## Layout

```
docs/
  refine_restart.md           # description of the refine_restart tool
  superpowers/specs/          # design specs produced during brainstorming
```

Implementation code lands here as each tool is designed.
