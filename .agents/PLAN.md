# Refactoring plan

## Current stage

Scaffold the repository for the ICLR 2027 blog post. Preserve all exploratory code,
results, and uncommitted changes under `experiments/_archive/` for reference.

## Next steps

1. Agree on the blog's main claim, scope, and evidence requirements.
2. Audit the archived journal and select the experiments that support that claim.
3. Rebuild the selected experiments as maintained `<NN>_<slug>` groups with shared
   harness code, tests, Makefiles, concise READMEs, and final figures.
4. Consolidate the existing draft and design notes into the post under `docs/_posts/`.
5. Check that every published number and figure comes from a maintained artifact.

## Source documents

- `README.md`: existing blog draft; not yet migrated or rewritten.
- `PLAN.md`: earlier design document; retain until the storyline review is complete.
- `RELATED_WORK.md`: existing literature notes.
- `experiments/_archive/README.md`: exploratory run log.

## Review bar

The blog should make a small number of clear claims. Each claim needs a maintained
experiment, an appropriate control, and a stated limitation. Do not carry forward a
conclusion merely because it appears in the archive.
