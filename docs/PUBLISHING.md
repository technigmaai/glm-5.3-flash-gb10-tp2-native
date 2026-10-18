# Maintain releases and public documentation

## Public layout

Keep one operator entry point: `launchers/launch.sh`. Put installation steps in
`docs/DEPLOYMENT.md`, human-readable release contents in `VERSIONS.md`, and the
image/recipe/digest mapping in `profiles/images.json`. Keep the root README brief
and route readers to those pages. Preserve dated benchmarks as reference evidence,
not as claims about every image version.

`docs/DOCKERHUB.md` is the source for Docker Hub's full description. Its links
must be absolute GitHub URLs because Docker Hub does not resolve repository-relative
paths. Publish that file's contents as the overview after the corresponding GitHub
revision is available, then read the public overview back to verify the update.

## Image release policy

Use consecutive fixed version tags (`v0.1.0` through `v0.6.0` for this inventory).
The numbers identify releases, not patch counts. Publish a new version for changed
contents; do not silently retarget a documented release. Existing long-form tags
remain compatibility references. Do not introduce a floating image selector in
the public instructions.

For each new release:

1. Record the parent release, concrete changes, upstream commit pins and validation
   limits in `VERSIONS.md` and `profiles/provenance/`.
2. Add the matching repository-owned overlays and Compose recipe. Preserve source
   attribution and license notices. Keep private settings, weights, caches and logs out of source.
3. Publish the image and inspect its OCI index, Linux ARM64 manifest and configuration
   digests. Record all three in `profiles/images.json`; a local Docker image ID alone
   is not a portable registry pin.
4. Confirm both nodes can resolve the tag and all recipe mounts. Run tests and a
   read-only two-node switch plan before describing a release as usable by the launcher.
5. Update the version tables in the root README and Docker Hub overview. Use clear
   labels such as baseline, experiment and observed deployment; describe validation
   separately from performance expectations.

After intentional source edits, stage the intended files and run
`python3 scripts/update-manifest.py`, then stage the updated manifest. It hashes
tracked source only. A fresh clone must contain every source file referenced by
a recipe; nested source directories named `models` must not be omitted by ignore rules.

Repository checks are `python3 -m unittest discover -s tests` (Jinja2 and the Docker
Compose CLI are needed for the existing tests). Real model starts and inference
checks require an appropriate maintenance window and are distinct from CPU tests,
Compose rendering and a dry run. Never reuse historical results as qualification
of a newly changed runtime.
