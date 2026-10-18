# Technigma AI launcher branding

`technigma-ai.png` is a crop of the circuit-head logo supplied by Technigma AI
(`image_other_big_white.png`). The text assets are generated from those pixels,
not a manually approximated symbol.

The conversion was performed on macOS with jp2a 1.3.3:

```bash
for width in 34 44 52; do
  jp2a --width="$width" --invert --chars=' .:-=+*#@@' technigma-ai.png \
    | sed 's/[[:space:]]*$//' > "technigma-ai-${width}.txt"
done
```

Only the checked-in ASCII text is read by the launcher. The GB10 nodes need
neither jp2a nor an image-processing package. The menu chooses the largest logo
that fits beside its controls; narrow or short terminals retain the name and
tagline with a compact layout. Resize the terminal to show a larger rendition.
