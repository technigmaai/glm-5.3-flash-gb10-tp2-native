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

The larger `Technigma AI` wordmark uses pre-generated FIGlet lettering. Its
three sizes were generated on macOS with FIGlet 2.2.5:

```bash
for font in mini small standard; do
  figlet -f "$font" -w 200 'Technigma AI' \
    | sed 's/[[:space:]]*$//' > "wordmark-${font}.txt"
done
```

Only the checked-in ASCII text is read by the launcher. The GB10 nodes need
neither jp2a, FIGlet nor an image-processing package. The menu chooses a logo and
wordmark that fit beside its controls, keeping all nine main actions visible in
a standard 80-column, 24-row terminal. Narrow or short terminals retain the name
and tagline with a compact layout. Resize the terminal to show larger renditions.
