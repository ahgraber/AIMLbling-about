#!/usr/bin/env -S uv run --script
"""WCAG contrast ratio calculator for the site's theme palette.

Evaluates candidate foreground/background color pairings against WCAG 2.1
contrast thresholds (AA >= 4.5:1, AAA >= 7:1 for normal text; AA >= 3:1,
AAA >= 4.5:1 for large text).  Edit the color lists below to test new
candidates.
"""


def srgb_to_linear(c):
    """Convert an 8-bit sRGB channel value to linear-light.

    Args:
        c: Integer channel value in 0–255.

    Returns:
        Linear-light value in 0.0–1.0.
    """
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(h):
    """Calculate relative luminance of a hex color per WCAG 2.1.

    Args:
        h: Hex color string (e.g. ``"#f4b300"`` or ``"f4b300"``).

    Returns:
        Relative luminance in 0.0–1.0.
    """
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return 0.2126 * srgb_to_linear(r) + 0.7152 * srgb_to_linear(g) + 0.0722 * srgb_to_linear(b)


def cr(c1, c2):
    """Compute the WCAG 2.1 contrast ratio between two hex colors.

    Args:
        c1: First hex color string.
        c2: Second hex color string.

    Returns:
        Contrast ratio >= 1.0 (lighter color is always in the numerator).
    """
    l1, l2 = luminance(c1), luminance(c2)
    if l1 < l2:
        l1, l2 = l2, l1
    return (l1 + 0.05) / (l2 + 0.05)


bgs = [("#f2f2f2", "current"), ("#faf7f2", "warm-ivory"), ("#f7f3eb", "warm-cream"), ("#f8f4ed", "warm-light")]
golds = [
    ("#f4b300", "xanthous"),
    ("#bf8c00", "darker"),
    ("#997000", "darkest"),
    ("#8a6500", "#8a6500"),
    ("#7a5a00", "#7a5a00"),
    ("#705200", "#705200"),
]
blues = [
    ("#4d78ae", "chefchaouen"),
    ("#3d6090", "#3d6090"),
    ("#3a6294", "#3a6294"),
    ("#345a85", "#345a85"),
    ("#2e5278", "#2e5278"),
]
texts = [("#1f1f1f", "eerie-black"), ("#2a2420", "warm-black"), ("#333333", "#333")]

print("=== LIGHT: Body text on backgrounds ===")
for bg, bn in bgs:
    for t, tn in texts:
        r = cr(t, bg)
        lv = "AAA" if r >= 7 else "AA" if r >= 4.5 else "FAIL"
        print(f"  {tn} on {bn}: {r:.1f}:1 [{lv}]")

print()
print("=== LIGHT: Gold headings on backgrounds ===")
for bg, bn in bgs:
    print(f"  --- {bn} ---")
    for g, gn in golds:
        r = cr(g, bg)
        lt = "AAA-lg" if r >= 4.5 else "AA-lg" if r >= 3 else "FAIL"
        nt = "AAA" if r >= 7 else "AA" if r >= 4.5 else "FAIL"
        print(f"    {gn}: {r:.1f}:1 [{lt}] [{nt}]")

print()
print("=== LIGHT: Blue links on backgrounds ===")
for bg, bn in bgs:
    print(f"  --- {bn} ---")
    for b, bln in blues:
        r = cr(b, bg)
        lv = "AAA" if r >= 7 else "AA" if r >= 4.5 else "FAIL"
        print(f"    {bln}: {r:.1f}:1 [{lv}]")

print()
print("=== LIGHT: Text on warm surfaces ===")
surfs = [("#f0ebe3", "warm-surf"), ("#ede7dd", "#ede7dd"), ("#e8e2d8", "#e8e2d8")]
for s, sn in surfs:
    for t, tn in texts:
        r = cr(t, s)
        lv = "AAA" if r >= 7 else "AA" if r >= 4.5 else "FAIL"
        print(f"  {tn} on {sn}: {r:.1f}:1 [{lv}]")

print()
print("=== DARK: Gold on dark backgrounds ===")
dbgs = [("#1f1f1f", "eerie-black"), ("#141414", "footer"), ("#282828", "code-bg")]
for db, dn in dbgs:
    print(f"  --- {dn} ---")
    for g, gn in golds:
        r = cr(g, db)
        lv = "AAA" if r >= 7 else "AA" if r >= 4.5 else "FAIL"
        print(f"    {gn}: {r:.1f}:1 [{lv}]")
    r2 = cr("#f2f2f2", db)
    print(f"    white-smoke: {r2:.1f}:1")

print()
print("=== DARK: Blue on dark backgrounds ===")
for db, dn in dbgs:
    for b, bln in blues:
        r = cr(b, db)
        lv = "AAA" if r >= 7 else "AA" if r >= 4.5 else "FAIL"
        print(f"  {bln} on {dn}: {r:.1f}:1 [{lv}]")
