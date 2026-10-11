"""Minimal dependency-free SVG line plots, 900x460."""
W, H = 900, 460
L, R, T, B = 70, 20, 40, 60
COL = ["#0f766e", "#1e3a8a", "#b45309", "#6b7280"]

def line_plot(path, title, xlabel, ylabel, series, points=None):
    """series: list of (label, xs, ys). points: optional list of (label, xs, ys) drawn as dots."""
    allx = [x for _, xs, _ in series for x in xs]; ally = [y for _, _, ys in series for y in ys]
    if points:
        allx += [x for _, xs, _ in points for x in xs]; ally += [y for _, _, ys in points for y in ys]
    x0, x1, y0, y1 = min(allx), max(allx), min(ally), max(ally)
    pad = 0.05 * (y1 - y0 or 1); y0 -= pad; y1 += pad
    fx = lambda x: L + (x - x0) / (x1 - x0 or 1) * (W - L - R)
    fy = lambda y: H - B - (y - y0) / (y1 - y0 or 1) * (H - T - B)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Arial, sans-serif" font-size="13">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{L}" y="24" font-size="15" font-weight="bold" fill="#111827">{title}</text>']
    for i in range(6):
        yv = y0 + (y1 - y0) * i / 5; yy = fy(yv)
        o.append(f'<line x1="{L}" y1="{yy:.1f}" x2="{W-R}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        o.append(f'<text x="{L-8}" y="{yy+4:.1f}" text-anchor="end" fill="#374151">{yv:.2f}</text>')
    for i in range(6):
        xv = x0 + (x1 - x0) * i / 5; xx = fx(xv)
        o.append(f'<text x="{xx:.1f}" y="{H-B+20}" text-anchor="middle" fill="#374151">{xv:.0f}</text>')
    o.append(f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="#374151"/><line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="#374151"/>')
    o.append(f'<text x="{(L+W-R)/2}" y="{H-24}" text-anchor="middle" fill="#111827">{xlabel}</text>')
    o.append(f'<text x="16" y="{(T+H-B)/2}" transform="rotate(-90 16 {(T+H-B)/2})" text-anchor="middle" fill="#111827">{ylabel}</text>')
    for k, (lab, xs, ys) in enumerate(series):
        pts = " ".join(f"{fx(x):.1f},{fy(y):.1f}" for x, y in zip(xs, ys))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{COL[k%4]}" stroke-width="2.2"/>')
        o.append(f'<line x1="{W-250}" y1="{T+6+k*18}" x2="{W-226}" y2="{T+6+k*18}" stroke="{COL[k%4]}" stroke-width="2.2"/><text x="{W-220}" y="{T+10+k*18}" fill="#111827">{lab}</text>')
    for k, (lab, xs, ys) in enumerate(points or []):
        kk = len(series) + k
        for x, y in zip(xs, ys):
            o.append(f'<circle cx="{fx(x):.1f}" cy="{fy(y):.1f}" r="3.5" fill="{COL[kk%4]}"/>')
        o.append(f'<circle cx="{W-238}" cy="{T+6+kk*18}" r="3.5" fill="{COL[kk%4]}"/><text x="{W-220}" y="{T+10+kk*18}" fill="#111827">{lab}</text>')
    o.append('</svg>')
    open(path, "w").write("\n".join(o))
