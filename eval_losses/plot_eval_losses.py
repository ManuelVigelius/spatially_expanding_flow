import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

with open("eval_losses_results.json") as f:
    data = json.load(f)

models = list(data.keys())
grid_types = list(data[models[0]].keys())
metrics = ["vel_loss_lr", "vel_loss_fr", "img_mse_lr", "img_mse_fr"]
metric_labels = {
    "vel_loss_lr": "Velocity Loss (LR)",
    "vel_loss_fr": "Velocity Loss (FR)",
    "img_mse_lr": "Image MSE (LR)",
    "img_mse_fr": "Image MSE (FR)",
}

# Color palette per model
model_colors = {
    "baseline/ema": "#636EFA",
    "loss_a_8k/ema": "#EF553B",
    "loss_a_8k/normal": "#E89788",
    "loss_c_6k/ema": "#00CC96",
    "loss_c_6k/normal": "#69BFA8",
    "baseline_virtual_resize/ema": "#AA00CC",
}

# Dash style per metric type (lr vs fr)
dash_styles = {"lr": "solid", "fr": "dash"}

grid_sizes = [int(g.split("_")[1].split("x")[0]) for g in grid_types]

# ── Figure 1: Loss vs Timestep (one subplot per metric, one trace per model×grid) ──
# Too many combos — use dropdown to select grid, then show all models

fig1 = make_subplots(
    rows=2, cols=2,
    subplot_titles=[metric_labels[m] for m in metrics],
    shared_xaxes=True,
    vertical_spacing=0.12,
    horizontal_spacing=0.08,
)

positions = {metrics[i]: (i // 2 + 1, i % 2 + 1) for i in range(4)}

# Build traces for each (grid, model, metric) — show/hide via buttons
all_traces = []
for grid in grid_types:
    for model in models:
        timesteps = sorted(data[model][grid].keys(), key=float)
        t_vals = [float(t) for t in timesteps]
        for metric in metrics:
            y_vals = [data[model][grid][t].get(metric, None) for t in timesteps]
            row, col = positions[metric]
            trace = go.Scatter(
                x=t_vals,
                y=y_vals,
                mode="lines+markers",
                name=f"{model} | {grid}",
                legendgroup=f"{model}|{grid}",
                showlegend=(metric == metrics[0]),
                line=dict(
                    color=model_colors[model],
                    dash="solid" if grid == grid_types[0] else "dot",
                    width=2,
                ),
                marker=dict(size=5),
                visible=(grid == grid_types[0]),
                hovertemplate=f"<b>{model}</b><br>{grid}<br>t=%{{x:.4f}}<br>{metric_labels[metric]}=%{{y:.4f}}<extra></extra>",
            )
            fig1.add_trace(trace, row=row, col=col)
            all_traces.append((grid, model, metric))

# Build dropdown buttons for grid selection
buttons = []
for g in grid_types:
    visibility = [tr[0] == g for tr in all_traces]
    buttons.append(dict(
        label=g,
        method="update",
        args=[{"visible": visibility}, {"title": f"Loss vs Timestep — {g}"}],
    ))

fig1.update_layout(
    title=f"Loss vs Timestep — {grid_types[0]}",
    updatemenus=[dict(
        buttons=buttons,
        direction="down",
        x=0.0,
        xanchor="left",
        y=1.12,
        yanchor="top",
        showactive=True,
    )],
    legend=dict(x=1.02, y=1, bgcolor="rgba(0,0,0,0)"),
    height=700,
    template="plotly_dark",
    font=dict(size=12),
)
fig1.update_xaxes(title_text="Timestep (t)")
fig1.update_yaxes(title_text="Loss")

# ── Figure 2: Loss vs Grid Size (fixed timestep, one subplot per metric) ──
# Use slider for timestep selection
timesteps_all = sorted(data[models[0]][grid_types[0]].keys(), key=float)

fig2 = make_subplots(
    rows=2, cols=2,
    subplot_titles=[metric_labels[m] for m in metrics],
    shared_xaxes=True,
    vertical_spacing=0.12,
    horizontal_spacing=0.08,
)

traces2 = []
for t in timesteps_all:
    for model in models:
        for metric in metrics:
            y_vals = [data[model][grid][t].get(metric, None) for grid in grid_types]
            row, col = positions[metric]
            trace = go.Scatter(
                x=grid_sizes,
                y=y_vals,
                mode="lines+markers",
                name=model,
                legendgroup=model,
                showlegend=(metric == metrics[0]),
                line=dict(color=model_colors[model], width=2),
                marker=dict(size=7),
                visible=(t == timesteps_all[0]),
                hovertemplate=f"<b>{model}</b><br>t={t}<br>Grid=%{{x}}x%{{x}}<br>{metric_labels[metric]}=%{{y:.4f}}<extra></extra>",
            )
            fig2.add_trace(trace, row=row, col=col)
            traces2.append(t)

steps = []
for t in timesteps_all:
    visibility = [tr == t for tr in traces2]
    steps.append(dict(
        method="update",
        args=[{"visible": visibility}, {"title": f"Loss vs Grid Size — t={float(t):.4f}"}],
        label=f"{float(t):.2f}",
    ))

fig2.update_layout(
    title=f"Loss vs Grid Size — t={float(timesteps_all[0]):.4f}",
    sliders=[dict(
        active=0,
        steps=steps,
        x=0.05,
        y=-0.05,
        len=0.9,
        currentvalue=dict(prefix="Timestep: ", font=dict(size=14)),
    )],
    legend=dict(x=1.02, y=1, bgcolor="rgba(0,0,0,0)"),
    height=700,
    template="plotly_dark",
    font=dict(size=12),
)
fig2.update_xaxes(title_text="Grid Size (NxN)")
fig2.update_yaxes(title_text="Loss")

# ── Figure 3: Heatmap of each metric per model — grid size × timestep ──
fig3_figs = []
for metric in metrics:
    subfig = make_subplots(
        rows=1, cols=len(models),
        subplot_titles=models,
        shared_yaxes=True,
    )
    for col_idx, model in enumerate(models):
        z = []
        for t in timesteps_all:
            row_vals = [data[model][grid][t].get(metric, None) for grid in grid_types]
            z.append(row_vals)
        subfig.add_trace(
            go.Heatmap(
                z=z,
                x=[f"{s}x{s}" for s in grid_sizes],
                y=[f"{float(t):.2f}" for t in timesteps_all],
                colorscale="Viridis",
                showscale=(col_idx == len(models) - 1),
                hovertemplate="Grid=%{x}<br>t=%{y}<br>Value=%{z:.4f}<extra></extra>",
            ),
            row=1, col=col_idx + 1,
        )
    subfig.update_layout(
        title=f"Heatmap: {metric_labels[metric]}",
        height=500,
        template="plotly_dark",
        font=dict(size=12),
    )
    subfig.update_xaxes(title_text="Grid Size")
    subfig.update_yaxes(title_text="Timestep", col=1)
    fig3_figs.append(subfig)

# ── Figure 4: Model comparison — LR vs FR scatter per grid+timestep ──
fig4 = make_subplots(rows=1, cols=2, subplot_titles=["Velocity Loss: LR vs FR", "Image MSE: LR vs FR"])
for model in models:
    vel_lr, vel_fr, img_lr, img_fr, hover = [], [], [], [], []
    for grid in grid_types:
        for t in timesteps_all:
            vel_lr.append(data[model][grid][t].get("vel_loss_lr", None))
            vel_fr.append(data[model][grid][t].get("vel_loss_fr", None))
            img_lr.append(data[model][grid][t].get("img_mse_lr", None))
            img_fr.append(data[model][grid][t].get("img_mse_fr", None))
            hover.append(f"{grid} | t={float(t):.2f}")
    fig4.add_trace(go.Scatter(
        x=vel_lr, y=vel_fr, mode="markers", name=model, legendgroup=model,
        marker=dict(color=model_colors[model], size=5, opacity=0.6),
        text=hover, hovertemplate="<b>%{text}</b><br>LR=%{x:.4f}<br>FR=%{y:.4f}<extra></extra>",
    ), row=1, col=1)
    fig4.add_trace(go.Scatter(
        x=img_lr, y=img_fr, mode="markers", name=model, legendgroup=model,
        showlegend=False,
        marker=dict(color=model_colors[model], size=5, opacity=0.6),
        text=hover, hovertemplate="<b>%{text}</b><br>LR=%{x:.4f}<br>FR=%{y:.4f}<extra></extra>",
    ), row=1, col=2)

# Diagonal reference lines
for col in [1, 2]:
    fig4.add_shape(type="line", x0=0, y0=0, x1=2, y1=2,
                   line=dict(dash="dash", color="gray", width=1), row=1, col=col)

fig4.update_layout(
    title="LR vs FR Loss Scatter (diagonal = equal performance)",
    height=500,
    template="plotly_dark",
    font=dict(size=12),
)
fig4.update_xaxes(title_text="Low-Res Loss")
fig4.update_yaxes(title_text="Full-Res Loss")

# ── Write all to a single HTML ──
from plotly.io import to_html

html_parts = ["<html><head><meta charset='utf-8'><title>Eval Losses</title></head><body>"]
html_parts.append("<h1 style='font-family:sans-serif;color:#eee;background:#111;padding:16px;margin:0'>Eval Losses Results</h1>")
html_parts.append("<div style='background:#111;padding:8px'>")

for i, fig in enumerate([fig1, fig2] + fig3_figs + [fig4]):
    html_parts.append(to_html(fig, full_html=False, include_plotlyjs=(i == 0)))
    html_parts.append("<hr style='border-color:#333'>")

html_parts.append("</div></body></html>")

out_path = "eval_losses_plots.html"
with open(out_path, "w") as f:
    f.write("\n".join(html_parts))

print(f"Saved to {out_path}")
