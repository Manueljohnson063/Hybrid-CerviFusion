"""Save the simplified CerviFusion architecture diagram."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.transforms import build_transforms
from src.models.cervifusion import create_cervifusion

IMAGE = ROOT / "archive/im_Dyskeratotic/im_Dyskeratotic/CROPPED/007_01.bmp"
OUT_DIR = ROOT / "results" / "architecture"


def fmt(shape) -> str:
    return str(shape).replace(" ", "")


def draw_box(ax, x, y, title, shape, color, width=3.8, height=0.85):
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="#333333",
            linewidth=1.2,
        )
    )
    ax.text(x, y + 0.14, title, ha="center", va="center", fontsize=10, weight="bold")
    ax.text(x, y - 0.18, fmt(shape), ha="center", va="center", fontsize=9, family="monospace")


def draw_arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", color="#333333", linewidth=1.3),
    )


def save_simple_diagram(shapes: dict, output_path: Path) -> None:
    """Stage-level flow: two branches merged by attention fusion."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 11))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.8, 10.2)
    ax.axis("off")
    ax.set_title("CerviFusion (simplified)", fontsize=15, weight="bold")

    en_x, vit_x, mid_x = 3.0, 9.0, 6.0
    en_color, vit_color, head_color = "#cfe2f3", "#d9ead3", "#fce5cd"
    box_half = 0.45

    draw_box(ax, mid_x, 9.4, "Input image", shapes["input"], "#eeeeee")

    en_stages = [
        ("Conv stem", shapes["conv_stem"]),
        ("MBConv blocks ×16", shapes["en_blocks"]),
        ("conv_head (Grad-CAM)", shapes["conv_head"]),
        ("Global avg pool", shapes["en_pool"]),
        ("EN projection", shapes["en_projection"]),
    ]
    vit_stages = [
        ("Patch embed 16×16", shapes["patch_embed"]),
        ("Transformer blocks ×12", shapes["vit_blocks"]),
        ("CLS token", shapes["vit_out"]),
        ("ViT projection", shapes["vit_projection"]),
    ]
    en_rows = [8.0, 6.8, 5.6, 4.4, 3.2]
    vit_rows = [8.0, 6.8, 5.6, 3.2]

    draw_arrow(ax, (mid_x - 0.6, 9.4 - box_half), (en_x, en_rows[0] + box_half))
    draw_arrow(ax, (mid_x + 0.6, 9.4 - box_half), (vit_x, vit_rows[0] + box_half))

    for column_x, stages, rows, color in [
        (en_x, en_stages, en_rows, en_color),
        (vit_x, vit_stages, vit_rows, vit_color),
    ]:
        for index, ((title, shape), y) in enumerate(zip(stages, rows)):
            box_color = "#f4d19b" if "Grad-CAM" in title else color
            draw_box(ax, column_x, y, title, shape, box_color)
            if index > 0:
                draw_arrow(ax, (column_x, rows[index - 1] - box_half), (column_x, y + box_half))

    draw_box(ax, mid_x, 1.9, "Attention fusion", shapes["fusion"], head_color)
    draw_arrow(ax, (en_x, en_rows[-1] - box_half), (mid_x - 0.8, 1.9 + box_half))
    draw_arrow(ax, (vit_x, vit_rows[-1] - box_half), (mid_x + 0.8, 1.9 + box_half))

    draw_box(ax, mid_x, 0.6, "Classifier 512→256→5", shapes["classifier"], head_color)
    draw_arrow(ax, (mid_x, 1.9 - box_half), (mid_x, 0.6 + box_half))

    ax.text(
        mid_x,
        -0.45,
        "5 classes: Dyskeratotic, Koilocytotic, Metaplastic, Parabasal, Superficial-Intermediate",
        ha="center",
        fontsize=9,
        style="italic",
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def collect_stage_shapes(model, tensor) -> dict:
    en = model.en_branch.backbone
    vit = model.vit_branch.backbone
    stages = {
        "conv_stem": en.conv_stem,
        "en_blocks": en.blocks,
        "conv_head": en.conv_head,
        "en_pool": en.global_pool,
        "en_projection": model.en_projection,
        "patch_embed": vit.patch_embed,
        "vit_blocks": vit.blocks,
        "vit_out": model.vit_branch,
        "vit_projection": model.vit_projection,
        "fusion": model.fusion,
        "classifier": model.classifier,
    }
    shapes = {"input": tuple(tensor.shape)}

    def watch(key):
        def hook(_module, _inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            shapes[key] = tuple(out.shape)

        return hook

    hooks = [module.register_forward_hook(watch(key)) for key, module in stages.items()]
    with torch.no_grad():
        model(tensor)
    for hook in hooks:
        hook.remove()
    return shapes


def main() -> None:
    model = create_cervifusion(pretrained=False)
    model.eval()

    image = Image.open(IMAGE).convert("RGB")
    tensor = build_transforms(224, train=False)(image).unsqueeze(0)

    simple_path = OUT_DIR / "cervifusion_simplified.png"
    save_simple_diagram(collect_stage_shapes(model, tensor), simple_path)
    print(f"saved simplified image: {simple_path}")


if __name__ == "__main__":
    main()