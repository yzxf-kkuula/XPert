"""
独立生成分子注意力三联图：
1. 左图：按平均注意力分数给分子原子着色；
2. 中图：不同样本的原子注意力热图；
3. 右图：五个随机种子的平均原子注意力散点图。

该脚本参考以下文件中的实现逻辑整理而成：
- reproducing/fig3/Extended fig2_hadci_attn_molecule.ipynb
- reproducing/fig3/fig3D_egfr_attn_molecule.ipynb
- reproducing/fig3/molecule_vis.py

可直接在 PyCharm 中运行。
如果你想替换分子或注意力数据，只需要修改“用户可直接编辑区域”中的内容。
"""

import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from molecule_vis import (  # noqa: E402
    colorMap,
    plot_attention_heatmap,
    plot_attention_scatter,
    visualize_molecule_attention,
)


# =============================================================================
# 用户可直接编辑区域
# =============================================================================
# 分子名称：只用于总标题显示。
MOLECULE_NAME = "Vorinostat"

# 分子 SMILES：如果你想换分子，只需要把这个字符串改成目标分子的 SMILES。
# 当前默认示例为 vorinostat（HDAC inhibitor）。
MOLECULE_SMILES = "O=C(CCCCCCC(=O)Nc1ccccc1)NO"

# 直接定义：该值必须等于 MOLECULE_SMILES 的重原子数（RDKit 解析）。
# 你换 SMILES 后，这个值会自动更新；可用它检查你提供的数据长度是否匹配。
MOLECULE_HEAVY_ATOM_COUNT = Chem.MolFromSmiles(MOLECULE_SMILES).GetNumAtoms()

# 选择是否绘制左/中/右图：
# - 想省略右图：把 DRAW_RIGHT_PANEL 改为 False
# - 同理可关闭左图/中图
# - 关闭某图后，可在 main() 的 build_figure(...) 里把对应数据参数设为 None
DRAW_LEFT_PANEL = True
DRAW_MIDDLE_PANEL = True
DRAW_RIGHT_PANEL = True

# 颜色与样式配置（可直接改）：
# 1) 左图原子注意力颜色映射：LEFT_ATTENTION_CMAP
# 2) 中图热图颜色映射：MIDDLE_HEATMAP_CMAP
# 3) 左图原子高亮样式：
#    - "gradient": 当前默认的渐变光晕
#    - "solid": 实心高亮（例如橙色实心）
LEFT_ATTENTION_CMAP = colorMap
MIDDLE_HEATMAP_CMAP = colorMap
RIGHT_SCATTER_CMAP = "Paired"
LEFT_HIGHLIGHT_STYLE = "gradient"  # 可改为 "solid"
# 实心模式下的颜色（RGB；可用 0~1 或 0~255），例如橙色实心：
LEFT_SOLID_HIGHLIGHT_COLOR = (1.0, 0.65, 0.0)
LEFT_SOLID_HIGHLIGHT_ALPHA = 0.90

# 左图数据：长度必须等于 MOLECULE_HEAVY_ATOM_COUNT。
# 你可以直接修改成自己的原子平均注意力权重。
LEFT_ATOM_ATTENTION = np.array(
    [
        0.0125,
        0.0093,
        0.0088,
        0.0081,
        0.0074,
        0.0070,
        0.0082,
        0.0089,
        0.0117,
        0.0156,
        0.0101,
        0.0082,
        0.0076,
        0.0074,
        0.0078,
        0.0084,
        0.0091,
        0.0138,
        0.0165,
    ],
    dtype=float,
)

# 中图数据：shape = [样本数, 原子数]。
# 每一行表示一个样本中，各原子的注意力分布。
# 你可以自由替换为自己的热图数据。
MIDDLE_HEATMAP_ATTENTION = np.array(
    [
        [0.0120, 0.0090, 0.0086, 0.0079, 0.0072, 0.0069, 0.0080, 0.0087, 0.0114, 0.0152, 0.0098, 0.0080, 0.0074, 0.0072, 0.0077, 0.0083, 0.0089, 0.0135, 0.0161],
        [0.0127, 0.0092, 0.0089, 0.0082, 0.0075, 0.0071, 0.0083, 0.0090, 0.0118, 0.0159, 0.0102, 0.0083, 0.0076, 0.0074, 0.0079, 0.0085, 0.0092, 0.0141, 0.0168],
        [0.0124, 0.0094, 0.0087, 0.0080, 0.0073, 0.0070, 0.0082, 0.0088, 0.0115, 0.0157, 0.0100, 0.0081, 0.0075, 0.0073, 0.0078, 0.0084, 0.0090, 0.0137, 0.0164],
        [0.0129, 0.0095, 0.0090, 0.0083, 0.0076, 0.0072, 0.0084, 0.0091, 0.0119, 0.0161, 0.0103, 0.0084, 0.0078, 0.0075, 0.0080, 0.0086, 0.0093, 0.0140, 0.0169],
        [0.0122, 0.0091, 0.0085, 0.0078, 0.0071, 0.0068, 0.0079, 0.0086, 0.0113, 0.0150, 0.0097, 0.0079, 0.0073, 0.0071, 0.0076, 0.0082, 0.0088, 0.0134, 0.0160],
        [0.0126, 0.0093, 0.0088, 0.0081, 0.0074, 0.0070, 0.0082, 0.0089, 0.0116, 0.0155, 0.0101, 0.0082, 0.0076, 0.0074, 0.0078, 0.0085, 0.0091, 0.0139, 0.0166],
    ],
    dtype=float,
)

# 右图数据：shape = [5, 原子数]。
# 每一行表示一个随机种子对应的“平均原子注意力”。
# 如果你想完全复现 notebook 的右图逻辑，可以把这里替换成 5 个 seed 的均值向量。
RIGHT_SCATTER_ATTENTION = np.array(
    [
        [0.0120, 0.0090, 0.0086, 0.0079, 0.0072, 0.0069, 0.0080, 0.0087, 0.0114, 0.0152, 0.0098, 0.0080, 0.0074, 0.0072, 0.0077, 0.0083, 0.0089, 0.0135, 0.0161],
        [0.0127, 0.0092, 0.0089, 0.0082, 0.0075, 0.0071, 0.0083, 0.0090, 0.0118, 0.0159, 0.0102, 0.0083, 0.0076, 0.0074, 0.0079, 0.0085, 0.0092, 0.0141, 0.0168],
        [0.0124, 0.0094, 0.0087, 0.0080, 0.0073, 0.0070, 0.0082, 0.0088, 0.0115, 0.0157, 0.0100, 0.0081, 0.0075, 0.0073, 0.0078, 0.0084, 0.0090, 0.0137, 0.0164],
        [0.0129, 0.0095, 0.0090, 0.0083, 0.0076, 0.0072, 0.0084, 0.0091, 0.0119, 0.0161, 0.0103, 0.0084, 0.0078, 0.0075, 0.0080, 0.0086, 0.0093, 0.0140, 0.0169],
        [0.0122, 0.0091, 0.0085, 0.0078, 0.0071, 0.0068, 0.0079, 0.0086, 0.0113, 0.0150, 0.0097, 0.0079, 0.0073, 0.0071, 0.0076, 0.0082, 0.0088, 0.0134, 0.0160],
    ],
    dtype=float,
)

# 中图热图的样本标签；数量应与 MIDDLE_HEATMAP_ATTENTION 的行数一致。
# 如果设为 None，会自动生成 Sample 1、Sample 2 ...；
# 如果你想自定义行标签，可改成一个字符串列表。
HEATMAP_SAMPLE_LABELS = None

# 输出路径：改成 None 表示只显示不保存。
OUTPUT_PATH = CURRENT_DIR / "molecule_attention_visualization.svg"

# 是否显示热图数值。
SHOW_HEATMAP_VALUES = False

# 左图分子光晕大小。
ATOM_HIGHLIGHT_RADIUS = 35
# 左图颜色偏移强度（越大整体越亮）。
LEFT_COLOR_FACTOR = 0.5
# =============================================================================


def get_atom_symbols(smiles: str) -> List[str]:
    """根据 SMILES 返回原子符号列表。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无法解析 SMILES: {smiles}")
    return [atom.GetSymbol() for atom in mol.GetAtoms()]


def validate_inputs(
    smiles: str,
    left_atom_attention: Optional[np.ndarray] = None,
    middle_heatmap_attention: Optional[np.ndarray] = None,
    right_scatter_attention: Optional[np.ndarray] = None,
    heatmap_sample_labels: Optional[List[str]] = None,
    draw_left_panel: bool = True,
    draw_middle_panel: bool = True,
    draw_right_panel: bool = True,
) -> List[str]:
    """检查输入数据维度是否与分子原子数一致。"""
    atom_symbols = get_atom_symbols(smiles)
    atom_count = len(atom_symbols)

    if atom_count != MOLECULE_HEAVY_ATOM_COUNT:
        raise ValueError(
            f"MOLECULE_HEAVY_ATOM_COUNT={MOLECULE_HEAVY_ATOM_COUNT}，"
            f"但由 SMILES 实际解析得到原子数为 {atom_count}。"
        )

    if not (draw_left_panel or draw_middle_panel or draw_right_panel):
        raise ValueError("DRAW_LEFT_PANEL / DRAW_MIDDLE_PANEL / DRAW_RIGHT_PANEL 不能同时为 False。")

    if draw_left_panel:
        if left_atom_attention is None:
            raise ValueError("DRAW_LEFT_PANEL=True 时，left_atom_attention 不能为 None。")
        left_atom_attention = np.asarray(left_atom_attention, dtype=float)
        if left_atom_attention.ndim != 1:
            raise ValueError("LEFT_ATOM_ATTENTION 必须是一维数组。")
        if left_atom_attention.shape[0] != atom_count:
            raise ValueError(
                f"LEFT_ATOM_ATTENTION 长度为 {left_atom_attention.shape[0]}，"
                f"但分子原子数为 {atom_count}。"
            )

    if draw_middle_panel:
        if middle_heatmap_attention is None:
            raise ValueError("DRAW_MIDDLE_PANEL=True 时，middle_heatmap_attention 不能为 None。")
        middle_heatmap_attention = np.asarray(middle_heatmap_attention, dtype=float)
        if middle_heatmap_attention.ndim != 2:
            raise ValueError("MIDDLE_HEATMAP_ATTENTION 必须是二维数组。")
        if middle_heatmap_attention.shape[1] != atom_count:
            raise ValueError(
                f"MIDDLE_HEATMAP_ATTENTION 列数为 {middle_heatmap_attention.shape[1]}，"
                f"但分子原子数为 {atom_count}。"
            )
        if heatmap_sample_labels is not None and len(heatmap_sample_labels) != middle_heatmap_attention.shape[0]:
            raise ValueError(
                "HEATMAP_SAMPLE_LABELS 的长度必须与 MIDDLE_HEATMAP_ATTENTION 的行数一致。"
            )

    if draw_right_panel:
        if right_scatter_attention is None:
            raise ValueError("DRAW_RIGHT_PANEL=True 时，right_scatter_attention 不能为 None。")
        right_scatter_attention = np.asarray(right_scatter_attention, dtype=float)
        if right_scatter_attention.ndim != 2:
            raise ValueError("RIGHT_SCATTER_ATTENTION 必须是二维数组。")
        if right_scatter_attention.shape[1] != atom_count:
            raise ValueError(
                f"RIGHT_SCATTER_ATTENTION 列数为 {right_scatter_attention.shape[1]}，"
                f"但分子原子数为 {atom_count}。"
            )
        if right_scatter_attention.shape[0] == 0:
            raise ValueError(
                "RIGHT_SCATTER_ATTENTION 至少应包含 1 行数据。"
            )

    return atom_symbols


def build_figure(
    molecule_name: str,
    smiles: str,
    left_atom_attention: Optional[np.ndarray],
    middle_heatmap_attention: Optional[np.ndarray],
    right_scatter_attention: Optional[np.ndarray],
    output_path: Optional[Path] = None,
    draw_left_panel: bool = True,
    draw_middle_panel: bool = True,
    draw_right_panel: bool = True,
):
    """
    生成与 notebook 中相同结构的三联图。

    left_atom_attention、middle_heatmap_attention、right_scatter_attention
    彼此独立，便于你分别修改左图、中图、右图的数据。
    """
    atom_symbols = validate_inputs(
        smiles=smiles,
        left_atom_attention=left_atom_attention,
        middle_heatmap_attention=middle_heatmap_attention,
        right_scatter_attention=right_scatter_attention,
        heatmap_sample_labels=HEATMAP_SAMPLE_LABELS,
        draw_left_panel=draw_left_panel,
        draw_middle_panel=draw_middle_panel,
        draw_right_panel=draw_right_panel,
    )
    print(f"Atom count: {len(atom_symbols)}")
    print("Atom order:", atom_symbols)

    enabled_panels = []
    if draw_left_panel:
        enabled_panels.append("left")
    if draw_middle_panel:
        enabled_panels.append("middle")
    if draw_right_panel:
        enabled_panels.append("right")

    fig, axes = plt.subplots(
        1,
        len(enabled_panels),
        figsize=(7.3 * len(enabled_panels), 6),
        gridspec_kw={"width_ratios": [1] * len(enabled_panels)},
    )
    if len(enabled_panels) == 1:
        axes = [axes]

    axis_idx = 0

    if draw_left_panel:
        # 左图：分子结构 + 原子平均注意力着色
        visualize_molecule_attention(
            smiles=smiles,
            attn_weights=np.asarray(left_atom_attention, dtype=float),
            norm=True,
            cmap=LEFT_ATTENTION_CMAP,
            radius=ATOM_HIGHLIGHT_RADIUS,
            color_factor=LEFT_COLOR_FACTOR,
            highlight_style=LEFT_HIGHLIGHT_STYLE,
            solid_highlight_color=LEFT_SOLID_HIGHLIGHT_COLOR,
            solid_highlight_alpha=LEFT_SOLID_HIGHLIGHT_ALPHA,
            ax=axes[axis_idx],
        )
        axes[axis_idx].set_title("Chemical structure", fontsize=18, pad=12)
        axis_idx += 1

    if draw_middle_panel:
        # 中图：样本 × 原子的注意力热图
        heatmap_sample_labels = (
            [f"Sample {i + 1}" for i in range(np.asarray(middle_heatmap_attention).shape[0])]
            if HEATMAP_SAMPLE_LABELS is None
            else HEATMAP_SAMPLE_LABELS
        )
        plot_attention_heatmap(
            attention_matrix=np.asarray(middle_heatmap_attention, dtype=float),
            xticklabels=atom_symbols,
            yticklabels=heatmap_sample_labels,
            title="Attention heatmap across samples",
            cmap=MIDDLE_HEATMAP_CMAP,
            annot=SHOW_HEATMAP_VALUES,
            fmt=".4f",
            ax=axes[axis_idx],
        )
        axis_idx += 1

    if draw_right_panel:
        # 右图：五个随机种子的平均原子注意力散点图
        scatter_labels = [f"Seed {i + 1}" for i in range(np.asarray(right_scatter_attention).shape[0])]
        plot_attention_scatter(
            attention_vectors=np.asarray(right_scatter_attention, dtype=float),
            labels=scatter_labels,
            title="Mean attention across random seeds",
            cmap=RIGHT_SCATTER_CMAP,
            ax=axes[axis_idx],
        )

    fig.suptitle(
        f"Attention Visualization for {molecule_name}",
        fontsize=24,
        y=1.06,
    )
    plt.subplots_adjust(wspace=0.28 if len(enabled_panels) > 1 else 0.12)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix:
            save_path = output_path
            save_format = output_path.suffix.lstrip(".")
        else:
            save_path = output_path.with_suffix(".svg")
            save_format = "svg"
        fig.savefig(save_path, format=save_format, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


def main():
    """脚本入口。"""
    build_figure(
        molecule_name=MOLECULE_NAME,
        smiles=MOLECULE_SMILES,
        left_atom_attention=LEFT_ATOM_ATTENTION,
        middle_heatmap_attention=MIDDLE_HEATMAP_ATTENTION,
        right_scatter_attention=RIGHT_SCATTER_ATTENTION,
        output_path=OUTPUT_PATH,
        draw_left_panel=DRAW_LEFT_PANEL,
        draw_middle_panel=DRAW_MIDDLE_PANEL,
        draw_right_panel=DRAW_RIGHT_PANEL,
    )
    plt.show()


if __name__ == "__main__":
    main()
