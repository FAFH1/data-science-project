"""Reproducible Iris exploratory analysis and nearest-centroid classification.

The project intentionally uses only pandas, NumPy, and Pillow so the analysis is
lightweight and can be run in a basic Python environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "iris.csv"
FIGURE_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "results"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
SPECIES = ["setosa", "versicolor", "virginica"]
COLORS = {
    "setosa": "#2563eb",
    "versicolor": "#ea580c",
    "virginica": "#16a34a",
}


def load_data() -> pd.DataFrame:
    """Load and validate the Iris dataset."""
    df = pd.read_csv(DATA_PATH)
    expected = FEATURES + ["species"]
    if list(df.columns) != expected:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    if len(df) != 150 or df["species"].nunique() != 3:
        raise ValueError("Expected 150 rows and 3 species.")
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values.")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics by species."""
    summary = (
        df.groupby("species", sort=False)[FEATURES]
        .agg(["mean", "std", "min", "max"])
        .round(3)
    )
    RESULT_DIR.mkdir(exist_ok=True)
    summary.to_csv(RESULT_DIR / "species_summary.csv")
    return summary


def stratified_split(
    df: pd.DataFrame, train_fraction: float = 0.8, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a deterministic, stratified train/test split."""
    rng = np.random.default_rng(seed)
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in df.groupby("species", sort=False):
        indices = rng.permutation(len(group))
        cutoff = int(len(group) * train_fraction)
        train_parts.append(group.iloc[indices[:cutoff]])
        test_parts.append(group.iloc[indices[cutoff:]])
    return (
        pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True),
        pd.concat(test_parts).sample(frac=1, random_state=seed).reset_index(drop=True),
    )


def nearest_centroid(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Classify observations using standardized nearest class centroids."""
    mean = train[FEATURES].mean()
    std = train[FEATURES].std(ddof=0).replace(0, 1)
    x_train = (train[FEATURES] - mean) / std
    x_test = (test[FEATURES] - mean) / std

    centroids = {
        species: x_train.loc[train["species"] == species].mean().to_numpy()
        for species in SPECIES
    }
    centroid_matrix = np.vstack([centroids[species] for species in SPECIES])

    distances = np.linalg.norm(
        x_test.to_numpy()[:, None, :] - centroid_matrix[None, :, :], axis=2
    )
    predictions = np.array(SPECIES)[distances.argmin(axis=1)]

    confusion = pd.crosstab(
        pd.Categorical(test["species"], categories=SPECIES),
        pd.Categorical(predictions, categories=SPECIES),
        rownames=["actual"],
        colnames=["predicted"],
        dropna=False,
    ).fillna(0).astype(int)

    accuracy = float((predictions == test["species"].to_numpy()).mean())
    metrics = {
        "model": "standardized nearest centroid",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "random_seed": 42,
        "accuracy": round(accuracy, 4),
    }
    confusion.to_csv(RESULT_DIR / "confusion_matrix.csv")
    (RESULT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return predictions, confusion.to_numpy(), metrics


def font(size: int) -> ImageFont.ImageFont:
    """Load a readable font with cross-platform fallbacks."""
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, value: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    """Return text dimensions across Pillow versions."""
    box = draw.textbbox((0, 0), value, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def save_scatter(df: pd.DataFrame) -> None:
    """Plot petal length against petal width by species."""
    width, height = 1400, 900
    margin = {"left": 150, "right": 60, "top": 120, "bottom": 120}
    image = Image.new("RGB", (width, height), "#fffdf8")
    draw = ImageDraw.Draw(image)
    title_font, label_font, tick_font = font(42), font(26), font(20)

    x_values = df["petal_length"]
    y_values = df["petal_width"]
    x_min, x_max = 0.5, 7.2
    y_min, y_max = -0.1, 2.7

    def px(value: float) -> float:
        return margin["left"] + (value - x_min) / (x_max - x_min) * (
            width - margin["left"] - margin["right"]
        )

    def py(value: float) -> float:
        return height - margin["bottom"] - (value - y_min) / (y_max - y_min) * (
            height - margin["top"] - margin["bottom"]
        )

    plot_left, plot_right = margin["left"], width - margin["right"]
    plot_top, plot_bottom = margin["top"], height - margin["bottom"]

    # Grid and axes.
    for tick in np.arange(1, 8):
        x = px(float(tick))
        draw.line((x, plot_top, x, plot_bottom), fill="#e7e5e4", width=2)
        label = str(tick)
        label_w, _ = text_size(draw, label, tick_font)
        draw.text((x - label_w / 2, plot_bottom + 18), label, font=tick_font, fill="#44403c")
    for tick in np.arange(0, 2.6, 0.5):
        y = py(float(tick))
        draw.line((plot_left, y, plot_right, y), fill="#e7e5e4", width=2)
        label = f"{tick:.1f}"
        label_w, _ = text_size(draw, label, tick_font)
        draw.text((plot_left - label_w - 18, y - 10), label, font=tick_font, fill="#44403c")
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#292524", width=3)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#292524", width=3)

    # Data points.
    for species in SPECIES:
        group = df[df["species"] == species]
        for row in group.itertuples():
            x, y = px(row.petal_length), py(row.petal_width)
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=COLORS[species], outline="#ffffff", width=2)

    draw.text((margin["left"], 36), "Petal length and width separate Iris species", font=title_font, fill="#1c1917")
    x_label = "Petal length (cm)"
    y_label = "Petal width (cm)"
    xw, _ = text_size(draw, x_label, label_font)
    draw.text(((plot_left + plot_right - xw) / 2, height - 62), x_label, font=label_font, fill="#292524")
    yw, _ = text_size(draw, y_label, label_font)
    temp = Image.new("RGBA", (int(yw) + 8, 34), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp)
    temp_draw.text((0, 0), y_label, font=label_font, fill="#292524")
    temp = temp.rotate(90, expand=True)
    image.paste(temp, (20, int((plot_top + plot_bottom - temp.height) / 2)), temp)

    # Legend.
    legend_x, legend_y = width - 310, 55
    for index, species in enumerate(SPECIES):
        y = legend_y + index * 38
        draw.ellipse((legend_x, y, legend_x + 18, y + 18), fill=COLORS[species])
        draw.text((legend_x + 30, y - 3), species, font=tick_font, fill="#292524")

    image.save(FIGURE_DIR / "petal_scatter.png", quality=95)


def save_species_means(df: pd.DataFrame) -> None:
    """Plot mean petal length with one standard-deviation error bars."""
    width, height = 1200, 800
    margin = {"left": 130, "right": 60, "top": 120, "bottom": 120}
    image = Image.new("RGB", (width, height), "#fffdf8")
    draw = ImageDraw.Draw(image)
    title_font, label_font, tick_font = font(40), font(25), font(20)

    grouped = df.groupby("species", sort=False)["petal_length"]
    means = grouped.mean().reindex(SPECIES)
    stds = grouped.std().reindex(SPECIES)
    max_y = 8.0

    def py(value: float) -> float:
        return height - margin["bottom"] - value / max_y * (height - margin["top"] - margin["bottom"])

    plot_left, plot_right = margin["left"], width - margin["right"]
    plot_top, plot_bottom = margin["top"], height - margin["bottom"]

    for tick in np.arange(0, 9, 1):
        y = py(float(tick))
        draw.line((plot_left, y, plot_right, y), fill="#e7e5e4", width=2)
        label = str(tick)
        label_w, _ = text_size(draw, label, tick_font)
        draw.text((plot_left - label_w - 16, y - 10), label, font=tick_font, fill="#44403c")
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#292524", width=3)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#292524", width=3)

    plot_width = plot_right - plot_left
    slot = plot_width / len(SPECIES)
    bar_width = slot * 0.48
    for index, species in enumerate(SPECIES):
        center = plot_left + slot * (index + 0.5)
        mean = float(means[species])
        std = float(stds[species])
        top = py(mean)
        draw.rounded_rectangle(
            (center - bar_width / 2, top, center + bar_width / 2, plot_bottom),
            radius=12,
            fill=COLORS[species],
        )
        error_top, error_bottom = py(mean + std), py(max(mean - std, 0))
        draw.line((center, error_top, center, error_bottom), fill="#1c1917", width=4)
        draw.line((center - 18, error_top, center + 18, error_top), fill="#1c1917", width=4)
        draw.line((center - 18, error_bottom, center + 18, error_bottom), fill="#1c1917", width=4)
        value = f"{mean:.2f} cm"
        value_w, value_h = text_size(draw, value, tick_font)
        value_y = min(top - 42, error_top - 52)
        draw.rectangle(
            (center - value_w / 2 - 7, value_y - 5, center + value_w / 2 + 7, value_y + value_h + 7),
            fill="#fffdf8",
        )
        draw.text((center - value_w / 2, value_y), value, font=tick_font, fill="#1c1917")
        species_w, _ = text_size(draw, species, label_font)
        draw.text((center - species_w / 2, plot_bottom + 24), species, font=label_font, fill="#292524")

    draw.text((margin["left"], 38), "Virginica has the longest petals on average", font=title_font, fill="#1c1917")
    y_label = "Petal length (cm)"
    yw, _ = text_size(draw, y_label, label_font)
    temp = Image.new("RGBA", (int(yw) + 8, 34), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp)
    temp_draw.text((0, 0), y_label, font=label_font, fill="#292524")
    temp = temp.rotate(90, expand=True)
    image.paste(temp, (20, int((plot_top + plot_bottom - temp.height) / 2)), temp)
    draw.text((margin["left"], height - 45), "Bars show mean; whiskers show one standard deviation.", font=tick_font, fill="#57534e")
    image.save(FIGURE_DIR / "species_means.png", quality=95)


def save_confusion_matrix(confusion: np.ndarray, accuracy: float) -> None:
    """Render the confusion matrix as a clean table."""
    width, height = 1100, 860
    image = Image.new("RGB", (width, height), "#fffdf8")
    draw = ImageDraw.Draw(image)
    title_font, label_font, cell_font = font(38), font(24), font(28)

    draw.text((70, 42), "Nearest-centroid confusion matrix", font=title_font, fill="#1c1917")
    draw.text((70, 98), f"Hold-out accuracy: {accuracy:.1%}", font=label_font, fill="#57534e")

    left, top, cell = 300, 210, 160
    for column, species in enumerate(SPECIES):
        value_w, _ = text_size(draw, species, label_font)
        draw.text((left + column * cell + (cell - value_w) / 2, top - 54), species, font=label_font, fill="#292524")
    for row, species in enumerate(SPECIES):
        draw.text((left - 28 - text_size(draw, species, label_font)[0], top + row * cell + 60), species, font=label_font, fill="#292524")
        for column in range(len(SPECIES)):
            x0, y0 = left + column * cell, top + row * cell
            value = int(confusion[row, column])
            fill = "#dcfce7" if row == column else ("#fee2e2" if value else "#f5f5f4")
            draw.rounded_rectangle((x0 + 4, y0 + 4, x0 + cell - 4, y0 + cell - 4), radius=12, fill=fill, outline="#d6d3d1", width=2)
            value_text = str(value)
            value_w, value_h = text_size(draw, value_text, cell_font)
            draw.text((x0 + (cell - value_w) / 2, y0 + (cell - value_h) / 2 - 2), value_text, font=cell_font, fill="#1c1917")

    x_label = "Predicted species"
    y_label = "Actual species"
    xw, _ = text_size(draw, x_label, label_font)
    draw.text((left + (cell * 3 - xw) / 2, top + cell * 3 + 28), x_label, font=label_font, fill="#292524")
    yw, _ = text_size(draw, y_label, label_font)
    temp = Image.new("RGBA", (int(yw) + 8, 34), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp)
    temp_draw.text((0, 0), y_label, font=label_font, fill="#292524")
    temp = temp.rotate(90, expand=True)
    image.paste(temp, (70, int(top + (cell * 3 - temp.height) / 2)), temp)
    draw.text((70, height - 44), "Deterministic 80/20 stratified split; random seed = 42.", font=font(19), fill="#78716c")
    image.save(FIGURE_DIR / "confusion_matrix.png", quality=95)


def main() -> None:
    FIGURE_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)
    df = load_data()
    summary = summarize(df)
    train, test = stratified_split(df)
    _, confusion, metrics = nearest_centroid(train, test)
    save_scatter(df)
    save_species_means(df)
    save_confusion_matrix(confusion, metrics["accuracy"])

    print(f"Rows: {len(df)}")
    print("Species means for petal length (cm):")
    print(summary[("petal_length", "mean")].to_string())
    print(f"Hold-out accuracy: {metrics['accuracy']:.1%}")
    print(f"Figures written to: {FIGURE_DIR}")
    print(f"Results written to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
