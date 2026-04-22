import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from enum import Enum
from sys import argv

from mewtwo.parsers.tabular import Tabular

class Attribute(Enum):
    SPECIES = 1
    IS_SYNTHETIC = 2

    def get_legend_name(self):
        name_map = {self.SPECIES: "Species", self.IS_SYNTHETIC: "Origin"}
        return name_map[self]

NAME_MAP = {"Escherichia coli (a)": "E. coli",
            "Bacillus subtilis (d)": "B. subtilis"}

STYLE_MAP = {"E. coli": {"marker": "o", "color": "tab:blue"},
             "B. subtilis": {"marker": "^", "color": "tab:orange"},
             "natural": {"marker": "o", "color": "tab:pink"},
             "synthetic": {"marker": "^", "color": "tab:red"}}


def file_to_plot_input(actual_predicted_file: str,
                       attribute: Attribute) -> list[list[float], list[float], list[str]]:
    plot_data = Tabular(actual_predicted_file, [0])
    x = []
    y = []
    attribute_labels = []
    for data_id in plot_data.data:
        attribute_value = plot_data.get_value(data_id, attribute.name.lower())

        if attribute == Attribute.IS_SYNTHETIC:
            if attribute_value == "True":
                attribute_labels.append("synthetic")
            else:
                attribute_labels.append("natural")
        else:
            attribute_labels.append(NAME_MAP[attribute_value])

        x_value = float(plot_data.get_value(data_id, "actual"))
        y_value = float(plot_data.get_value(data_id, "predicted"))

        x.append(x_value)
        y.append(y_value)


    return x, y, attribute_labels

def scatterplot(x, y, attr1, attr1_name: Attribute, out_file: str) -> None:
    x = np.array(x)
    y = np.array(y)
    attr1 = np.array(attr1)

    # --- Style mapping (single attribute: shape + color) ---
    unique_attr1 = np.unique(attr1)
    style_map = STYLE_MAP

    fig, ax = plt.subplots(figsize=(6, 5))

    # --- Scatter plot ---
    for val in unique_attr1:
        mask = attr1 == val
        if np.any(mask):
            ax.scatter(
                x[mask],
                y[mask],
                marker=style_map[val]["marker"],
                color=style_map[val]["color"],
                edgecolor="black",
                linewidth=0.5,
                s=50,
                label=str(val),
            )

    # --- Single combined legend ---
    ax.legend(
        title=f"{attr1_name.get_legend_name()}",
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0)
    )

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_file)


def plot_actual_vs_predicted(actual_predicted_file: str, attribute: Attribute, out_file: str) -> None:
    x, y, attribute_labels = file_to_plot_input(actual_predicted_file, attribute)
    scatterplot(x, y, attribute_labels, attribute, out_file)

if __name__ == "__main__":
    input_dir = argv[1]
    plot_file = os.path.join(input_dir, "actual_vs_predicted.svg")

    actual_predicted_file = os.path.join(argv[1], "actual_vs_predicted.txt")
    x, y, attribute_labels= file_to_plot_input(actual_predicted_file, Attribute[argv[2].upper()])
    scatterplot(x, y, attribute_labels, Attribute[argv[2].upper()], plot_file)
