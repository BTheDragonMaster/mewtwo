from dataclasses import dataclass, field
import os

from math import pi, sin, cos

from mewtwo.embeddings.terminator.terminator import get_terminator_part_sizes, Terminator
from mewtwo.parsers.parse_feature_file import parse_feature_file
from mewtwo.machine_learning.feature_inference.infer_features_rf import sum_importance_per_base, \
    normalize_importances, SortedFeatures

from mewtwo.embeddings.feature_labels import FeatureLabel, FeatureCategory, FeatureType


@dataclass
class Point:
    x: float
    y: float

    def rotate_around_origin(self, angle):
        new_x = self.x * cos(angle) - self.y * sin(angle)
        new_y = self.x * sin(angle) + self.y * cos(angle)
        self.x = new_x
        self.y = new_y

    def move(self, x, y):
        self.x += x
        self.y += y


def get_extremes(points: list[Point]) -> tuple[float, float, float, float]:
    return min([point.x for point in points]), max([point.x for point in points]), \
           min([point.y for point in points]), max([point.y for point in points])


@dataclass
class TerminatorDrawingOptions:
    base_diameter: float = 20
    circle_diameter: float = 15
    color: tuple[int, int, int] = (255, 0, 0)

    def __hash__(self):
        return hash((self.base_diameter, self.circle_diameter, self.color))


DEFAULT_OPTIONS = TerminatorDrawingOptions()


@dataclass
class BaseDrawing:
    centre: Point
    radius: float
    color: tuple[int, int, int] = (255, 0, 0)
    transparency: float = 1.0

    def get_svg_circle(self):
        return f'<circle cx="{self.centre.x}" cy="{self.centre.y}" r="{self.radius}" fill="rgba({self.color[0]},{self.color[1]},{self.color[2]},{self.transparency:.2f})"/>\n'


@dataclass
class Line:
    point_1: Point
    point_2: Point

    def get_svg_line(self):
        return f'<line x1="{self.point_1.x}" y1="{self.point_1.y}" x2="{self.point_2.x}" y2="{self.point_2.y}"  />'


@dataclass
class TerminatorDrawing:
    loop_size: int
    stem_size: int
    a_tract_size: int
    u_tract_size: int
    options: TerminatorDrawingOptions = DEFAULT_OPTIONS
    feature_to_importance: dict[FeatureLabel, float] = field(default_factory=dict)

    @classmethod
    def from_feature_file(cls, feature_file: str, features_to_visualise: FeatureType,
                          options: TerminatorDrawingOptions = DEFAULT_OPTIONS) -> "TerminatorDrawing":
        feature_to_importance = parse_feature_file(feature_file)
        feature_category_to_size = {FeatureCategory.STEM: 0,
                                    FeatureCategory.LOOP: 0,
                                    FeatureCategory.U_TRACT: 0,
                                    FeatureCategory.A_TRACT: 0}

        for feature_label in feature_to_importance:
            if feature_category_to_size[feature_label.feature_category] < feature_label.base_index:
                feature_category_to_size[feature_label.feature_category] = feature_label.base_index

        selected_features_to_importance = {}

        for feature, importance in feature_to_importance.items():
            if feature.feature_type in features_to_visualise:
                selected_features_to_importance[feature] = importance

        if len(features_to_visualise) > 1:
            selected_features_to_importance = sum_importance_per_base(selected_features_to_importance)

        return cls(feature_category_to_size[FeatureCategory.LOOP],
                   feature_category_to_size[FeatureCategory.STEM],
                   feature_category_to_size[FeatureCategory.A_TRACT],
                   feature_category_to_size[FeatureCategory.U_TRACT], options=options,
                   feature_to_importance=selected_features_to_importance)

    @classmethod
    def from_terminators(cls, terminators: list[Terminator],
                         drawing_options: TerminatorDrawingOptions = DEFAULT_OPTIONS) -> "TerminatorDrawing":

        max_loop, max_stem, max_a, max_u = get_terminator_part_sizes(terminators)
        return cls(max_loop, max_stem, max_a, max_u, drawing_options)

    def get_svg(self):
        transparencies = []

        loop_coords = self.get_loop_coords()
        min_x, max_x, min_y, max_y = get_extremes(loop_coords)
        x_translation = -1 * min(0.0, min_x) + 10
        y_translation = -1 * min(0.0, min_y) + 10

        for loop_coord in loop_coords:
            loop_coord.move(x_translation, y_translation)

        circle_radius = self.options.circle_diameter / 2

        loop = []
        for loop_coord in loop_coords:
            loop.append(BaseDrawing(loop_coord, circle_radius))
            transparencies.append(0.0)

        stem_left_x = loop_coords[0].x
        stem_right_x = loop_coords[-1].x

        stem_y = loop_coords[0].y

        left_stem_coords = []
        right_stem_coords = []
        lines = [Line(Point(stem_left_x + circle_radius, stem_y), Point(stem_right_x - circle_radius, stem_y))]

        for i in range(self.stem_size - 1):
            stem_y += self.options.base_diameter

            left_stem_coords.append(Point(stem_left_x, stem_y))
            right_stem_coords.append(Point(stem_right_x, stem_y))
            lines.append(Line(Point(stem_left_x + circle_radius, stem_y), Point(stem_right_x - circle_radius, stem_y)))
            transparencies.append(0.0)
            transparencies.append(0.0)

        left_stem_coords.reverse()
        right_stem_coords.reverse()

        tract_y = left_stem_coords[0].y

        a_tract_x = left_stem_coords[0].x
        u_tract_x = right_stem_coords[0].x

        a_tract_coords = []
        u_tract_coords = []

        for i in range(self.a_tract_size):
            a_tract_x -= self.options.base_diameter
            a_tract_coords.append(Point(a_tract_x, tract_y))
            transparencies.append(0.0)

        a_tract_coords.reverse()

        for i in range(self.u_tract_size):
            u_tract_x += self.options.base_diameter
            u_tract_coords.append(Point(u_tract_x, tract_y))
            transparencies.append(0.0)

        base_coords = a_tract_coords + left_stem_coords + right_stem_coords + loop_coords + u_tract_coords

        if self.feature_to_importance is not None:
            normalize_importances(self.feature_to_importance)
            sorted_features = SortedFeatures(self.feature_to_importance)
            loop_transparencies = [sorted_features.left_stem_shoulder[-1][1]] + [x[1] for x in sorted_features.loop] + [sorted_features.right_stem_shoulder[-1][1]]
            left_shoulder_transparencies = [x[1] for x in sorted_features.left_stem_shoulder[:-1]]
            right_shoulder_transparencies = [x[1] for x in sorted_features.right_stem_shoulder[:-1]]
            a_tract_transparencies = [x[1] for x in sorted_features.a_tract]
            u_tract_transparencies = [x[1] for x in sorted_features.u_tract]
            transparencies = a_tract_transparencies + left_shoulder_transparencies + right_shoulder_transparencies + loop_transparencies + u_tract_transparencies

        min_x, max_x, min_y, max_y = get_extremes(base_coords)
        viewbox = f"0 0 {int(max_x) + 10} {int(max_y) + 10}"

        header = f'<svg viewBox="{viewbox}" xmlns="http://www.w3.org/2000/svg">\n'
        style = '<style> circle {stroke: black;stroke-width: 2px;} line {stroke: black;stroke-width: 2px;}</style>\n'
        body = ""

        for line in lines:
            body += line.get_svg_line()
        for i, base_coord in enumerate(base_coords):
            transparency = transparencies[i]
            base_drawing = BaseDrawing(base_coord, circle_radius, transparency=transparency)

            body += base_drawing.get_svg_circle()

        footer = '</svg>'

        return header + style + body + footer

    def write_svg(self, out_file):
        svg_string = self.get_svg()
        with open(out_file, 'w') as out:
            out.write(svg_string)

    def get_loop_coords(self):
        nr_polygon_sides = self.loop_size + 2
        theta = 2 * pi / nr_polygon_sides

        r = self.options.base_diameter / (2 * sin(pi / nr_polygon_sides))

        angle = pi / nr_polygon_sides
        loop_centres = []

        while angle < 2 * pi:
            coords = Point(cos(angle) * r, sin(angle) * r)
            loop_centres.append(coords)
            angle += theta

        for base_centre in loop_centres:
            base_centre.rotate_around_origin(pi / 2)

        return loop_centres


def visualise_feature_importances(feature_file, out_dir):

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    out_a = os.path.join(out_dir, "is_a.svg")
    out_c = os.path.join(out_dir, "is_c.svg")
    out_u = os.path.join(out_dir, "is_u.svg")
    out_g = os.path.join(out_dir, "is_g.svg")
    out_all = os.path.join(out_dir, "all_features.svg")

    terminator_drawing = TerminatorDrawing.from_feature_file(feature_file, FeatureType.ONE_HOT_TYPES)
    terminator_drawing.write_svg(out_all)

    terminator_drawing = TerminatorDrawing.from_feature_file(feature_file, FeatureType.IS_A)
    terminator_drawing.write_svg(out_a)

    terminator_drawing = TerminatorDrawing.from_feature_file(feature_file, FeatureType.IS_U)
    terminator_drawing.write_svg(out_u)

    terminator_drawing = TerminatorDrawing.from_feature_file(feature_file, FeatureType.IS_G)
    terminator_drawing.write_svg(out_g)

    terminator_drawing = TerminatorDrawing.from_feature_file(feature_file, FeatureType.IS_C)
    terminator_drawing.write_svg(out_c)
