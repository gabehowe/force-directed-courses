from __future__ import annotations
import operator
import sys
from functools import reduce
from typing import List, Dict
import tkinter as tk
from PIL import Image, ImageDraw, ImageColor, ImageFont, ImageTk
import numpy as np

import yaml
from dataclasses import dataclass, asdict

filename = "course_description.yml"

depth_segments = 1


@dataclass
class PhysicalNode:
    pos: np.ndarray
    code: str
    link_codes: List[str]
    depth: int

    def dist(self, other: np.ndarray):
        return np.sqrt(np.sum((self.pos - other) ** 2))
        # return np.sum(np.abs(self.pos - other))

    def angle(self, other: np.ndarray):
        d = other - self
        return np.atan2(d[1], d[0])

    @property
    def weight(self):
        return 1


@dataclass
class Box:
    children: List[PhysicalNode | Box]
    pos: np.ndarray
    bounds: np.ndarray
    weight: int
    width: float


@dataclass
class Course:
    code: str
    title: str
    prereqs: List[str | tuple]
    flags: int
    credit: int

    def __post_init__(self):
        self.prereqs = list(filter(lambda it: len(it) > 0, self.prereqs))


def input_courses():
    code, title = input("Code: ").strip().upper(), input("Title: ").strip(),

    prerequisites = [i.strip() for i in input("Prereqs: ").strip().split(",")]
    prerequisites = [tuple([e.strip() for e in i.split('/')]) if '/' in i else i for i in prerequisites]

    flags = [int(i.strip()) for i in input("flags (1cbe, 2dc): ").strip().split()]
    credit = int(input("credits: ").strip())
    bitwise = reduce(operator.or_, flags)
    course = Course(code, title, prerequisites, bitwise, credit)
    with open(filename) as file:
        data = yaml.safe_load(file)
    data[code] = asdict(course)
    with open(filename, "w+") as file:
        yaml.safe_dump(data, file)
    input_courses()


def load_courses() -> Dict[str, Course]:
    with open(filename) as file:
        courses = yaml.safe_load(file)
    return {k: Course(**v) for k, v in courses.items()}


def broken_courses():
    courses = load_courses()
    broken = []
    for v in courses.values():
        for i in v.prereqs:
            if isinstance(i, str):
                if i not in courses.keys():
                    broken.append(i)
            else:
                if not any([e in courses.keys() for e in i]):
                    broken.append(i)

    broken = set(broken)
    if len(broken) != 0:
        print("Unsatisifed Prerequisites!")
        print(", ".join(broken))


def find_depth(key, codes) -> int:
    depths = [0]
    for child in codes[key]:
        v = child if isinstance(child, list) else [child]
        for e in v:
            if e not in codes.keys():
                continue
            depths.append(1 + find_depth(e, codes))
    return max(depths)


def draw_bounds(bounds: List, draw: ImageDraw, min_size, max_size, imagesize):
    # if isinstance(bounds, np.ndarray):
    #     scale_factor = bounds[-1] * 2 ** 1
    #     draw.rectangle(((bounds[:-1] * scale_factor + 0.5) * imagesize).tolist(), outline="blue")
    # # else:
    for i in bounds:
        if i is None:
            continue
        # scale_factor = bounds[-1] * 2 ** 1
        p1 = np.array(i[:2])
        p2 = np.array(i[2:])
        p1 = coords_to_px(p1, min_size, max_size, imagesize)
        p2 = coords_to_px(p2, min_size, max_size, imagesize)
        draw.rectangle(np.concat((p1, p2)).tolist(), outline="blue")


def coords_to_px(coords, min_size, max_size, imagesize):
    return (coords - min_size) / (max_size - min_size) * (imagesize * 0.8) + imagesize * 0.1


def bounds_list(current_arr, tree: Box):
    for i in tree.children:
        if isinstance(i, Box):
            current_arr.append(i.bounds)
            bounds_list(current_arr, i)
    return current_arr


def render_frame(i, min_size, max_size, font):
    im = Image.new("RGBA", (750, 750), (17, 17, 17))
    draw = ImageDraw.Draw(im)
    for index in range(len(i.values())):
        node = list(i.values())[index]
        for k in node.link_codes:
            v = k
            if isinstance(k, str):
                v = [v]
            for j in v:
                if j in i.keys():
                    p1 = coords_to_px(node.pos, min_size, max_size, im.width)
                    p2 = coords_to_px(i[j].pos, min_size, max_size, im.width)
                    # vector_values = np.array((node.pos, i[j].pos)) * 500 + 0.5 * im.width
                    vector_values = np.array((p1, p2)).flatten().tolist()
                    brightness = int(.1 * 255)
                    draw.line(vector_values, fill=(brightness, brightness, brightness, brightness), width=2)

    # draw_bounds(bounds_arr[n], draw, min_size, max_size, im.width)
    for index in range(len(i.values())):
        node = list(i.values())[index]
        # screen_pos = list(node.pos * 500 + 0.5 * im.width)
        screen_pos = coords_to_px(node.pos, min_size, max_size, im.width)
        draw.circle(screen_pos, 3, ImageColor.getrgb(f'hsv({node.depth * 50 % 360}, 100%, 100%)'))

    for index in range(len(i.values())):
        node = list(i.values())[index]
        screen_pos = coords_to_px(node.pos, min_size, max_size, im.width)
        draw.text([screen_pos[0], screen_pos[1] + 5], node.code, "white", font)

    for i in range(max([e.depth for e in list(i.values())]) + 1):
        bw = 20
        pos = [i * bw, 0, (i + 1) * bw, bw]
        draw.rectangle(pos, ImageColor.getrgb(f'hsv({i * 50 % 360}, 100%, 100%)'))
        draw.text((np.array(pos) + 5).tolist(), str(i), 'black', font=font)
    return im


def force_directed_graph(codes: Dict[str, List[str]]):
    global depth_segments
    root = tk.Tk()
    canvas = tk.Canvas(root, width=500, height=500)
    canvas.pack()
    depths = {i: find_depth(i, codes) for i in codes.keys()}
    depth_segments = 1 / max(depths.values())
    nodes = [PhysicalNode(
        np.array([(np.random.random() * depth_segments + depth_segments * depths[k]), np.random.random()]) - 0.5, k, v,
        depths[k]) for k, v in codes.items()]
    frames: List[Dict[str, PhysicalNode]] = []
    bounds_arr = []
    font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 15)
    fc = 1000
    for i in range(fc):
        tree = simulate(nodes)
        bounds = bounds_list([], tree)

        if i % 10 == 0:
            print(f'{i / fc * 100:.1f}%')
            frames.append(
                {n.code: PhysicalNode(n.pos.copy(), n.code, n.link_codes.copy(), depth=n.depth) for n in nodes})
            bounds_arr.append(bounds)
    images = []
    min_size = np.min([np.min([k.pos for i in range(len(frames)) for k in frames[i].values()], 0)], 0)
    max_size = np.max([np.max([k.pos for i in range(len(frames)) for k in frames[i].values()], 0)], 0)
    for n in range(len(frames)):
        images.append(render_frame(frames[n], min_size, max_size, font).convert("RGBA"))
    images[0].save("out.gif", save_all=True, append_images=images[1:], duration=60, loop=0)


def build_quadtree(nodes, bounds: np.ndarray):
    tree = Box([], (bounds[2:] + bounds[:2]) / 2, bounds, 0, np.sqrt(np.sum((bounds[2:] - bounds[:2]) ** 2)))
    subnodes = list(filter(lambda it: np.all(np.logical_and(bounds[:2] < it.pos, it.pos < bounds[2:])), nodes))
    if len(subnodes) > 1:
        for x in range(0, 2):
            for y in range(0, 2):
                # (min, min + width/2)
                # (min + width/2, min + width)
                dimensions = (bounds[:2] - bounds[2:]) / 2
                subbounds = (bounds[2:] + np.array(
                    [[(1 + x) * dimensions[0], (1 + y) * dimensions[1]],
                     [x * dimensions[0], y * dimensions[1]]])).flatten()
                subtree = build_quadtree(subnodes, subbounds)
                tree.weight += subtree.weight
                tree.children.append(subtree)
    elif len(subnodes) == 1:
        return Box(subnodes, subnodes[0].pos, bounds, 1, 0.05)
    return tree


depth_factor = 0.7


def create_comparison_points(node, tree: Box | PhysicalNode, array: list, depth):
    if isinstance(tree, PhysicalNode):
        array.append([*tree.pos, tree.weight])
        return array
    for i in tree.children:
        ds = node.dist(i.pos)
        if ds < 0.001:  # try to ignore self
            continue
        v = i.width if isinstance(i, Box) else 0.005
        if v / ds < depth_factor:
            array.append([i.pos[0], i.pos[1], i.weight])
        else:
            create_comparison_points(node, i, array, depth + 1)
    return array


def calculate_force(weight: int, pos: np.ndarray, opos: np.ndarray):
    ds = np.sqrt(np.sum((pos - opos) ** 2))
    return (pos - opos) / ds * (weight * opposite_push) / ds ** 2


def calculate_force_arr(opoints: np.ndarray, pos: np.ndarray):
    delta = pos - opoints[:, [0, 1]]
    ds = np.sqrt(np.sum(delta ** 2, axis=1)).reshape(-1, 1)
    force = delta / ds * (opoints[:, [2]] * opposite_push) / ds ** 2
    return force


opposite_push = 0.00002
gravity = [-0.015, -0.003]
link_push = -0.10


def simulate(nodes: List[PhysicalNode]):
    positions = np.array([it.pos for it in nodes])
    # tree, bounds = build_quadtree(nodes, np.concatenate((np.min(positions, 0), np.max(positions, 0))), 1)
    minimum = np.min(np.array([it.pos for it in nodes]), 0) - 0.001
    maximum = np.max(positions, 0) + 0.001
    tree = build_quadtree(nodes, np.concat((minimum, maximum)))
    for i in nodes:
        # pull of the center -- should be proportional to the distance
        # center_pull =  i.pos/i.dist(np.zeros(2)) * gravity/i.dist(np.zeros(2))
        depth_center = np.array([((depth_segments * 0.9) * i.depth) - 0.35, 0])
        # depth_center = np.zeros(2)
        center_force = (i.pos - depth_center) * gravity
        other_force = np.zeros(2)
        opoints = np.array(create_comparison_points(i, tree, [], 0))
        forces = calculate_force_arr(opoints, i.pos)
        other_force += np.sum(forces, axis=0)

        # test_force = 0
        # for o in opoints:
        #     test_force += calculate_force(o[2], i.pos, o[:2])

        for o in nodes:
            if o.code == i.code:
                continue
            if o.code in i.link_codes:
                # or i.code in o.link_codes -- make courses attracted to classes that require them
                other_force += (i.pos - o.pos) * i.dist(o.pos) * link_push
        total_force = center_force + other_force
        i.pos += total_force
    return tree


def chart_courses():
    broken_courses()
    courses = load_courses()
    roots = []
    for v in courses.values():
        if v.flags & 1 != 0 or v.flags & 2 != 0:
            continue
        satisfied = True
        for i in v.prereqs:
            q = [i] if isinstance(i, str) else i
            if any([e in courses.keys() and (courses[e].flags & 1 != 0 or courses[e].flags & 2 != 0) for e in q]):
                continue
            satisfied = False
        if satisfied:
            roots.append(v)

    print("\n".join([i.code + ' ' + i.title for i in roots]))
    force_directed_graph({k: v.prereqs for k, v in filter(lambda it: it[1].flags == 0, courses.items())})


# Store courses in file with references to previously required courses
# Sort by first required.
# somehow visualize -- matplotlib?
# create simple text input method to input courses

if __name__ == '__main__':
    match sys.argv[1].strip():
        case "-i":
            input_courses()
        case "-c":
            chart_courses()
            # 3.88x speed increase
        case "-b":
            broken_courses()
