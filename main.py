import operator
import sys
from functools import reduce
from typing import List, Dict

from PIL import Image, ImageDraw, ImageColor, ImageFont
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

    def angle(self, other: np.ndarray):
        d = other - self
        return np.atan2(d[1], d[0])


@dataclass
class Box:
    children: List[PhysicalNode]
    centroid: np.ndarray
    bounds: np.ndarray
    weight: int


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


def draw_bounds(bounds: List, draw: ImageDraw, imagesize):
    if isinstance(bounds, np.ndarray):
        scale_factor = bounds[-1] * 2 ** 1
        draw.rectangle(((bounds[:-1] * scale_factor + 0.5) * imagesize).tolist(), outline="blue")
    else:
        for i in bounds:
            if i is None:
                continue
            draw_bounds(i, draw, imagesize)

def coords_to_px(coords, min_size, max_size, imagesize):
    return (coords - min_size) / (max_size-min_size) * (imagesize * 0.8) + imagesize * 0.1

def force_directed_graph(codes: Dict[str, List[str]]):
    global depth_segments
    depths = {i: find_depth(i, codes) for i in codes.keys()}
    depth_segments = 1 / max(depths.values())
    nodes = [PhysicalNode(
        np.array([(np.random.random() * depth_segments + depth_segments * depths[k]), np.random.random()]) - 0.5, k, v,
        depths[k]) for k, v in codes.items()]
    frames: List[Dict[str, PhysicalNode]] = []
    bounds_arr = []
    font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 15)
    fc = 4000
    for i in range(fc):
        bounds = simulate(nodes)
        if i % 100 == 0:
            print(f'{i / fc * 100:.1f}%')
            frames.append(
                {n.code: PhysicalNode(n.pos.copy(), n.code, n.link_codes.copy(), depth=n.depth) for n in nodes})
            bounds_arr.append(bounds)
    images = []
    min_size = np.min([np.min([k.pos for i in range(len(frames)) for k in frames[i].values()],0)],0)
    max_size = np.max([np.max([k.pos for i in range(len(frames)) for k in frames[i].values()],0)],0)
    for n in range(len(frames)):
        i = frames[n]
        im = Image.new("RGBA", (750, 750), (17, 17, 17))
        draw = ImageDraw.Draw(im)
        # draw_bounds(bounds_arr[n], draw, im.width)
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
                        vector_values = np.array((p1,p2)).flatten().tolist()
                        draw.line(vector_values, fill=(100, 100, 100, 100), width=2)

        for index in range(len(i.values())):
            node = list(i.values())[index]
            # screen_pos = list(node.pos * 500 + 0.5 * im.width)
            screen_pos = coords_to_px(node.pos, min_size, max_size, im.width)
            draw.circle(screen_pos, 3, ImageColor.getrgb(f'hsv({node.depth * 50 % 360}, 100%, 100%)'))

        for index in range(len(i.values())):
            node = list(i.values())[index]
            screen_pos = coords_to_px(node.pos, min_size, max_size, im.width)
            draw.text([screen_pos[0], screen_pos[1] + 5], node.code, "white", font)

        images.append(im.convert("RGBA"))
    images[0].save("out.gif", save_all=True, append_images=images[1:], duration=100, loop=0)


def build_quadtree(nodes, bounds: np.ndarray, depth):
    tree = np.zeros(4, dtype=object)
    # nodes = filter(lambda it: it.pos[0] in range(bounds[0], bounds[2]) and it.pos[1] in range(bounds[1], bounds[3]),
    #                nodes)
    boundsl: List[object] = [None for i in range(4)]
    for x in range(0, 2):
        for y in range(0, 2):
            coordinate = y * 2 + x
            width = abs(bounds[0] - bounds[2]) / 2
            new_bounds = np.concat(
                (np.array((x, y)) * width + bounds[:2], np.array((x, y)) * width + bounds[:2] + [width, width]))
            local_nodes = list(filter(
                lambda it: new_bounds[0] < it.pos[0] < new_bounds[2] and new_bounds[1] < it.pos[1] < new_bounds[3],
                nodes))
            if len(local_nodes) > 0:
                if depth > 0:
                    tree[coordinate], boundsl[coordinate] = build_quadtree(local_nodes, new_bounds, depth - 1)
                else:
                    tree[coordinate] = Box(local_nodes, new_bounds[:2] + width, new_bounds, len(local_nodes))
                    boundsl[coordinate] = np.append(new_bounds, width)
    return tree, boundsl


grid_size = 8


# def build_grid(nodes, min, max):
#     grid = [[Box([], np.array((x, y)) / grid_size * (max-min + 0.01) + min, np.array(()), 0) for y in range(grid_size)] for x in
#             range(grid_size)]
#     for i in nodes:
#         coordinate = np.floor(((i.pos - min) / (max + 0.01 - min)) * grid_size)
#         grid[int(coordinate[0])][int(coordinate[1])].children.append(i)
#     return grid


def calculate_force(weight: int, pos: np.ndarray, opos: np.ndarray):
    opposite_push = 0.00002
    ds = np.sqrt(np.sum((pos - opos) ** 2))
    return (pos - opos) / ds * (weight * opposite_push) / ds ** 2

def simulate(nodes: List[PhysicalNode]):
    positions = np.array([it.pos for it in nodes])
    # tree, bounds = build_quadtree(nodes, np.concatenate((np.min(positions, 0), np.max(positions, 0))), 1)
    bounds = []
    minimum = np.min(np.array([it.pos for it in nodes]), 0)
    maximum = np.max(positions, 0)
    # grid = build_grid(nodes, minimum, maximum)
    gravity = [-0.007, -0.002]
    link_push = -0.02
    out_of_bounds_scalar = 10
    bound = 70000 # effectively disable
    # for i in nodes:
    #     coordinate = np.array(np.floor((i.pos+0.5) * grid_size/2)).tolist()
    #     grid[int(coordinate[0])][int(coordinate[1])].append(i)
    for i in nodes:
        coordinate = np.floor(((i.pos - minimum) / (maximum + 0.01 - minimum)) * grid_size)
        # pull of the center -- should be proportional to the distance
        # center_pull =  i.pos/i.dist(np.zeros(2)) * gravity/i.dist(np.zeros(2))
        depth_center = np.array([((depth_segments * 0.9) * i.depth) - 0.35, 0])
        center_force = (i.pos - depth_center) * gravity
        if abs(i.pos[0]) > bound or abs(i.pos[1]) > bound:
            center_force *= out_of_bounds_scalar
        other_force = np.zeros(2)
        # for x in range(grid_size):
        #     for y in range(grid_size):
        #         if x == coordinate[0] and y == coordinate[1]:
        #             other_force += sum([calculate_force(1, i.pos, o.pos) for o in grid[x][y].children])
        #         else:
        #             other_force += calculate_force(grid[x][y].weight, i.pos, grid[x][y].centroid)


        # for x in range(-1,2):
        #     if coordinate[0] + x not in range(0,grid_size):
        #         continue
        #     for y in range(-1,2):
        #         if coordinate[1] + y not in range(0,grid_size):
        #             continue
        #         for o in grid[int(x + coordinate[0])][int(y + coordinate[1])]:  # calculate the push that other nodes have on this node
        for o in nodes:
            if o.code == i.code:
                continue
            other_force += calculate_force(1, i.pos, o.pos)
            if o.code in i.link_codes:
                # or i.code in o.link_codes -- make courses attracted to classes that require them
                other_force += (i.pos - o.pos) * i.dist(o.pos) * link_push
        total_force = center_force + other_force
        i.pos += total_force
    return bounds


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
        case "-b":
            broken_courses()
