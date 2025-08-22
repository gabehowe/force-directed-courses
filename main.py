from __future__ import annotations
import operator
import sys
from functools import reduce
from typing import List, Dict
import tkinter as tk
from PIL import Image, ImageDraw, ImageColor, ImageFont, ImageTk
import numpy as np
import os
import time
import itertools

import yaml
from dataclasses import dataclass, asdict

filename = "course_description.yml"

depth_segments = 1
type CourseData = Dict[str, Course]
type TreeData = List[PhysicalNode | Box]


@dataclass
class PhysicalNode:
    """Physical node analogue for force directed graphs."""
    pos: np.ndarray
    code: str
    link_codes: List[str]
    depth: int

    def dist(self, other: np.ndarray):
        return np.hypot(*(self.pos - other))

    def angle(self, other: np.ndarray):
        d = other - self
        return np.atan2(d[1], d[0])

    @property
    def weight(self):
        return 1


@dataclass
class Box:
    """Quadtree node for reducing computation in force directed comparisons."""
    children: TreeData
    pos: np.ndarray
    bounds: np.ndarray
    weight: int
    width: float

@dataclass
class Course:
    """Data representation of a course."""
    code: str
    title: str
    prereqs: List[str | tuple]
    flags: int
    credit: int

    def __post_init__(self):
        # remove empty lists
        self.prereqs = list(filter(lambda it: len(it) > 0, self.prereqs))

def input_courses():
   """
   Asks for and saves user input.
   """

   code, title = input("Code: ").strip().upper(), input("Title: ").strip().upper()
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


def load_courses(filepath: os.path.PathLike) -> Dict[str, Course]:
    """
    Loads courses from the file global file
    Returns:
        a mapping of each course ID to each course object.
    """
    with open(filepath) as file:
        courses = yaml.safe_load(file)
    return {k: Course(**v) for k, v in courses.items()}


def broken_courses(filepath: os.path.PathLike):
    """
    Prints courses with prerequisites unrepresented in the data.
    Args:
        filepath: The path to load the courses from.
    """
    courses = load_courses(filepath)
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


def find_depth(key: str, codes: Dict[str, List[str]]) -> int:
    """
    The number of prerequisites of a node.
    Args:
        key: The course key of the node we want to check.
        codes: The course data.
    Returns:
        The depth of the node.
    """
    depths = [0]
    for child in codes[key]:
        v = child if isinstance(child, list) else [child]
        for e in v:
            if e not in codes.keys():
                continue
            depths.append(1 + find_depth(e, codes))
    return max(depths)


def draw_bounds(bounds: List, draw: ImageDraw, min_size: np.ndarray, max_size: np.ndarray, image_size: int):
    """
    Draws the bounds of the quadtree for debugging.
    Args:
        bounds: The bounds of the quadtree sections.
        draw: The drawing object to draw on.
        min_size: the minimum point of the bounds.
        max_size: the maximum point of the bounds.
        image_size: the side length of the square image.
    """
    for i in bounds:
        if i is None:
            continue
        # scale_factor = bounds[-1] * 2 ** 1
        p1 = np.array(i[:2])
        p2 = np.array(i[2:])
        p1 = coords_to_px(p1, min_size, max_size, image_size)
        p2 = coords_to_px(p2, min_size, max_size, image_size)
        draw.rectangle(np.concat((p1, p2)).tolist(), outline="blue")


def coords_to_px(coords: np.ndarray, min_size: np.ndarray, max_size: np.ndarray, image_size: int) -> np.ndarray:
    """
    Converts graph coordinates to pixel positions.
    Args:
        coords: the coordinate to convert.
        min_size: the minimum point of the graph.
        max_size: the maximum point of the graph.
        image_size: the side length of the square image.
    Returns:
        The converted pixel coordinates.
    """
    return (coords - min_size) / (max_size - min_size) * (image_size * 0.8) + image_size * 0.1


def bounds_list(current_arr: List[np.ndarray], tree: Box) -> List[np.ndarray]:
    """
    Recursively creates a list of bounds from the bound quadtree.
    Args:
        current_arr: The array being built.
        tree: The tree to build the array from.
    Returns:
        current_arr, but only after it's built.
    """
    for i in tree.children:
        if isinstance(i, Box):
            current_arr.append(i.bounds)
            bounds_list(current_arr, i)
    return current_arr


def render_frame(frame_data: Dict[str, PhysicalNode], min_size: np.ndarray, max_size: np.ndarray, font: ImageFont.Font) -> Image.Image:
    """
    Render a single frame from stored data
    Args:
        frame_data: The stored graph state for this frame.
        min_size: The minimum point of the graph.
        max_size: The maximum point of the graph.
        font: The font to use when rendering the names of the nodes.
    Returns:
        The rendered image frame
    """
    im = Image.new("RGBA", (750, 750), (17, 17, 17))
    draw = ImageDraw.Draw(im)
    for index in range(len(frame_data.values())):
        node = list(frame_data.values())[index]
        for k in node.link_codes:
            v = k
            if isinstance(k, str):
                v = [v]
            for j in v:
                if j in frame_data.keys():
                    p1 = coords_to_px(node.pos, min_size, max_size, im.width)
                    p2 = coords_to_px(frame_data[j].pos, min_size, max_size, im.width)
                    # vector_values = np.array((node.pos, i[j].pos)) * 500 + 0.5 * im.width
                    vector_values = np.array((p1, p2)).flatten().tolist()
                    brightness = int(.1 * 255)
                    draw.line(vector_values, fill=(brightness, brightness, brightness, brightness), width=2)

    # draw_bounds(bounds_arr[n], draw, min_size, max_size, im.width)
    for index in range(len(frame_data.values())):
        node = list(frame_data.values())[index]
        # screen_pos = list(node.pos * 500 + 0.5 * im.width)
        screen_pos = coords_to_px(node.pos, min_size, max_size, im.width)
        radius = 200 / np.hypot(*(max_size - min_size))
        draw.circle(screen_pos, radius, ImageColor.getrgb(f'hsv({node.depth * 50 % 360}, 100%, 100%)'))

    for index in range(len(frame_data.values())):
        node = list(frame_data.values())[index]
        screen_pos = coords_to_px(node.pos, min_size, max_size, im.width)
        draw.text([screen_pos[0], screen_pos[1] + 5], node.code, "white", font)

    for i in range(max([e.depth for e in list(frame_data.values())]) + 1):
        bw = 20
        pos = [i * bw, 0, (i + 1) * bw, bw]
        draw.rectangle(pos, ImageColor.getrgb(f'hsv({i * 50 % 360}, 100%, 100%)'))
        draw.text((np.array(pos) + 5).tolist(), str(i), 'black', font=font)
    return im


def force_directed_graph(codes: Dict[str, List[str]]):
    """
    Runs a force directed graph simulation of the codes to approximate a correct order and draws it with tk.
    Args:
        codes: A mapping of the course titles to their prerequisites.
    """
    global depth_segments
    # TODO: stop using Image.
    root = tk.Tk()
    im = Image.new("RGBA", (750, 750), (17, 17, 17))

    depths = {i: find_depth(i, codes) for i in codes.keys()}
    depth_segments = .25 / max(depths.values())
    nodes = [PhysicalNode(
        np.array([(np.random.random() * depth_segments + depth_segments * depths[k]), np.random.random()]) - 0.5, k, v,
        depths[k]) for k, v in codes.items()]
    frames: List[Dict[str, PhysicalNode]] = []
    bounds_arr = []
    font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 15)

    tkimg = ImageTk.PhotoImage(im)
    label = tk.Label(root, image=tkimg)
    coords  = [ 0,0 ]


    # bind to events.
    def set_motion(e):
        coords[0]=e.x
        coords[1]=e.y
    def set_exit(e): # because of poor python lambda support.
        should_exit[0] = True

    should_exit = [False]
    root.bind("<Motion>", set_motion)
    root.bind("<Escape>", set_exit)
    label.pack(side="bottom")

    images = []
    frame_time = 0
    i=0
    while not should_exit[0]:
        i+=1
        times =[time.perf_counter_ns()]

        size = np.array((root.winfo_height(), root.winfo_height()))
        cursor_pos =  (np.array(coords) - 80) * 1.25

        frame =  {n.code: PhysicalNode(n.pos.copy(), n.code, n.link_codes.copy(), depth=n.depth) for n in nodes}
        frames.append(frame)
        min_size = np.min([np.min([k.pos for k in frame.values()], 0)], 0)
        max_size = np.max([np.max([k.pos for k in frame.values()], 0)], 0)
        graph_cursor_pos = ((cursor_pos / size * (max_size - min_size)) + min_size)
        # print(size, cursor_pos, graph_cursor_pos)
        times.append(time.perf_counter_ns())
        tree = simulate(nodes,frame_time, graph_cursor_pos)
        times.append(time.perf_counter_ns())
        bounds = bounds_list([], tree)

        frame['cursor'] = PhysicalNode(graph_cursor_pos, "cursor", [], 0)
        times.append(time.perf_counter_ns())
        images.append(render_frame(frame, min_size, max_size, font).convert("RGBA"))
        times.append(time.perf_counter_ns())
        del frame['cursor']

        tkimg = ImageTk.PhotoImage(images[-1])
        label.configure(image = tkimg)

        bounds_arr.append(bounds)
        root.update()

        times.append(time.perf_counter_ns())
        frame_time = (times[-1] - times[0])/1_000_000
        # print(",".join([str((times[i] - times[i-i])/1_000_000) for i in range(1, len(times))]))


    # for n in range(len(frames)):
    #     images.append(render_frame(frames[n], min_size, max_size, font).convert("RGBA"))
    images[0].save("out.gif", save_all=True, append_images=images[1:], duration=60, loop=0)


def build_quadtree(nodes: List[PhysicalNode | Box], bounds: np.ndarray) -> Box:
    """
    Builds a quadtree from a list of nodes for gravity approximation.
    Args:
        nodes: The nodes in their current position.
        bounds: The minimum bounding rectangle of the nodes.
    Returns:
        The built quadtree box.
    """
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


def create_comparison_points(node: PhysicalNode, tree: Box | PhysicalNode, array: list, depth: int) -> List[float]:
    """
    Recursively creates approximated masses for use in calculation relative to the input node.
    Args:
        node: The node we want to create weights relative to.
        tree: The tree we want to compare against.
        array: The array of approximate masses that we will build.
        depth: The current depth of the search into the tree.
    Returns:
        An array of points in the format [point xpos, point ypos, point mass]
    """
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


def calculate_force(weight: float, pos: np.ndarray, opos: np.ndarray) -> np.ndarray:
    """
    Calculates the force between pos and opos based on their distance and forces using (almost)
    Newton's law of universal gravitation.
    Args:
        weight: The mass of the objects.
        pos: The position of the first object.
        opos: The position of the second object.
    Returns:
        A 2d force vector.
    """
    ds = np.hypot(*(pos - opos))
    return (pos - opos) / ds * (weight * opposite_push) / ds ** 2


def calculate_force_arr(opoints: np.ndarray, pos: np.ndarray):
    """
    Calculates the forces between every element in opoints and pos.
    Args:
        opoints: A list of xy coordinates with weights in the format [x,y,weight].
        pos: The point to compare against.
    Returns:
        An array of 2d force vectors.
    """
    delta = pos - opoints[:, [0, 1]]
    ds = np.sqrt(np.sum(delta ** 2, axis=1)).reshape(-1, 1)
    # f = (p1-p2)/(||p1 - p2||) * (mc)/||p1-p2||^2
    #   = mc(p1-p2)/||p1-p2||^3. Too aggressive!
    # force = delta / ds * (opoints[:, [2]] * opposite_push) / ds ** 2
    #   => mc(p1-p2)/||p1-p2||^2
    m = opoints[:, [2]]
    force = m*opposite_push*delta/ds**2

    return force

# constants which can be changed manually for different results
opposite_push = 0.1
gravity = [-.035, -0.03]
link_push = .9
spring_length = 50


def calculate_log_attractive_force(desired_length: float, p1: np.ndarray, p2: np.ndarray):
	"""
	Calculates the spring force based on the log of the distance between two points.
	Args:
		desired_length: The desired distance between the two points to settle at.
		p1: The first point.
		p2: The second point.
	Returns:
		A 2d force vector.
	References:
		https://youtu.be/gJiSvGbH0CA?si=Uq5vodxAltA6grg9	
		A heuristic for graph drawing, Peter Eades
	"""
	delta = p2-p1
	return np.log(np.hypot(*delta))/desired_length * delta
	


def simulate(nodes: List[PhysicalNode], frame_time: float, cursor_pos: np.ndarray) -> Box:
    """
    Updates the node force simulation.
    Args:
        nodes: The nodes to update.
        frame_time: The time since the last frame.
        cursor_pos: The position of the cursor.
    Returns:
        The updated tree of nodes.
    """
    positions = np.array([it.pos for it in nodes])
    minimum = np.min(np.array([it.pos for it in nodes]), 0) - 0.001
    maximum = np.max(positions, 0) + 0.001
    tree = build_quadtree(nodes, np.concat((minimum, maximum)))
    for i in nodes:
        # pull of the center -- should be proportional to the distance
        depth_center = np.array([((depth_segments * 0.9) * i.depth) - 0.35, 0])
        center_force = (i.pos - depth_center) * gravity
        other_force = np.zeros(2)
        opoints = np.array(create_comparison_points(i, tree, [], 0))
        forces = calculate_force_arr(opoints, i.pos)
        other_force += np.sum(forces, axis=0)

        for o in nodes:
            if o.code == i.code:
                continue
            if o.code in itertools.chain.from_iterable(i.link_codes) or i.code in itertools.chain.from_iterable(o.link_codes):
                # or i.code in o.link_codes -- make courses attracted to classes that require them
                other_force += calculate_log_attractive_force(spring_length, i.pos, o.pos) * link_push

        cursor_force = calculate_force(1, i.pos, cursor_pos)
        total_force = center_force + other_force + cursor_force
        mv = 100
        if np.sum(total_force) > mv:
            # force is too high by manhattan distance
            print("Force too high!")
        i.pos += np.clip(total_force * frame_time / 30, -mv, mv)
    return tree


def chart_courses(filepath: os.path.PathLike):
    """
    Draw all the courses.
    Args:
        filepath: The path to from which to load courses.
    """
    broken_courses(filepath)
    courses = load_courses(filepath)
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
    try: 
        match sys.argv[1].strip():
            case "-i":
                input_courses()
            case "-c":
                chart_courses(filename)
                # 3.88x speed increase
            case "-b":
                broken_courses(filename)
    except IndexError:
        print("Args must be one of -i, -c, -b", file=sys.stderr)
