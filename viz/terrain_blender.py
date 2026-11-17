"""3D spectrogram: the hearing threshold as a wire canopy, the poison rising beneath it.

    blender -b --python viz/terrain_blender.py -- out/vibeace [--still 120]

Reads <run>/viz/terrain.npz, saves <run>/viz/terrain.blend, and renders 1080x1920
PNG frames to <run>/viz/terrain_frames/. Axes: X = frequency (low to high, log),
Y = time (into the screen), Z = loudness in dB SPL.
"""

import sys
from pathlib import Path

import bpy
import numpy as np

args = sys.argv[sys.argv.index("--") + 1:]
run = Path(args[0]).resolve()
still = int(args[args.index("--still") + 1]) if "--still" in args else None

WIDTH, LENGTH, HEIGHT = 4.0, 9.0, 2.4
FRAMES, RISE_START, RISE_END = 180, 25, 110
SURFACE = (0.0103, 0.0103, 0.0097)  # #1a1a19, linear
BLUE = (0.0409, 0.2423, 0.7835)  # #3987e5
ORANGE = (0.6939, 0.0999, 0.0194)  # #d95926
EMBER = (0.08, 0.008, 0.002)

data = np.load(run / "viz" / "terrain.npz")
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


def emission(name, build):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    links.new(emit.outputs[0], out.inputs["Surface"])
    build(nodes, links, emit)
    return mat


def flat(color, strength):
    def build(nodes, links, emit):
        emit.inputs["Color"].default_value = (*color, 1)
        emit.inputs["Strength"].default_value = strength
    return build


def by_height(nodes, links, emit):
    """Dark ember in the valleys, bright orange on the peaks, so the surface reads as a shape."""
    geo = nodes.new("ShaderNodeNewGeometry")
    xyz = nodes.new("ShaderNodeSeparateXYZ")
    rng = nodes.new("ShaderNodeMapRange")
    rng.inputs["From Min"].default_value = HEIGHT * 0.2
    rng.inputs["From Max"].default_value = HEIGHT * 0.6
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*EMBER, 1)
    ramp.color_ramp.elements[1].color = (*ORANGE, 1)
    links.new(geo.outputs["Position"], xyz.inputs[0])
    links.new(xyz.outputs["Z"], rng.inputs["Value"])
    links.new(rng.outputs[0], ramp.inputs[0])
    links.new(ramp.outputs["Color"], emit.inputs["Color"])
    emit.inputs["Strength"].default_value = 1.6


def terrain(name, grid, material):
    n_freq, n_time = grid.shape
    xs = np.linspace(-WIDTH / 2, WIDTH / 2, n_freq)
    ys = np.linspace(0, LENGTH, n_time)
    verts = [(xs[f], ys[t], grid[f, t] / 96.0 * HEIGHT) for t in range(n_time) for f in range(n_freq)]
    faces = [(t * n_freq + f, t * n_freq + f + 1, (t + 1) * n_freq + f + 1, (t + 1) * n_freq + f)
             for t in range(n_time - 1) for f in range(n_freq - 1)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


canopy = terrain("HearingThreshold", data["threshold"], emission("ThresholdWire", flat(BLUE, 1.4)))
wire = canopy.modifiers.new("Wire", "WIREFRAME")
wire.thickness = 0.006

poison = terrain("Poison", data["poison"], emission("PoisonGlow", by_height))
for poly in poison.data.polygons:
    poly.use_smooth = True
poison.scale.z = 0.001
poison.keyframe_insert("scale", frame=RISE_START)
poison.scale.z = 1.0
poison.keyframe_insert("scale", frame=RISE_END)

floor = bpy.data.meshes.new("Floor")
floor.from_pydata([(-WIDTH / 2, 0, 0), (WIDTH / 2, 0, 0), (WIDTH / 2, LENGTH, 0), (-WIDTH / 2, LENGTH, 0)],
                  [(0, 1), (1, 2), (2, 3), (3, 0)], [])
floor_obj = bpy.data.objects.new("FloorOutline", floor)
scene.collection.objects.link(floor_obj)

aim = bpy.data.objects.new("Aim", None)
scene.collection.objects.link(aim)
aim.location = (0, LENGTH * 0.5, HEIGHT * 0.3)

cam_data = bpy.data.cameras.new("Camera")
cam_data.lens = 25
cam = bpy.data.objects.new("Camera", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
track = cam.constraints.new("TRACK_TO")
track.target = aim
track.track_axis, track.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
for frame, loc in ((1, (0.0, -2.5, 10.5)), (RISE_END, (1.6, -5.0, 7.0)), (FRAMES, (3.2, -6.0, 4.6))):
    cam.location = loc
    cam.keyframe_insert("location", frame=frame)

world = bpy.data.worlds.new("World")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (*SURFACE, 1)
scene.world = world

scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x, scene.render.resolution_y = 1080, 1920
scene.render.resolution_percentage = 100
scene.render.fps = 30
scene.frame_start, scene.frame_end = 1, FRAMES
scene.view_settings.view_transform = "Standard"
scene.render.image_settings.file_format = "PNG"

bpy.ops.wm.save_as_mainfile(filepath=str(run / "viz" / "terrain.blend"))

if still is not None:
    scene.frame_set(still)
    scene.render.filepath = str(run / "viz" / f"terrain_still_{still:03d}.png")
    bpy.ops.render.render(write_still=True)
else:
    scene.render.filepath = str(run / "viz" / "terrain_frames" / "f_")
    bpy.ops.render.render(animation=True)
