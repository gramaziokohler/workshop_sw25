import math

from compas.geometry import Frame, Transformation, Vector, Brep, Curve, NurbsCurve

from compas_timber.elements import Beam
from compas_timber.fabrication import JackRafterCut, JackRafterCutProxy, StepJoint, StepJointNotch
from compas_timber.fabrication import Lap, LapProxy
from compas_timber.fabrication import BTLxProcessing

from compas_rhino.conversions import frame_to_rhino_plane
from compas_rhino.geometry import RhinoNurbsSurface

from Rhino.Geometry import CurveOffsetCornerStyle # type: ignore


# Spiral paths start out from the center of the slice and move outwards if this is enabled
USE_CENTER_OUT_CUTTING = False
Z_FIGHTING_OFFSET = 0.0001
CURVE_OFFSET_STYLE = CurveOffsetCornerStyle.NONE
ADD_SAFE_FRAMES = True


def get_toolpath_from_lap_processing(beam: Beam, 
                                     processing: BTLxProcessing,
                                     machining_transformation: Transformation = None,
                                     machining_frame: Frame = None,
                                     tool_radius: float = 0.2,
                                     stepdown: float = 0.05,
                                     min_step: float = None,
                                     approach_height: float = 0.5,
                                     tolerance : float = 1e-3,
                                     **kwargs):
    volume = processing.volume_from_params_and_beam(beam)
    volume_at_origin = volume.transformed(machining_transformation)

    e = Brep.from_mesh(volume_at_origin.to_mesh())
    
    slices = []
    slicing_frames = []

    levels = int(beam.height / stepdown) + 1

    for i in range(levels):
        frame = machining_frame.copy()
        frame.point += frame.zaxis * ((stepdown * i) + Z_FIGHTING_OFFSET)
        slicing_frames.append(frame)
        slices += e.slice(frame)

    radius = tool_radius / 2
    offset_step = radius * -1

    flat_spirals = []
    spirals = []
    
    for current_slice, slicing_frame in zip(slices, slicing_frames):
        num_offsets = int((beam.width / 2) / radius) - 1
        current_offset = current_slice
        slice_offsets = [current_offset]

        for i in range(num_offsets):
            slicing_plane = frame_to_rhino_plane(slicing_frame)
            native_offset = current_offset.native_curve.Offset(slicing_plane, offset_step, tolerance, CURVE_OFFSET_STYLE)
            current_offset = Curve.from_native(native_offset[0])
            
            slice_offsets.append(current_offset)
            flat_spirals.append(current_offset)
        
        spirals.append(slice_offsets)

    path = []

    for slice_offsets, slicing_frame in zip(spirals, slicing_frames):
        next_slice_start_point = None

        # Spiral paths can start from the center and move outwards or vice versa
        if USE_CENTER_OUT_CUTTING:
            arranged_offset_spirals = slice_offsets
        else:
            arranged_offset_spirals = reversed(slice_offsets)

        for slice_offset in arranged_offset_spirals:
            for subcurve in slice_offset.native_curve.GetSubCurves():
                segment = NurbsCurve.from_native(subcurve)

                max_divisions = max(1, int(segment.length() / min_step))
                _params, points = segment.divide_by_count(max_divisions, return_points=True)

                if next_slice_start_point is None:
                    next_slice_start_point = points[0]
                for point in points:
                    frame = slicing_frame.copy()
                    frame.point = point
                    path.append(frame)

        # Move in the same frame to the start of the next slice
        if next_slice_start_point:
            frame = slicing_frame.copy()
            frame.point = next_slice_start_point
            path.append(frame)

    if ADD_SAFE_FRAMES:
        # We negate the approach height to have the approach vector point outwards
        # from the beam because Z axis points inwards
        approach_vector = path[0].zaxis * -approach_height
        path = add_safe_frames(path, approach_vector)

    return "subtraction", path, volume_at_origin, flat_spirals, slicing_frames


def get_toolpath_for_plane_cut(beam: Beam, 
                               blank_brep_at_origin: Brep, 
                               frame: Frame,
                               machining_frame: Frame = None,
                               tool_radius: float = 0.2,
                               min_step: float = None,
                               approach_height: float = 0.5,
                               flip_direction: bool = False) -> list[Frame]:
    slices = blank_brep_at_origin.slice(frame)
    if len(slices) != 1:
        raise ValueError("Expected exactly one slice from the blank at the machining plane.")
    
    slice_surface = RhinoNurbsSurface.from_corners(slices[0].points[0:4])

    path = []
    radius = tool_radius / 2
    num_steps = int((beam.height / 2) / radius) - 1
    isocurves = []

    # Determine the U/V curve direction (most aligned with world X axis)
    p1 = slice_surface.isocurve_u(0).point_at(0)
    p2 = slice_surface.isocurve_u(0).point_at(1)
    isocurve_u_vector = (p2 - p1).unitized()
    dot_u = abs(isocurve_u_vector.dot(Vector(1, 0, 0)))

    p1 = slice_surface.isocurve_v(0).point_at(0)
    p2 = slice_surface.isocurve_v(0).point_at(1)
    isocurve_v_vector = (p2 - p1).unitized()
    dot_v = abs(isocurve_v_vector.dot(Vector(1, 0, 0)))

    direction = dot_u > dot_v
    if flip_direction:
        direction = not direction

    if direction:
        linear_spacer = slice_surface.space_u
        isocurve_selector = slice_surface.isocurve_u
    else:
        linear_spacer = slice_surface.space_v
        isocurve_selector = slice_surface.isocurve_v

    params = linear_spacer(num_steps)

    for i, param in enumerate(params):
        isocurve = isocurve_selector(param)

        max_divisions = max(1, int(isocurve.length() / min_step))
        _params, points = isocurve.divide_by_count(max_divisions, return_points=True)
        if i % 2 == 1:
            points.reverse()

        for point in points:
            frame = machining_frame.copy()
            frame.point = point
            path.append(frame)

        isocurves.append(isocurve)

    if ADD_SAFE_FRAMES:
        start_end_vector = isocurves[0].point_at(0) - isocurves[-1].point_at(0)
        start_end_vector.unitize()
        approach_vector = start_end_vector * approach_height
        path = add_safe_frames(path, approach_vector)

    return path, slice_surface, isocurves


def get_toolpath_from_jackraftercut_processing(beam: Beam, 
                                               processing: BTLxProcessing,
                                               machining_transformation: Transformation = None,
                                               machining_frame: Frame = None,
                                               tool_radius: float = 0.2,
                                               stepdown: float = 0.05,
                                               min_step: float = None,
                                               approach_height: float = 0.5,
                                               flip_direction: bool = False,
                                               tolerance : float = 1e-3,
                                               **kwargs):
    plane = processing.plane_from_params_and_beam(beam)
    plane_at_origin = plane.transformed(machining_transformation)
    blank_brep_at_origin = beam.blank.to_brep().transformed(machining_transformation)

    frame = Frame.from_plane(plane_at_origin)
    slices = blank_brep_at_origin.slice(frame)
    if len(slices) != 1:
        raise ValueError("Expected exactly one slice from the blank at the machining plane.")

    path, slice_surface, isocurves = get_toolpath_for_plane_cut(beam, blank_brep_at_origin, frame, machining_frame=machining_frame, tool_radius=tool_radius, min_step=min_step, approach_height=approach_height, flip_direction=flip_direction)

    return "cut", path, slice_surface, isocurves


def add_safe_frames(path: list[Frame], approach_vector: Vector) -> list[Frame]:
    # Add safe approach and retract frames to the toolpath
    safe_approach = path[0].copy()
    safe_approach.point += approach_vector
    path.insert(0, safe_approach)

    safe_retract = path[-1].copy()
    safe_retract.point += approach_vector
    path.append(safe_retract)

    return path


def get_toolpath_from_processing(beam: Beam, processing: BTLxProcessing, machining_transformation: Transformation, machining_side: int, **kwargs):
    # Automatically pick machining side if not specified (-1)
    machining_side = machining_side if machining_side != -1 else processing.ref_side_index
    machining_frame = beam.ref_sides[machining_side].transformed(machining_transformation)

    # We flip the machining frame to point Z axis inwards into the beam
    # matching what would likely be the TCP frame of the machine
    machining_frame.yaxis *= -1

    toolpath_function = None

    if isinstance(processing, (Lap, LapProxy)):
        toolpath_function = get_toolpath_from_lap_processing
    elif isinstance(processing, (JackRafterCut, JackRafterCutProxy)):
        toolpath_function = get_toolpath_from_jackraftercut_processing

    if toolpath_function:
        return toolpath_function(beam, processing, machining_transformation, machining_frame=machining_frame, **kwargs)
