import numpy as np
import cv2
import math

camera_matrix = np.array([
    [1221.445, 0.0, 637.226],
    [0.0, 1223.398, 502.549],
    [0.0, 0.0, 1.0]
], dtype=np.float32)
dist_coeffs = np.array([0.177168, -0.457341, 0.000360, 0.002753, 0.178259], dtype=np.float32)

w_obj, h_obj = 0.0889, 0.0381
object_points = np.array([
    [-w_obj/2, -h_obj/2, 0],
    [ w_obj/2, -h_obj/2, 0],
    [ w_obj/2,  h_obj/2, 0],
    [-w_obj/2,  h_obj/2, 0]
], dtype=np.float32)

FRAME_HEIGHT = 240.0
FRAME_WIDTH = 320.0
FOV_VERTICAL_HALF = 20.5
FOV_HORIZONTAL_HALF = 27.0

MIN_AREA_AT_TOP = 50.0
MIN_AREA_AT_BOTTOM = 200.0
MAX_AREA_AT_TOP = -75.0
MAX_AREA_AT_BOTTOM = 1500.0

CAMERA_LENS_POSITION_WORLD = np.array([-0.16063272254, -0.1277051126, 0.38390702339], dtype=np.float64)
POINT_IN_FRONT_OF_LENS_WORLD = np.array([-0.15260885545, -0.12585040949, 0.37823448808], dtype=np.float64)
POINT_TO_RIGHT_OF_LENS_WORLD = np.array([-0.15838062025, -0.13744821453, 0.38390702272], dtype=np.float64)
OBJECT_PLANE_Z_WORLD = 0.0

pts = np.array([
    [int(FRAME_WIDTH * 0.4), int(FRAME_HEIGHT * 0.3)],
    [int(FRAME_WIDTH * 0.7), int(FRAME_HEIGHT * 0.2)],
    [int(FRAME_WIDTH * 0.8), int(FRAME_HEIGHT * 0.9)],
    [int(FRAME_WIDTH * 0.1), int(FRAME_HEIGHT * 0.9)],
], np.int32)

polygon = pts.reshape((-1, 1, 2))

def _normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return v / norm

def get_camera_rotation_matrix(cam_pos_world, pt_in_front_world, pt_to_right_world):
    C_w = np.array(cam_pos_world, dtype=np.float64)
    F_w = np.array(pt_in_front_world, dtype=np.float64)
    R_pt_w = np.array(pt_to_right_world, dtype=np.float64)

    try:
        vec_z_cam_world = _normalize_vector(F_w - C_w)
        vec_to_right_def_point = R_pt_w - C_w
        if np.linalg.norm(vec_to_right_def_point) < 1e-9:
            raise ValueError("Camera position and right-defining point are too close.")

        x_dir_component = (R_pt_w - C_w) - np.dot(R_pt_w - C_w, vec_z_cam_world) * vec_z_cam_world
        vec_x_cam_world = _normalize_vector(x_dir_component)
        vec_y_cam_world = _normalize_vector(np.cross(vec_z_cam_world, vec_x_cam_world))

    except ValueError as e:
        print(f"Error calculating camera rotation matrix: {e}")
        return None

    R_cam_to_world = np.column_stack((vec_x_cam_world, vec_y_cam_world, vec_z_cam_world))
    return R_cam_to_world

def calculate_angle(contour):
    if len(contour) < 5:
        return 0.0
    (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
    return angle

def round_to_discrete_angle(angle_deg):
    thresholds = [0, 15, 30, 45, 60, 75, 90]
    abs_angle = abs(angle_deg)
    for t in thresholds:
        if abs_angle <= t:
            rounded = t
            break
    rounded_signed = math.copysign(rounded, angle_deg)
    return rounded_signed

def get_sample_position_python(llpython_output, color_param,
                               camera_pos_world, pt_in_front_world, pt_to_right_world,
                               plane_z_world):
    try:
        R_cam_to_world = get_camera_rotation_matrix(camera_pos_world, pt_in_front_world, pt_to_right_world)
        if R_cam_to_world is None:
            print("Failed to get camera rotation matrix in get_sample_position_python.")
            return None

        tx_degrees = llpython_output[9]
        ty_degrees = llpython_output[10]

        tx_rad = np.deg2rad(tx_degrees)
        ty_rad = np.deg2rad(ty_degrees)

        x_cam_local = np.tan(tx_rad)
        y_cam_local = -np.tan(ty_rad)

        ray_direction_camera_local = np.array([x_cam_local, y_cam_local, 1.0], dtype=np.float64)
        vectorDir_world_unnormalized = R_cam_to_world @ ray_direction_camera_local
        vectorDir_world = _normalize_vector(vectorDir_world_unnormalized)

        if abs(vectorDir_world[2]) < 1e-9:
            print("Ray is parallel to the target plane or points away from it in Z (get_sample_position_python).")
            return None

        t = (plane_z_world - camera_pos_world[2]) / vectorDir_world[2]

        x_inters_raw = camera_pos_world[0] + vectorDir_world[0] * t
        y_inters = camera_pos_world[1] + vectorDir_world[1] * t

        llpython_output[4] = x_inters_raw - 0.05
        llpython_output[5] = y_inters

        angle_input_from_llpython3 = llpython_output[3]
        processed_angle = angle_input_from_llpython3 % 180.0
        if processed_angle > 90.0:
            processed_angle -= 180.0

        final_angle_component = processed_angle - 10.0

        conditional_value = 0.0
        if math.isclose(llpython_output[5], color_param):
            conditional_value = llpython_output[0]

        return [final_angle_component, conditional_value]

    except ValueError as e:
        print(f"Error in get_sample_position_python: {e}")
        return None
    except IndexError:
        print("Error: llpython_output does not have expected length for indices in get_sample_position_python.")
        return None
def calculate_contour_world_coords(cx_contour, cy_contour, cam_matrix, dist_coeffs_cam,
                                   cam_translation_world, pt_in_front_world, pt_to_right_world,
                                   obj_plane_z_world=0.0):
    R_cam_to_world = get_camera_rotation_matrix(cam_translation_world, pt_in_front_world, pt_to_right_world)
    if R_cam_to_world is None:
        print("Failed to get camera rotation matrix in calculate_contour_world_coords.")
        return None

    C_w = np.array(cam_translation_world, dtype=np.float64)
    img_pt_distorted = np.array([[[float(cx_contour), float(cy_contour)]]], dtype=np.float32)
    norm_cam_coords_undistorted = cv2.undistortPoints(img_pt_distorted, cam_matrix, dist_coeffs_cam, None, None)

    if norm_cam_coords_undistorted is None or norm_cam_coords_undistorted.shape[0] == 0:
        print("Undistortion failed in calculate_contour_world_coords.")
        return None

    x_n, y_n = norm_cam_coords_undistorted[0, 0, 0], norm_cam_coords_undistorted[0, 0, 1]
    ray_direction_camera = np.array([x_n, y_n, 1.0], dtype=np.float64)
    ray_direction_world_unnormalized = R_cam_to_world @ ray_direction_camera
    try:
        ray_direction_world = _normalize_vector(ray_direction_world_unnormalized)
    except ValueError:
        print("Normalization of world ray direction failed in calculate_contour_world_coords.")
        return None

    if abs(ray_direction_world[2]) < 1e-9:
        return None

    t = (obj_plane_z_world - C_w[2]) / ray_direction_world[2]
    if t < 0:
        return None

    intersection_point_world = C_w + t * ray_direction_world
    return intersection_point_world.tolist()

def process_color(frame, mask):
    debug_info = None
    kernel = np.ones((3, 3), np.uint8)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)  if 1 else frame
    gray_masked = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)  if 1 else masked_frame
    gray_boosted = cv2.addWeighted(gray_masked, 1.5, mask, 0.5, 0)  if 1 else gray_masked
    blurred = cv2.GaussianBlur(gray_boosted, (3, 3), 0) if 1 else gray_boosted

    sobelx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=1)
    sobely = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=1)

    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.uint8(magnitude * 255 / np.max(magnitude))

    _, edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY) if 1 else magnitude

    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) if 1 else edges
    edges = cv2.bitwise_not(edges) if 1 else edges
    edges = cv2.bitwise_and(edges, edges, mask=mask) if 1 else edges
    edges = cv2.GaussianBlur(edges, (3, 3), 0) if 1 else edges
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy, gray_masked, False, debug_info

def debug_return(frame):
    return np.array([[]]), frame, [0, 0, 0, 0, 0, 0, 0, 0]

def pipeline_debug_return(frame):
    return None, None, None, True, frame

def runPipeline(image, llrobot):
    output_image = image.copy()
    img_height, img_width = image.shape[:2]
    center_x_frame = img_width / 2.0
    center_y_frame = img_height / 2.0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red_1 = np.array([0, 100, 100])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 100, 100])
    upper_red_2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, yellow_hierarchy, yellow_gray, isDebug, debug_info = process_color(image, mask)
    if isDebug:
        return debug_return(debug_info)

    detected_objects_info = []
    valid_contours_to_draw = []
    min_crosshair_distance = float('inf')
    closest_to_crosshair_idx = -1
    closest_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 1.0:
            continue
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx_obj = M["m10"] / M["m00"]
        cy_obj = M["m01"] / M["m00"]

        point = (cx_obj, cy_obj)
        is_inside = cv2.pointPolygonTest(polygon, point, False) >= 0

        if not is_inside:
            continue

        y_pos_contour = np.clip(cy_obj, 0.0, FRAME_HEIGHT)
        y_ratio = y_pos_contour / FRAME_HEIGHT
        current_min_area = MIN_AREA_AT_TOP * (1 - y_ratio) + MIN_AREA_AT_BOTTOM * y_ratio
        current_max_area = MAX_AREA_AT_TOP * (1 - y_ratio) + MAX_AREA_AT_BOTTOM * y_ratio
        current_min_area = max(1.0, current_min_area)

        if current_min_area <= area <= current_max_area:
            rect = cv2.minAreaRect(contour)
            rect_width, rect_height = rect[1]
            angle_from_ellipse = calculate_angle(contour)
            if angle_from_ellipse > 95.0:
                mapped_angle = angle_from_ellipse - 180.0
            else:
                mapped_angle = angle_from_ellipse - 8
            mapped_angle = max(-90.0, min(mapped_angle, 90.0))

            rounded_mapped_angle = round_to_discrete_angle(mapped_angle)

            if min(rect_width, rect_height) <= 1e-3:
                continue
            aspect_ratio = max(rect_width, rect_height) / min(rect_width, rect_height)

            if 1.0 < aspect_ratio < 6.0:
                valid_contours_to_draw.append(contour)
                distance_to_crosshair = np.sqrt((cx_obj - center_x_frame) ** 2 + (cy_obj - center_y_frame) ** 2)
                if distance_to_crosshair < min_crosshair_distance:
                    min_crosshair_distance = distance_to_crosshair
                    closest_to_crosshair_idx = len(detected_objects_info)
                    closest_contour = contour

                current_object_info = {
                    'area': float(area),
                    'cx_img': float(cx_obj),
                    'cy_img': float(cy_obj),
                    'distance_to_crosshair': float(distance_to_crosshair),
                    'angle': float(rounded_mapped_angle),
                    'raw_ellipse_angle': float(angle_from_ellipse),
                    'rect_width': float(rect_width),
                    'rect_height': float(rect_height),
                    'success_pnp': False
                }
                if len(contour) >= 4:
                    box_points_cv = cv2.boxPoints(rect)
                    box_sorted_y = sorted(list(box_points_cv), key=lambda pt: pt[1])
                    top_points_sorted_x = sorted(box_sorted_y[:2], key=lambda pt: pt[0])
                    tl_img, tr_img = top_points_sorted_x[0], top_points_sorted_x[1]
                    bottom_points_sorted_x = sorted(box_sorted_y[2:], key=lambda pt: pt[0])
                    bl_img, br_img = bottom_points_sorted_x[0], bottom_points_sorted_x[1]
                    image_points = np.array([tl_img, tr_img, br_img, bl_img], dtype=np.float32)
                    success_pnp, rvec, tvec = cv2.solvePnP(
                        object_points,
                        image_points,
                        camera_matrix,
                        dist_coeffs
                    )
                    current_object_info['success_pnp'] = bool(success_pnp)
                    if success_pnp:
                        current_object_info.update({
                            'tvec_x': float(tvec[0][0]),
                            'tvec_y': float(tvec[1][0]),
                            'tvec_z': float(tvec[2][0]),
                            'rvec_x': float(rvec[0][0]),
                            'rvec_y': float(rvec[1][0]),
                            'rvec_z': float(rvec[2][0])
                        })
                detected_objects_info.append(current_object_info)

    cv2.drawContours(output_image, valid_contours_to_draw, -1, (0, 255, 0), 2)
    if closest_contour is not None:
        cv2.drawContours(output_image, [closest_contour], -1, (255, 0, 0), 2)

    world_coords_projection = None
    degrees_x, degrees_y = 0.0, 0.0

    llpython = [0.0] * 11
    llpython[1] = 2.0

    if closest_to_crosshair_idx != -1 and closest_to_crosshair_idx < len(detected_objects_info):
        closest_obj_data = detected_objects_info[closest_to_crosshair_idx]
        cx_closest_obj, cy_closest_obj = closest_obj_data['cx_img'], closest_obj_data['cy_img']

        world_coords_projection = calculate_contour_world_coords(
            cx_contour=cx_closest_obj,
            cy_contour=cy_closest_obj,
            cam_matrix=camera_matrix,
            dist_coeffs_cam=dist_coeffs,
            cam_translation_world=CAMERA_LENS_POSITION_WORLD,
            pt_in_front_world=POINT_IN_FRONT_OF_LENS_WORLD,
            pt_to_right_world=POINT_TO_RIGHT_OF_LENS_WORLD,
            obj_plane_z_world=OBJECT_PLANE_Z_WORLD
        )

        if FRAME_WIDTH > 0 and FRAME_HEIGHT > 0:
            degrees_per_pixel_x = (2 * FOV_HORIZONTAL_HALF) / FRAME_WIDTH
            degrees_per_pixel_y = (2 * FOV_VERTICAL_HALF) / FRAME_HEIGHT
            degrees_x = (cx_closest_obj - center_x_frame) * degrees_per_pixel_x
            degrees_y = (center_y_frame - cy_closest_obj) * degrees_per_pixel_y

        llpython[0] = 1.0

        if closest_obj_data.get('success_pnp', False):
            llpython[2] = closest_obj_data.get('tvec_y', 0.0)
            llpython[3] = closest_obj_data.get('tvec_z', 0.0)

        if world_coords_projection is not None:
            llpython[4] = float(world_coords_projection[0])
            llpython[5] = float(world_coords_projection[1])
            llpython[6] = float(world_coords_projection[2])
        else:
            llpython[6] = OBJECT_PLANE_Z_WORLD

        llpython[7] = closest_obj_data['area']
        llpython[8] = closest_obj_data['angle']
        llpython[9] = degrees_x
        llpython[10] = degrees_y

        example_color_param = 0.0

        gsp_other_results = get_sample_position_python(
            llpython_output=llpython,
            color_param=example_color_param,
            camera_pos_world=CAMERA_LENS_POSITION_WORLD,
            pt_in_front_world=POINT_IN_FRONT_OF_LENS_WORLD,
            pt_to_right_world=POINT_TO_RIGHT_OF_LENS_WORLD,
            plane_z_world=OBJECT_PLANE_Z_WORLD
        )

        llpython[4] -= 0.035

        if gsp_other_results:
            final_angle_comp_from_gsp = gsp_other_results[0]
            conditional_val_from_gsp = gsp_other_results[1]
        else:
            print("get_sample_position_python failed. llpython[4] and llpython[5] retain values from projection method (if any).")

    print("----- Final llpython Array Values -----")
    print(f"[0] Has Target: {llpython[0]}")
    print(f"[1] Pipeline Color: {llpython[1]}")
    print(f"[2] tvec_y (PnP): {llpython[2]}")
    print(f"[3] tvec_z (PnP): {llpython[3]}")
    print(f"[4] World X (Final): {llpython[4]}")
    print(f"[5] World Y (Final): {llpython[5]}")
    print(f"[6] World Z (Plane): {llpython[6]}")
    print(f"[7] Area: {llpython[7]}")
    print(f"[8] Angle (−90..90, visual, rounded): {llpython[8]}")
    print(f"[9] Degrees X (tx): {llpython[9]}")
    print(f"[10] Degrees Y (ty): {llpython[10]}")
    print("-------------------------------------")

    if closest_contour is None:
        return np.array([], dtype=np.int32), output_image, llpython
    else:
        return closest_contour, output_image, llpython