import cv2
import numpy as np
import serial
import serial.tools.list_ports
import time
import threading
import logging

from flask import Flask, Response, request, jsonify


# ============================================================
# SYSTEM
# ============================================================

stop_event = threading.Event()
logging.getLogger("werkzeug").setLevel(logging.ERROR)


# ============================================================
# SERIAL
# ============================================================

SERIAL_BAUD = 115200
RECONNECT_INTERVAL = 1.0

ser = None
ser_lock = threading.Lock()


# ============================================================
# CAMERA
# ============================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
CAMERA_FPS = 30

cap = None

latest_frame = None
frame_lock = threading.Lock()


# ============================================================
# SPEED
# ============================================================

# 直線最高 AUTO 速度
AUTO_MAX_SPEED = 0.24

# 中度轉向
AUTO_MID_SPEED = 0.21

# 大轉向
AUTO_TURN_SPEED = 0.18

# 手動
MANUAL_DRIVE_SPEED = 0.30


# ============================================================
# LOST SPEED
# ============================================================

LOST_SPEED_1 = 0.15
LOST_SPEED_2 = 0.10


# ============================================================
# LOST TIME
# ============================================================

LOST_STAGE1_TIME = 0.35
LOST_STOP_TIME = 0.65


# ============================================================
# STEERING CONTROL
#
# 這版改成比較不會發散
# ============================================================

DEAD_ZONE = 5

# P：主要修正力道
STEERING_KP = 0.62

# D：接近中心時提早收手
STEERING_KD = 0.045

# 最大 AUTO steering
MAX_STEERING = 28

# steering 平滑
STEERING_ALPHA = 0.38

# 每幀 steering 最大變化量
MAX_STEER_STEP = 4

smoothed_steering = 0.0
last_valid_steering = 0

previous_offset = None
filtered_offset_rate = 0.0
last_control_time = None

# derivative 濾波
OFFSET_RATE_ALPHA = 0.30


# ============================================================
# IMAGE PROCESSING
# ============================================================

BLACK_THRESHOLD = 70

MIN_LINE_WIDTH = 2
MAX_LINE_WIDTH = 36

NUM_SLICES = 24
MIN_VALID_PAIRS = 5

LEFT_SEARCH_MIN = 0
LEFT_SEARCH_MAX = 175

RIGHT_SEARCH_MIN = 145
RIGHT_SEARCH_MAX = 319

MIN_LANE_WIDTH = 45
MAX_LANE_WIDTH = 310

MAX_POINT_JUMP = 40


# ============================================================
# ROI
# ============================================================

ROI_TOP = 0.02
ROI_BOTTOM = 0.80


# ============================================================
# LINE FIT
# ============================================================

FIT_DEGREE = 1

MIN_REAL_POINTS = 5

EXTENSION_LIMIT = 60


# ============================================================
# CONTROL POSITION
#
# 稍微往前看
# ============================================================

TARGET_Y_RATIO = 0.58


# ============================================================
# LANE CENTER SMOOTH
#
# 不要太慢，也不要完全相信單幀
# ============================================================

LANE_CENTER_ALPHA = 0.48

last_lane_center_filtered = None


# ============================================================
# MODE
# ============================================================

MODE_AUTO = "AUTO"
MODE_MANUAL = "MANUAL"

current_mode = MODE_MANUAL


# ============================================================
# MANUAL
# ============================================================

manual_command = "STOP"
manual_speed = MANUAL_DRIVE_SPEED
manual_steering = 0

manual_lock = threading.Lock()


# ============================================================
# SERIAL RATE
# ============================================================

MANUAL_KEEPALIVE = 0.25

# AUTO 保持快
AUTO_KEEPALIVE = 0.10

AUTO_STEERING_CHANGE = 1

last_auto_steering = None
last_auto_speed = None
last_auto_send_time = 0.0


# ============================================================
# LOG
# ============================================================

TERMINAL_PRINT_INTERVAL = 0.50
last_lane_print = 0.0


# ============================================================
# LANE STATE
# ============================================================

last_real_lane_time = time.time()

lane_stop_sent = False


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# FIND ESP32
# ============================================================

def find_esp32_port():

    ports = serial.tools.list_ports.comports()

    for port in ports:

        device = port.device

        description = (
            port.description
            or ""
        ).lower()

        manufacturer = (
            port.manufacturer
            or ""
        ).lower()

        if (
            "cp210" in description
            or "cp210" in manufacturer
            or "ch340" in description
            or "ch341" in description
            or "usb serial" in description
            or "silicon labs" in manufacturer
            or "ttyusb" in device.lower()
            or "ttyacm" in device.lower()
        ):

            return device

    return None


# ============================================================
# CONNECT SERIAL
# ============================================================

def connect_serial():

    global ser

    while not stop_event.is_set():

        port = find_esp32_port()

        if port is None:

            print("[SERIAL] ESP32 not found")

            stop_event.wait(
                RECONNECT_INTERVAL
            )

            continue

        try:

            print(
                "[SERIAL] Connecting:",
                port
            )

            new_ser = serial.Serial(
                port,
                SERIAL_BAUD,
                timeout=0.005,
                write_timeout=0.2
            )

            time.sleep(1.0)

            try:

                new_ser.reset_input_buffer()
                new_ser.reset_output_buffer()

            except Exception:

                pass

            with ser_lock:

                if ser is not None:

                    try:
                        ser.close()

                    except Exception:
                        pass

                ser = new_ser

            print(
                "[SERIAL] ESP32 connected:",
                port
            )

            return True

        except Exception as e:

            print(
                "[SERIAL] Connect error:",
                e
            )

            stop_event.wait(
                RECONNECT_INTERVAL
            )

    return False


# ============================================================
# SEND SERIAL
# ============================================================

def send_serial(message):

    global ser

    with ser_lock:

        if ser is None:
            return False

        try:

            ser.write(
                (
                    message +
                    "\n"
                ).encode("utf-8")
            )

            return True

        except Exception as e:

            print(
                "[SERIAL] Write error:",
                e
            )

            try:
                ser.close()

            except Exception:
                pass

            ser = None

            return False


# ============================================================
# SERIAL READER
# ============================================================

def serial_reader():

    global ser

    last_status_print = 0.0

    while not stop_event.is_set():

        try:

            if ser is None:

                if not connect_serial():
                    break

                continue

            line = ""

            with ser_lock:

                if (
                    ser is not None
                    and
                    ser.is_open
                    and
                    ser.in_waiting > 0
                ):

                    line = (
                        ser.readline()
                        .decode(
                            "utf-8",
                            errors="ignore"
                        )
                        .strip()
                    )

            if not line:

                stop_event.wait(
                    0.005
                )

                continue

            if (
                "READY" in line
                or
                "ERROR" in line
                or
                "EMERGENCY" in line
            ):

                print(
                    "[ESP32]",
                    line
                )

            elif line.startswith("STATUS"):

                now = time.time()

                if (
                    now -
                    last_status_print
                    >=
                    TERMINAL_PRINT_INTERVAL
                ):

                    print(
                        "[MOTOR]",
                        line
                    )

                    last_status_print = now

        except Exception as e:

            if stop_event.is_set():
                break

            print(
                "[SERIAL] Read error:",
                e
            )

            with ser_lock:

                try:

                    if ser is not None:
                        ser.close()

                except Exception:
                    pass

                ser = None

            stop_event.wait(
                RECONNECT_INTERVAL
            )


# ============================================================
# MOTOR COMMAND
# ============================================================

def send_auto(
    speed,
    steering
):

    send_serial(
        "CTRL,{:.3f},{}".format(
            speed,
            int(steering)
        )
    )


def send_stop():

    send_serial(
        "CTRL,0.000,999"
    )


def send_manual(
    direction,
    speed,
    steering
):

    send_serial(
        "MANUAL,{},{:.3f},{}".format(
            direction,
            speed,
            int(steering)
        )
    )


# ============================================================
# SMART AUTO SEND
# ============================================================

def smart_send_auto(
    speed,
    steering,
    force=False
):

    global last_auto_steering
    global last_auto_speed
    global last_auto_send_time

    now = time.time()

    should_send = False

    if force:

        should_send = True

    elif last_auto_steering is None:

        should_send = True

    elif last_auto_speed is None:

        should_send = True

    elif (
        abs(
            int(steering)
            -
            int(last_auto_steering)
        )
        >=
        AUTO_STEERING_CHANGE
    ):

        should_send = True

    elif (
        abs(
            speed -
            last_auto_speed
        )
        >=
        0.015
    ):

        should_send = True

    elif (
        now -
        last_auto_send_time
        >=
        AUTO_KEEPALIVE
    ):

        should_send = True

    if should_send:

        send_auto(
            speed,
            steering
        )

        last_auto_steering = int(
            steering
        )

        last_auto_speed = float(
            speed
        )

        last_auto_send_time = now


# ============================================================
# MANUAL WORKER
# ============================================================

def manual_worker():

    while not stop_event.is_set():

        if current_mode == MODE_MANUAL:

            with manual_lock:

                command = manual_command
                speed = manual_speed
                steering = manual_steering

            if command == "F":

                send_manual(
                    "F",
                    speed,
                    steering
                )

            elif command == "B":

                send_manual(
                    "B",
                    speed,
                    steering
                )

        stop_event.wait(
            MANUAL_KEEPALIVE
        )


# ============================================================
# IMAGE PREPROCESS
# ============================================================

def preprocess_image(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    otsu_threshold, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV
        +
        cv2.THRESH_OTSU
    )

    if (
        otsu_threshold < 20
        or
        otsu_threshold > 180
    ):

        _, binary = cv2.threshold(
            gray,
            BLACK_THRESHOLD,
            255,
            cv2.THRESH_BINARY_INV
        )

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    return binary


# ============================================================
# FIND LINE CENTER
# ============================================================

def find_line_center(
    binary,
    y,
    x_min,
    x_max,
    previous_x=None
):

    if (
        y < 0
        or
        y >= binary.shape[0]
    ):
        return None

    x_min = max(
        0,
        x_min
    )

    x_max = min(
        binary.shape[1],
        x_max
    )

    row = binary[
        y,
        x_min:x_max
    ]

    pixels = np.where(
        row > 0
    )[0]

    if len(pixels) == 0:
        return None

    groups = []

    start = pixels[0]
    previous = pixels[0]

    for x in pixels[1:]:

        if x - previous > 1:

            groups.append(
                (
                    start,
                    previous
                )
            )

            start = x

        previous = x

    groups.append(
        (
            start,
            previous
        )
    )

    candidates = []

    for start_x, end_x in groups:

        w = (
            end_x -
            start_x +
            1
        )

        if (
            w < MIN_LINE_WIDTH
            or
            w > MAX_LINE_WIDTH
        ):
            continue

        center = (
            x_min +
            start_x +
            w // 2
        )

        candidates.append(
            (
                center,
                w
            )
        )

    if not candidates:
        return None

    if previous_x is not None:

        candidates.sort(
            key=lambda p:
            abs(
                p[0] -
                previous_x
            )
        )

        best = (
            candidates[0][0]
        )

        if (
            abs(
                best -
                previous_x
            )
            >
            MAX_POINT_JUMP
        ):
            return None

        return best

    candidates.sort(
        key=lambda p:
        p[1],
        reverse=True
    )

    return candidates[0][0]


# ============================================================
# LINE FIT
# ============================================================

def fit_lane_line(points):

    if (
        len(points)
        <
        MIN_REAL_POINTS
    ):

        return None

    pts = np.array(
        points,
        dtype=np.float32
    )

    ys = pts[:, 1]
    xs = pts[:, 0]

    try:

        coeff = np.polyfit(
            ys,
            xs,
            FIT_DEGREE
        )

    except Exception:

        return None

    return np.poly1d(
        coeff
    )


# ============================================================
# DRAW FIT
# ============================================================

def draw_predicted_line(
    frame,
    poly,
    min_y,
    max_y
):

    previous_point = None

    start_y = max(
        0,
        int(min_y)
        -
        EXTENSION_LIMIT
    )

    end_y = min(
        FRAME_HEIGHT - 1,
        int(max_y)
        +
        EXTENSION_LIMIT
    )

    for y in range(
        start_y,
        end_y + 1,
        4
    ):

        try:
            x = int(poly(y))

        except Exception:
            continue

        if (
            x < 0
            or
            x >= FRAME_WIDTH
        ):
            continue

        cv2.circle(
            frame,
            (
                x,
                y
            ),
            2,
            (
                255,
                0,
                0
            ),
            -1
        )

        if previous_point is not None:

            cv2.line(
                frame,
                previous_point,
                (
                    x,
                    y
                ),
                (
                    255,
                    0,
                    0
                ),
                1
            )

        previous_point = (
            x,
            y
        )


# ============================================================
# DETECT LANE
#
# 回傳：
# frame
# offset
# real_valid
#
# real_valid=True：
# 這幀真的有可靠左右線
#
# real_valid=False：
# 只是沿用上一幀
# ============================================================

def detect_lane(frame):

    global last_lane_center_filtered

    height, width = (
        frame.shape[:2]
    )

    binary = preprocess_image(
        frame
    )

    top = int(
        height *
        ROI_TOP
    )

    bottom = int(
        height *
        ROI_BOTTOM
    )

    slice_height = max(
        1,
        (
            bottom -
            top
        )
        //
        NUM_SLICES
    )

    left_points = []
    right_points = []

    last_left = None
    last_right = None

    for i in range(
        NUM_SLICES
    ):

        y = (
            bottom -
            1 -
            i *
            slice_height
        )

        if y < top:
            break

        left_x = find_line_center(
            binary,
            y,
            LEFT_SEARCH_MIN,
            LEFT_SEARCH_MAX,
            last_left
        )

        right_x = find_line_center(
            binary,
            y,
            RIGHT_SEARCH_MIN,
            RIGHT_SEARCH_MAX,
            last_right
        )

        if left_x is not None:

            left_points.append(
                (
                    left_x,
                    y
                )
            )

            last_left = left_x

        if right_x is not None:

            right_points.append(
                (
                    right_x,
                    y
                )
            )

            last_right = right_x

    # REAL POINT = YELLOW
    for x, y in left_points:

        cv2.circle(
            frame,
            (x, y),
            3,
            (0, 255, 255),
            -1
        )

    for x, y in right_points:

        cv2.circle(
            frame,
            (x, y),
            3,
            (0, 255, 255),
            -1
        )

    target_y = int(
        height *
        TARGET_Y_RATIO
    )

    # ================================================
    # 不夠點 = 真 LOST
    # ================================================

    if (
        len(left_points)
        <
        MIN_VALID_PAIRS
        or
        len(right_points)
        <
        MIN_VALID_PAIRS
    ):

        return (
            frame,
            None,
            False
        )

    left_poly = fit_lane_line(
        left_points
    )

    right_poly = fit_lane_line(
        right_points
    )

    if (
        left_poly is None
        or
        right_poly is None
    ):

        return (
            frame,
            None,
            False
        )

    left_ys = [
        p[1]
        for p in left_points
    ]

    right_ys = [
        p[1]
        for p in right_points
    ]

    draw_predicted_line(
        frame,
        left_poly,
        min(left_ys),
        max(left_ys)
    )

    draw_predicted_line(
        frame,
        right_poly,
        min(right_ys),
        max(right_ys)
    )

    try:

        left_target = int(
            left_poly(
                target_y
            )
        )

        right_target = int(
            right_poly(
                target_y
            )
        )

    except Exception:

        return (
            frame,
            None,
            False
        )

    geometry_good = True

    if (
        left_target < 0
        or
        left_target >= width
        or
        right_target < 0
        or
        right_target >= width
    ):

        geometry_good = False

    if (
        geometry_good
        and
        left_target >= right_target
    ):

        geometry_good = False

    lane_width = 0

    if geometry_good:

        lane_width = (
            right_target -
            left_target
        )

        if (
            lane_width
            <
            MIN_LANE_WIDTH
            or
            lane_width
            >
            MAX_LANE_WIDTH
        ):

            geometry_good = False

    # ================================================
    # 幾何暫時怪掉
    # 有上一幀可以 HOLD
    # 但不能當成真的重新看到線
    # ================================================

    if not geometry_good:

        if last_lane_center_filtered is None:

            return (
                frame,
                None,
                False
            )

        lane_center = int(
            round(
                last_lane_center_filtered
            )
        )

        camera_center = (
            width // 2
        )

        offset = (
            lane_center -
            camera_center
        )

        cv2.circle(
            frame,
            (
                lane_center,
                target_y
            ),
            7,
            (
                0,
                165,
                255
            ),
            -1
        )

        cv2.putText(
            frame,
            "LANE HOLD",
            (
                10,
                22
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                0,
                165,
                255
            ),
            2
        )

        return (
            frame,
            offset,
            False
        )

    # ================================================
    # REAL CENTER
    # ================================================

    raw_center = (
        left_target +
        right_target
    ) / 2.0

    if last_lane_center_filtered is None:

        last_lane_center_filtered = (
            raw_center
        )

    else:

        last_lane_center_filtered = (
            LANE_CENTER_ALPHA
            *
            raw_center
            +
            (
                1.0 -
                LANE_CENTER_ALPHA
            )
            *
            last_lane_center_filtered
        )

    lane_center = int(
        round(
            last_lane_center_filtered
        )
    )

    camera_center = (
        width // 2
    )

    offset = (
        lane_center -
        camera_center
    )

    # target points
    cv2.circle(
        frame,
        (
            left_target,
            target_y
        ),
        5,
        (
            255,
            0,
            255
        ),
        -1
    )

    cv2.circle(
        frame,
        (
            right_target,
            target_y
        ),
        5,
        (
            255,
            0,
            255
        ),
        -1
    )

    # center
    cv2.circle(
        frame,
        (
            lane_center,
            target_y
        ),
        7,
        (
            0,
            255,
            0
        ),
        -1
    )

    cv2.line(
        frame,
        (
            camera_center,
            0
        ),
        (
            camera_center,
            height
        ),
        (
            255,
            255,
            255
        ),
        1
    )

    cv2.line(
        frame,
        (
            0,
            target_y
        ),
        (
            width,
            target_y
        ),
        (
            120,
            120,
            120
        ),
        1
    )

    cv2.putText(
        frame,
        "OFFSET: {}".format(
            offset
        ),
        (
            10,
            22
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (
            0,
            255,
            0
        ),
        2
    )

    cv2.putText(
        frame,
        "WIDTH: {}".format(
            lane_width
        ),
        (
            10,
            44
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (
            255,
            255,
            0
        ),
        1
    )

    cv2.putText(
        frame,
        "L:{} R:{}".format(
            len(left_points),
            len(right_points)
        ),
        (
            205,
            22
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (
            0,
            255,
            255
        ),
        1
    )

    return (
        frame,
        offset,
        True
    )


# ============================================================
# CALCULATE STEERING
#
# PD CONTROL
# ============================================================

def calculate_steering(offset):

    global previous_offset
    global filtered_offset_rate
    global last_control_time
    global smoothed_steering
    global last_valid_steering

    now = time.time()

    # ================================================
    # DEAD ZONE
    # ================================================

    if abs(offset) <= DEAD_ZONE:

        previous_offset = offset
        last_control_time = now

        filtered_offset_rate *= 0.5

        smoothed_steering *= 0.4

        if abs(smoothed_steering) < 1.0:
            smoothed_steering = 0.0

        steering = int(
            round(
                smoothed_steering
            )
        )

        last_valid_steering = steering

        return steering, 0, 0.0

    # ================================================
    # OFFSET RATE
    # ================================================

    rate = 0.0

    if (
        previous_offset is not None
        and
        last_control_time is not None
    ):

        dt = (
            now -
            last_control_time
        )

        if dt > 0.005:

            rate = (
                offset -
                previous_offset
            ) / dt

            # 防止單幀雜訊爆掉
            rate = max(
                -250.0,
                min(
                    250.0,
                    rate
                )
            )

    filtered_offset_rate = (
        OFFSET_RATE_ALPHA
        *
        rate
        +
        (
            1.0 -
            OFFSET_RATE_ALPHA
        )
        *
        filtered_offset_rate
    )

    previous_offset = offset
    last_control_time = now

    # ================================================
    # PD
    # ================================================

    control_value = (
        STEERING_KP
        *
        offset
        +
        STEERING_KD
        *
        filtered_offset_rate
    )

    raw_steering = int(
        round(
            -control_value
        )
    )

    raw_steering = max(
        -MAX_STEERING,
        min(
            MAX_STEERING,
            raw_steering
        )
    )

    # ================================================
    # SMOOTH
    # ================================================

    smoothed_steering = (
        STEERING_ALPHA
        *
        raw_steering
        +
        (
            1.0 -
            STEERING_ALPHA
        )
        *
        smoothed_steering
    )

    steering = int(
        round(
            smoothed_steering
        )
    )

    # ================================================
    # SLEW LIMIT
    # ================================================

    upper = (
        last_valid_steering
        +
        MAX_STEER_STEP
    )

    lower = (
        last_valid_steering
        -
        MAX_STEER_STEP
    )

    steering = max(
        lower,
        min(
            upper,
            steering
        )
    )

    steering = max(
        -MAX_STEERING,
        min(
            MAX_STEERING,
            steering
        )
    )

    last_valid_steering = steering

    return (
        steering,
        raw_steering,
        filtered_offset_rate
    )


# ============================================================
# DYNAMIC SPEED
# ============================================================

def calculate_auto_speed(
    steering,
    lane_is_real
):

    # HOLD 狀態不要高速跑
    if not lane_is_real:

        return LOST_SPEED_1

    s = abs(
        steering
    )

    if s <= 7:

        return AUTO_MAX_SPEED

    elif s <= 16:

        return AUTO_MID_SPEED

    else:

        return AUTO_TURN_SPEED


# ============================================================
# CAMERA
# ============================================================

def open_camera():

    global cap

    while not stop_event.is_set():

        print(
            "[CAMERA] Opening /dev/video0..."
        )

        cap = cv2.VideoCapture(
            CAMERA_INDEX
        )

        if (
            cap is not None
            and
            cap.isOpened()
        ):

            cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                FRAME_WIDTH
            )

            cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                FRAME_HEIGHT
            )

            cap.set(
                cv2.CAP_PROP_FPS,
                CAMERA_FPS
            )

            time.sleep(
                0.5
            )

            ret, frame = (
                cap.read()
            )

            if (
                ret
                and
                frame is not None
                and
                frame.size > 0
            ):

                print(
                    "[CAMERA] Camera OK"
                )

                return True

        print(
            "[CAMERA] Open failed"
        )

        try:

            if cap is not None:
                cap.release()

        except Exception:
            pass

        cap = None

        stop_event.wait(
            1.0
        )

    return False


# ============================================================
# CAMERA LOOP
# ============================================================

def camera_loop():

    global cap
    global latest_frame

    global last_lane_print
    global last_real_lane_time
    global lane_stop_sent

    global last_auto_steering
    global last_auto_speed
    global last_auto_send_time

    global last_lane_center_filtered

    global previous_offset
    global filtered_offset_rate
    global last_control_time

    print(
        "[CAMERA] Camera loop started"
    )

    if not open_camera():
        return

    cv2.namedWindow(
        "Lane Detection",
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        "Lane Detection",
        640,
        480
    )

    while not stop_event.is_set():

        ret, frame = (
            cap.read()
        )

        if (
            not ret
            or
            frame is None
        ):

            time.sleep(
                0.01
            )

            continue

        frame = cv2.resize(
            frame,
            (
                FRAME_WIDTH,
                FRAME_HEIGHT
            )
        )

        display_frame = (
            frame.copy()
        )

        # ====================================================
        # ALWAYS DETECT LANE
        # ====================================================

        display_frame, offset, lane_is_real = detect_lane(
            display_frame
        )

        now = time.time()

        # ====================================================
        # AUTO
        # ====================================================

        if current_mode == MODE_AUTO:

            # ================================================
            # HAVE OFFSET
            # ================================================

            if offset is not None:

                # 只有真的有效車道
                # 才更新 last_real_lane_time
                if lane_is_real:

                    last_real_lane_time = now
                    lane_stop_sent = False

                lost_time = (
                    now -
                    last_real_lane_time
                )

                # HOLD 太久仍然要停
                if (
                    not lane_is_real
                    and
                    lost_time >= LOST_STOP_TIME
                ):

                    if not lane_stop_sent:

                        send_stop()

                        lane_stop_sent = True

                        last_auto_steering = None
                        last_auto_speed = None

                    status_text = (
                        "LANE LOST - STOP"
                    )

                    cv2.putText(
                        display_frame,
                        status_text,
                        (
                            10,
                            70
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (
                            0,
                            0,
                            255
                        ),
                        2
                    )

                else:

                    steering, raw, rate = calculate_steering(
                        offset
                    )

                    speed = calculate_auto_speed(
                        steering,
                        lane_is_real
                    )

                    smart_send_auto(
                        speed,
                        steering
                    )

                    if (
                        now -
                        last_lane_print
                        >=
                        TERMINAL_PRINT_INTERVAL
                    ):

                        print(
                            "[LANE] OFFSET={} RAW={} STEER={} RATE={:.1f} SPEED={:.2f} REAL={}".format(
                                offset,
                                raw,
                                steering,
                                rate,
                                speed,
                                lane_is_real
                            )
                        )

                        last_lane_print = now

                    cv2.putText(
                        display_frame,
                        "AUTO SPD:{:.2f} STEER:{}".format(
                            speed,
                            steering
                        ),
                        (
                            10,
                            70
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (
                            0,
                            255,
                            0
                        ),
                        2
                    )

            # ================================================
            # NO OFFSET
            # ================================================

            else:

                lost_time = (
                    now -
                    last_real_lane_time
                )

                if (
                    lost_time
                    <
                    LOST_STAGE1_TIME
                ):

                    smart_send_auto(
                        LOST_SPEED_1,
                        last_valid_steering
                    )

                    text = (
                        "TEMP LOST {:.2f}s"
                        .format(
                            lost_time
                        )
                    )

                elif (
                    lost_time
                    <
                    LOST_STOP_TIME
                ):

                    smart_send_auto(
                        LOST_SPEED_2,
                        last_valid_steering
                    )

                    text = (
                        "LOST SLOW {:.2f}s"
                        .format(
                            lost_time
                        )
                    )

                else:

                    if not lane_stop_sent:

                        send_stop()

                        lane_stop_sent = True

                        last_auto_steering = None
                        last_auto_speed = None

                        previous_offset = None
                        filtered_offset_rate = 0.0
                        last_control_time = None

                        print(
                            "[LANE] LOST | COMMAND=STOP"
                        )

                    text = (
                        "LANE LOST - STOP"
                    )

                cv2.putText(
                    display_frame,
                    text,
                    (
                        10,
                        70
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (
                        0,
                        0,
                        255
                    ),
                    2
                )

        # ====================================================
        # MANUAL
        # ====================================================

        else:

            if offset is None:

                text = (
                    "MANUAL SPD:{:.2f} NO LANE"
                    .format(
                        MANUAL_DRIVE_SPEED
                    )
                )

            else:

                text = (
                    "MANUAL SPD:{:.2f} OFFSET:{}"
                    .format(
                        MANUAL_DRIVE_SPEED,
                        offset
                    )
                )

            cv2.putText(
                display_frame,
                text,
                (
                    10,
                    70
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (
                    0,
                    255,
                    255
                ),
                2
            )

        # ====================================================
        # FRAME
        # ====================================================

        with frame_lock:

            latest_frame = (
                display_frame.copy()
            )

        cv2.imshow(
            "Lane Detection",
            display_frame
        )

        key = (
            cv2.waitKey(1)
            &
            0xFF
        )

        if key == ord("q"):

            stop_event.set()

            break

    try:

        if cap is not None:
            cap.release()

    except Exception:
        pass

    try:

        cv2.destroyAllWindows()

    except Exception:
        pass


# ============================================================
# VIDEO
# ============================================================

def generate_frames():

    while not stop_event.is_set():

        with frame_lock:

            frame = (
                None
                if latest_frame is None
                else latest_frame.copy()
            )

        if frame is None:

            time.sleep(
                0.02
            )

            continue

        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                70
            ]
        )

        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            +
            buffer.tobytes()
            +
            b"\r\n"
        )


# ============================================================
# WEB
# ============================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Jetson Car</title>

<style>
body {
    background:#111;
    color:white;
    text-align:center;
    font-family:Arial;
    margin:0;
    padding:10px;
    user-select:none;
}

img {
    width:100%;
    max-width:640px;
    border:2px solid #555;
    border-radius:8px;
}

button {
    width:90px;
    height:70px;
    margin:8px;
    border:none;
    border-radius:15px;
    touch-action:none;
}

.mode {
    width:130px;
    height:55px;
    font-size:18px;
}

.auto { background:#ffaa33; }
.manual { background:#66aaff; }
.forward { background:#44cc66; }
.backward { background:#dd5555; }
.left,.right { background:#4488dd; }
.stop { background:white; }
</style>
</head>

<body>

<img src="/video_feed">

<br>

<button class="mode auto" onclick="setMode('AUTO')">
AUTO
</button>

<button class="mode manual" onclick="setMode('MANUAL')">
MANUAL
</button>

<br><br>

<button class="forward"
onpointerdown="startCommand('F')"
onpointerup="stopCommand()"
onpointerleave="stopCommand()"
onpointercancel="stopCommand()">
</button>

<br>

<button class="left"
onpointerdown="startCommand('L')"
onpointerup="stopCommand()"
onpointerleave="stopCommand()"
onpointercancel="stopCommand()">
</button>

<button class="stop"
onclick="stopCommand()">
</button>

<button class="right"
onpointerdown="startCommand('R')"
onpointerup="stopCommand()"
onpointerleave="stopCommand()"
onpointercancel="stopCommand()">
</button>

<br>

<button class="backward"
onpointerdown="startCommand('B')"
onpointerup="stopCommand()"
onpointerleave="stopCommand()"
onpointercancel="stopCommand()">
</button>

<script>

let currentCommand = "STOP";

function sendCommand(cmd)
{
    fetch("/control", {
        method:"POST",
        headers:{
            "Content-Type":"application/json"
        },
        body:JSON.stringify({
            command:cmd
        })
    });
}

function startCommand(cmd)
{
    if(currentCommand === cmd)
        return;

    currentCommand = cmd;

    sendCommand(cmd);
}

function stopCommand()
{
    if(currentCommand === "STOP")
        return;

    currentCommand = "STOP";

    sendCommand("STOP");
}

function setMode(mode)
{
    stopCommand();

    fetch("/mode", {
        method:"POST",
        headers:{
            "Content-Type":"application/json"
        },
        body:JSON.stringify({
            mode:mode
        })
    });
}

window.addEventListener(
    "blur",
    stopCommand
);

document.addEventListener(
    "visibilitychange",
    function()
    {
        if(document.hidden)
            stopCommand();
    }
);

</script>
</body>
</html>
"""


@app.route("/")
def index():

    return HTML


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        )
    )


@app.route(
    "/mode",
    methods=["POST"]
)
def set_mode():

    global current_mode

    global manual_command
    global manual_speed
    global manual_steering

    global last_auto_steering
    global last_auto_speed
    global last_auto_send_time

    global smoothed_steering
    global last_valid_steering

    global last_lane_center_filtered

    global last_real_lane_time
    global lane_stop_sent

    global previous_offset
    global filtered_offset_rate
    global last_control_time

    data = (
        request.get_json(
            silent=True
        )
        or
        {}
    )

    mode = (
        data.get(
            "mode",
            ""
        )
        .upper()
    )

    if mode not in [
        MODE_AUTO,
        MODE_MANUAL
    ]:

        return jsonify(
            {
                "ok": False
            }
        )

    send_stop()

    current_mode = mode

    with manual_lock:

        manual_command = "STOP"
        manual_speed = MANUAL_DRIVE_SPEED
        manual_steering = 0

    last_auto_steering = None
    last_auto_speed = None
    last_auto_send_time = 0.0

    smoothed_steering = 0.0
    last_valid_steering = 0

    last_lane_center_filtered = None

    previous_offset = None
    filtered_offset_rate = 0.0
    last_control_time = None

    last_real_lane_time = time.time()

    lane_stop_sent = False

    print(
        "[MODE]",
        current_mode
    )

    return jsonify(
        {
            "ok": True,
            "mode": current_mode
        }
    )


@app.route(
    "/control",
    methods=["POST"]
)
def control():

    global manual_command
    global manual_speed
    global manual_steering

    if (
        current_mode
        !=
        MODE_MANUAL
    ):

        return jsonify(
            {
                "ok": False
            }
        )

    data = (
        request.get_json(
            silent=True
        )
        or
        {}
    )

    command = (
        data.get(
            "command",
            "STOP"
        )
        .upper()
    )

    speed = (
        MANUAL_DRIVE_SPEED
    )

    if command == "F":

        direction = "F"
        steering = 0

    elif command == "B":

        direction = "B"
        steering = 0

    elif command == "L":

        direction = "F"
        steering = 35

    elif command == "R":

        direction = "F"
        steering = -35

    elif command == "STOP":

        with manual_lock:

            manual_command = "STOP"
            manual_steering = 0

        send_stop()

        print(
            "[MANUAL] STOP"
        )

        return jsonify(
            {
                "ok": True
            }
        )

    else:

        return jsonify(
            {
                "ok": False
            }
        )

    with manual_lock:

        manual_command = direction
        manual_speed = speed
        manual_steering = steering

    send_manual(
        direction,
        speed,
        steering
    )

    return jsonify(
        {
            "ok": True
        }
    )


def flask_worker():

    print(
        "[WEB] http://0.0.0.0:5000"
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )


# ============================================================
# CLEANUP
# ============================================================

def cleanup():

    global ser
    global cap

    print(
        "[SYSTEM] Stopping..."
    )

    try:
        send_stop()

    except Exception:
        pass

    stop_event.set()

    try:

        if cap is not None:
            cap.release()

    except Exception:
        pass

    with ser_lock:

        try:

            if ser is not None:
                ser.close()

        except Exception:
            pass

        ser = None

    try:
        cv2.destroyAllWindows()

    except Exception:
        pass

    print(
        "[SYSTEM] Shutdown complete"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("============================================")
    print("Jetson Lane Control - Stable PD Version")
    print("============================================")
    print("AUTO MAX SPEED =", AUTO_MAX_SPEED)
    print("KP             =", STEERING_KP)
    print("KD             =", STEERING_KD)
    print("MAX STEERING   =", MAX_STEERING)
    print("STEER STEP     =", MAX_STEER_STEP)
    print("TARGET Y       =", TARGET_Y_RATIO)
    print("============================================")
    print()

    serial_thread = threading.Thread(
        target=serial_reader,
        daemon=True
    )

    manual_thread = threading.Thread(
        target=manual_worker,
        daemon=True
    )

    flask_thread = threading.Thread(
        target=flask_worker,
        daemon=True
    )

    serial_thread.start()
    manual_thread.start()
    flask_thread.start()

    try:

        camera_loop()

    except KeyboardInterrupt:

        print(
            "[SYSTEM] Ctrl+C"
        )

    finally:

        cleanup()