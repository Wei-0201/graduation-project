# -*- coding: utf-8 -*-
"""User-supplied sign colour/shape/OCR functions, without standalone camera loop.
Only import change: local Tesseract CLI adapter replaces pytesseract.
"""
import cv2
import numpy as np
import re
import sign_ocr_engine as pytesseract

CAMERA_ID = 0

FRAME_WIDTH = 640

FRAME_HEIGHT = 480

TARGET_FPS = 30

DETECT_INTERVAL = 2

OCR_INTERVAL = 2

MIN_RED_AREA = 1000

MIN_BLUE_AREA = 1000

MIN_SIGN_WIDTH = 40

MIN_SIGN_HEIGHT = 40

MAX_FRAME_AREA_RATIO = 0.40

STOP_MIN_VERTICES = 7

STOP_MAX_VERTICES = 10

STOP_MIN_ASPECT = 0.68

STOP_MAX_ASPECT = 1.40

STOP_MIN_FILL = 0.52

STOP_MIN_SOLIDITY = 0.84

PARK_MIN_VERTICES = 4

PARK_MAX_VERTICES = 8

PARK_MIN_ASPECT = 0.65

PARK_MAX_ASPECT = 1.40

PARK_MIN_BLUE_FILL = 0.30

PARK_MIN_SOLIDITY = 0.78

P_INNER_MARGIN = 0.16

P_MIN_WHITE_RATIO = 0.06

P_MAX_WHITE_RATIO = 0.55

P_MIN_HEIGHT_RATIO = 0.45

P_MAX_HEIGHT_RATIO = 0.98

P_MIN_WIDTH_RATIO = 0.20

P_MAX_WIDTH_RATIO = 0.85

P_LEFT_FILL_MIN = 0.28

P_BOTTOM_RIGHT_MAX = 0.40

STOP_CONFIRM = 2

PARK_CONFIRM = 2

RESULT_HOLD_TIME = 1.0

LOWER_RED_1 = np.array([0, 90, 60])

UPPER_RED_1 = np.array([12, 255, 255])

LOWER_RED_2 = np.array([165, 90, 60])

UPPER_RED_2 = np.array([179, 255, 255])

LOWER_BLUE = np.array([90, 70, 40])

UPPER_BLUE = np.array([140, 255, 255])

LOWER_WHITE = np.array([0, 0, 145])

UPPER_WHITE = np.array([179, 115, 255])

def clean_text(text):

    text = text.upper()

    text = re.sub(
        r'[^A-Z]',
        '',
        text
    )

    return text

def get_contours(mask):

    result = cv2.findContours(
        mask.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(result) == 3:
        _, contours, _ = result
    else:
        contours, _ = result

    return contours

def get_solidity(contour):

    area = cv2.contourArea(contour)

    hull = cv2.convexHull(contour)

    hull_area = cv2.contourArea(hull)

    if hull_area <= 0:
        return 0.0

    return float(area) / float(hull_area)

def expand_box(
    x,
    y,
    w,
    h,
    image_width,
    image_height,
    ratio=0.05
):

    px = int(w * ratio)
    py = int(h * ratio)

    x1 = max(0, x - px)
    y1 = max(0, y - py)

    x2 = min(
        image_width,
        x + w + px
    )

    y2 = min(
        image_height,
        y + h + py
    )

    return x1, y1, x2, y2

def find_stop_candidate(red_mask):

    contours = get_contours(red_mask)

    best = None
    best_score = -999

    frame_area = (
        FRAME_WIDTH *
        FRAME_HEIGHT
    )

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_RED_AREA:
            continue

        if area > (
            frame_area *
            MAX_FRAME_AREA_RATIO
        ):
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            0.025 * perimeter,
            True
        )

        vertices = len(approx)

        if vertices < STOP_MIN_VERTICES:
            continue

        if vertices > STOP_MAX_VERTICES:
            continue

        if not cv2.isContourConvex(approx):
            continue

        x, y, w, h = cv2.boundingRect(
            approx
        )

        if w < MIN_SIGN_WIDTH:
            continue

        if h < MIN_SIGN_HEIGHT:
            continue

        if h <= 0:
            continue

        aspect = float(w) / float(h)

        if aspect < STOP_MIN_ASPECT:
            continue

        if aspect > STOP_MAX_ASPECT:
            continue

        rect_area = float(w * h)

        if rect_area <= 0:
            continue

        fill_ratio = area / rect_area

        if fill_ratio < STOP_MIN_FILL:
            continue

        solidity = get_solidity(
            contour
        )

        if solidity < STOP_MIN_SOLIDITY:
            continue

        vertex_score = (
            1.0 -
            abs(vertices - 8) * 0.15
        )

        aspect_score = (
            1.0 -
            abs(aspect - 1.0)
        )

        fill_score = (
            1.0 -
            abs(fill_ratio - 0.80)
        )

        score = (
            vertex_score +
            aspect_score +
            fill_score +
            solidity
        )

        if score > best_score:

            best_score = score

            best = {
                "x": x,
                "y": y,
                "w": w,
                "h": h,

                "vertices": vertices,
                "fill": fill_ratio,
                "solidity": solidity,

                "approx": approx
            }

    return best

def find_blue_square_candidate(blue_mask):

    contours = get_contours(
        blue_mask
    )

    best = None
    best_score = -999

    frame_area = (
        FRAME_WIDTH *
        FRAME_HEIGHT
    )

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_BLUE_AREA:
            continue

        if area > (
            frame_area *
            MAX_FRAME_AREA_RATIO
        ):
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            0.03 * perimeter,
            True
        )

        vertices = len(approx)

        if vertices < PARK_MIN_VERTICES:
            continue

        if vertices > PARK_MAX_VERTICES:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if w < MIN_SIGN_WIDTH:
            continue

        if h < MIN_SIGN_HEIGHT:
            continue

        if h <= 0:
            continue

        aspect = float(w) / float(h)

        if aspect < PARK_MIN_ASPECT:
            continue

        if aspect > PARK_MAX_ASPECT:
            continue

        rect_area = float(w * h)

        if rect_area <= 0:
            continue

        blue_fill = area / rect_area

        if blue_fill < PARK_MIN_BLUE_FILL:
            continue

        solidity = get_solidity(
            contour
        )

        if solidity < PARK_MIN_SOLIDITY:
            continue

        aspect_score = (
            1.0 -
            abs(aspect - 1.0)
        )

        score = (
            aspect_score +
            blue_fill +
            solidity
        )

        if score > best_score:

            best_score = score

            best = {
                "x": x,
                "y": y,
                "w": w,
                "h": h,

                "vertices": vertices,
                "fill": blue_fill,
                "solidity": solidity,

                "approx": approx
            }

    return best

def check_white_p_shape(roi):

    """
    Parking sign:

    BLUE background
    WHITE P

    這裡只檢查中央區域，
    故意排除外圍白框。
    """

    if roi is None:
        return False, None

    if roi.size == 0:
        return False, None

    # --------------------------------------
    # Resize
    # --------------------------------------

    big = cv2.resize(
        roi,
        None,
        fx=3.0,
        fy=3.0,
        interpolation=cv2.INTER_CUBIC
    )

    h, w = big.shape[:2]

    # --------------------------------------
    # Cut border
    # --------------------------------------

    margin_x = int(
        w * P_INNER_MARGIN
    )

    margin_y = int(
        h * P_INNER_MARGIN
    )

    if (
        margin_x * 2 >= w
        or
        margin_y * 2 >= h
    ):
        return False, None

    inner = big[
        margin_y:h-margin_y,
        margin_x:w-margin_x
    ]

    if inner.size == 0:
        return False, None

    # --------------------------------------
    # HSV
    # --------------------------------------

    hsv = cv2.cvtColor(
        inner,
        cv2.COLOR_BGR2HSV
    )

    # --------------------------------------
    # WHITE MASK
    # --------------------------------------

    white_mask = cv2.inRange(
        hsv,
        LOWER_WHITE,
        UPPER_WHITE
    )

    # --------------------------------------
    # Clean
    # --------------------------------------

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # --------------------------------------
    # White ratio
    # --------------------------------------

    total_pixels = white_mask.size

    if total_pixels <= 0:
        return False, white_mask

    white_pixels = cv2.countNonZero(
        white_mask
    )

    white_ratio = (
        float(white_pixels) /
        float(total_pixels)
    )

    if white_ratio < P_MIN_WHITE_RATIO:
        return False, white_mask

    if white_ratio > P_MAX_WHITE_RATIO:
        return False, white_mask

    # --------------------------------------
    # Find white objects
    # --------------------------------------

    contours = get_contours(
        white_mask
    )

    if len(contours) == 0:
        return False, white_mask

    # --------------------------------------
    # 找最大的白色物體
    # --------------------------------------

    largest = max(
        contours,
        key=cv2.contourArea
    )

    letter_area = cv2.contourArea(
        largest
    )

    if letter_area <= 0:
        return False, white_mask

    lx, ly, lw, lh = cv2.boundingRect(
        largest
    )

    ih, iw = white_mask.shape[:2]

    # --------------------------------------
    # Size ratio
    # --------------------------------------

    height_ratio = (
        float(lh) /
        float(ih)
    )

    width_ratio = (
        float(lw) /
        float(iw)
    )

    if height_ratio < P_MIN_HEIGHT_RATIO:
        return False, white_mask

    if height_ratio > P_MAX_HEIGHT_RATIO:
        return False, white_mask

    if width_ratio < P_MIN_WIDTH_RATIO:
        return False, white_mask

    if width_ratio > P_MAX_WIDTH_RATIO:
        return False, white_mask

    # --------------------------------------
    # Extract P
    # --------------------------------------

    letter = white_mask[
        ly:ly+lh,
        lx:lx+lw
    ]

    if letter.size == 0:
        return False, white_mask

    letter_h, letter_w = (
        letter.shape[:2]
    )

    # ========================================================
    # FEATURE 1
    #
    # P 左側應該有一條長的白色直線
    # ========================================================

    left_width = max(
        1,
        int(letter_w * 0.35)
    )

    left_region = letter[
        :,
        0:left_width
    ]

    left_pixels = cv2.countNonZero(
        left_region
    )

    left_ratio = (
        float(left_pixels) /
        float(left_region.size)
    )

    if left_ratio < P_LEFT_FILL_MIN:
        return False, white_mask

    # ========================================================
    # FEATURE 2
    #
    # P 上半部應該比下半部有更多白色
    # ========================================================

    half_h = int(
        letter_h * 0.50
    )

    upper = letter[
        0:half_h,
        :
    ]

    lower = letter[
        half_h:letter_h,
        :
    ]

    if (
        upper.size == 0
        or
        lower.size == 0
    ):
        return False, white_mask

    upper_ratio = (
        float(
            cv2.countNonZero(upper)
        )
        /
        float(upper.size)
    )

    lower_ratio = (
        float(
            cv2.countNonZero(lower)
        )
        /
        float(lower.size)
    )

    # P 上半部應該比下半部寬
    if upper_ratio <= lower_ratio:
        return False, white_mask

    # ========================================================
    # FEATURE 3
    #
    # P 右下角應該是空的
    # ========================================================

    br_x = int(
        letter_w * 0.55
    )

    br_y = int(
        letter_h * 0.55
    )

    bottom_right = letter[
        br_y:letter_h,
        br_x:letter_w
    ]

    if bottom_right.size > 0:

        bottom_right_ratio = (
            float(
                cv2.countNonZero(
                    bottom_right
                )
            )
            /
            float(
                bottom_right.size
            )
        )

        if (
            bottom_right_ratio
            >
            P_BOTTOM_RIGHT_MAX
        ):
            return False, white_mask

    # ========================================================
    # FEATURE 4
    #
    # P 上方右側必須有白色
    # ========================================================

    tr_x = int(
        letter_w * 0.45
    )

    tr_y2 = int(
        letter_h * 0.50
    )

    top_right = letter[
        0:tr_y2,
        tr_x:letter_w
    ]

    if top_right.size == 0:
        return False, white_mask

    top_right_ratio = (
        float(
            cv2.countNonZero(
                top_right
            )
        )
        /
        float(
            top_right.size
        )
    )

    if top_right_ratio < 0.15:
        return False, white_mask

    # ========================================================
    # SUCCESS
    # ========================================================

    # Debug image
    debug = cv2.cvtColor(
        white_mask,
        cv2.COLOR_GRAY2BGR
    )

    cv2.rectangle(
        debug,
        (lx, ly),
        (
            lx + lw,
            ly + lh
        ),
        (0, 255, 0),
        2
    )

    return True, debug

def run_stop_ocr(roi):

    if roi is None:
        return "", None

    if roi.size == 0:
        return "", None

    big = cv2.resize(
        roi,
        None,
        fx=3.0,
        fy=3.0,
        interpolation=cv2.INTER_CUBIC
    )

    hsv = cv2.cvtColor(
        big,
        cv2.COLOR_BGR2HSV
    )

    # STOP 白色文字
    white = cv2.inRange(
        hsv,
        LOWER_WHITE,
        UPPER_WHITE
    )

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    white = cv2.morphologyEx(
        white,
        cv2.MORPH_CLOSE,
        kernel
    )

    config = (
        '--psm 7 '
        '-c tessedit_char_whitelist=STOP'
    )

    try:

        text = pytesseract.image_to_string(
            white,
            config=config
        )

    except Exception as e:

        print(
            "STOP OCR ERROR:",
            e
        )

        return "", white

    text = clean_text(text)

    return text, white

def run_p_ocr(roi):

    if roi is None:
        return "", None

    if roi.size == 0:
        return "", None

    # --------------------------------------
    # Resize
    # --------------------------------------

    big = cv2.resize(
        roi,
        None,
        fx=4.0,
        fy=4.0,
        interpolation=cv2.INTER_CUBIC
    )

    h, w = big.shape[:2]

    # --------------------------------------
    # REMOVE OUTER BORDER
    # --------------------------------------

    margin_x = int(
        w * P_INNER_MARGIN
    )

    margin_y = int(
        h * P_INNER_MARGIN
    )

    if (
        margin_x * 2 >= w
        or
        margin_y * 2 >= h
    ):
        return "", None

    inner = big[
        margin_y:h-margin_y,
        margin_x:w-margin_x
    ]

    if inner.size == 0:
        return "", None

    # --------------------------------------
    # HSV
    # --------------------------------------

    hsv = cv2.cvtColor(
        inner,
        cv2.COLOR_BGR2HSV
    )

    # --------------------------------------
    # WHITE P
    # --------------------------------------

    white = cv2.inRange(
        hsv,
        LOWER_WHITE,
        UPPER_WHITE
    )

    # --------------------------------------
    # CLEAN
    # --------------------------------------

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    white = cv2.morphologyEx(
        white,
        cv2.MORPH_OPEN,
        kernel
    )

    white = cv2.morphologyEx(
        white,
        cv2.MORPH_CLOSE,
        kernel
    )

    # --------------------------------------
    # 找最大白色物體
    #
    # 不把其他白色雜訊一起丟給 OCR
    # --------------------------------------

    contours = get_contours(
        white
    )

    if len(contours) == 0:
        return "", white

    largest = max(
        contours,
        key=cv2.contourArea
    )

    x, y, ww, hh = cv2.boundingRect(
        largest
    )

    if ww <= 0 or hh <= 0:
        return "", white

    pad = 10

    x1 = max(
        0,
        x - pad
    )

    y1 = max(
        0,
        y - pad
    )

    x2 = min(
        white.shape[1],
        x + ww + pad
    )

    y2 = min(
        white.shape[0],
        y + hh + pad
    )

    letter = white[
        y1:y2,
        x1:x2
    ]

    if letter.size == 0:
        return "", white

    # --------------------------------------
    # Tesseract 喜歡黑字白底
    #
    # 現在 mask 是：
    # P = 白
    # 背景 = 黑
    #
    # 所以反相
    # --------------------------------------

    ocr_img = cv2.bitwise_not(
        letter
    )

    # --------------------------------------
    # Add white border
    # --------------------------------------

    ocr_img = cv2.copyMakeBorder(
        ocr_img,
        30,
        30,
        30,
        30,
        cv2.BORDER_CONSTANT,
        value=255
    )

    # --------------------------------------
    # OCR
    # --------------------------------------

    config = (
        '--psm 10 '
        '-c tessedit_char_whitelist=P'
    )

    try:

        text = pytesseract.image_to_string(
            ocr_img,
            config=config
        )

    except Exception as e:

        print(
            "P OCR ERROR:",
            e
        )

        return "", ocr_img

    text = clean_text(text)

    return text, ocr_img
