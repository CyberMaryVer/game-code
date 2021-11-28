import os
import cv2.cv2 as cv2
# from PIL import Image
from time import time
import imutils
import numpy as np
from datetime import datetime
from mp_predictor import MpipePredictor, get_updated_keypoints
from visualization import visualize_keypoints, draw_box_with_text2
from geometry import get_distance

KEYPOINTS_FOR_GAME = ["left_wrist", "right_wrist", "nose"]
WITH_SKELETON = False
GAMES_NUMBER = 6
WINDOW_NAME = "Destroy the danger!"
SCALE = 1.2
IMAGE_SIZE = (275, 200)
SAVE_OUTPUT = True
SAVE_PATH = "demo.mp4"


def draw_nose(img, keypoints):
    if keypoints is None:
        return img
    if len(keypoints) == 17:
        keypoints = get_updated_keypoints(keypoints)

    center = (keypoints["nose"][0], keypoints["nose"][1])
    hcenter = (keypoints["head_center"][0], keypoints["head_center"][1])
    radius = int(get_distance(center, hcenter) / 2)
    color = (0, 0, 255)
    img = cv2.circle(img=img,
                     center=center,
                     radius=radius,
                     color=color,
                     thickness=-1,
                     lineType=cv2.LINE_AA)
    return img


def find_images(size=IMAGE_SIZE):
    safe_ims = [os.path.join("./images", p) for p in os.listdir("./images") if "safe" in p]
    danger_ims = [os.path.join("./images", p) for p in os.listdir("./images") if "danger" in p]
    safe = cv2.imread(np.random.choice(safe_ims))
    danger = cv2.imread(np.random.choice(danger_ims))
    safe = cv2.resize(safe, size)
    danger = cv2.resize(danger, size)

    return safe, danger


def get_cat_path():
    paths = [os.path.join("./cat", p) for p in os.listdir("./cat")]
    # return np.random.choice(paths)
    return paths[4]


def draw_box_with_text(img, text=None, edge_color=(255, 255, 255), border=2, mode=0):
    """
    draws box around
    """
    # width, height = img.shape[1::-1]
    # scale = max(width, height) / 400
    font_scale, font_thickness = .8, 2
    font_color = (0, 0, 0)

    if mode == 0:  # standard mode
        img = cv2.copyMakeBorder(img, 10 * border, border, border, border, cv2.BORDER_CONSTANT, value=edge_color)

    elif mode == 1:  # low vision
        img = cv2.copyMakeBorder(img, 10 * border, border, border, border, cv2.BORDER_CONSTANT, value=edge_color)
        font_scale, font_thickness = 1.6, 2

    if text is not None:
        x = y = border
        img = cv2.putText(img, text, (x + 2, y + 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness,
                          lineType=cv2.LINE_AA)

    img = cv2.copyMakeBorder(img, border, 42 * border, border, border, cv2.BORDER_ISOLATED, value=edge_color)
    cat = ""

    return img


def mp_pose_game(img, keypoints, shape, game_state=0,
                 active_point=None, right_point=None, wrong_point=None,
                 radius=1, color=None, right_answer="", img_r=None, img_w=None):
    w, h = shape
    color_end = (0, 255, 0)
    color_fail = (0, 0, 255)
    color1 = color2 = color
    txt = ""

    if game_state == 0:
        img_r, img_w = find_images()
        active_point = np.random.choice(KEYPOINTS_FOR_GAME)
        # xx, yy = np.random.randint(0, w), np.random.randint(0, h)
        xx1, yy1 = int(w * SCALE * .10), int(h * SCALE * .05)
        xx2, yy2 = int(w * SCALE * .60), int(h * SCALE * .05)
        points = ((xx1, yy1), (xx2, yy2))
        right_idx = np.random.choice([0, 1])
        wrong_idx = 1 - right_idx
        right_point = points[right_idx]
        wrong_point = points[wrong_idx]
        right_answer = f"right answer is {wrong_idx}"
        txt = f"GAME STARTED\nUSE YOUR {active_point.upper().replace('_', ' ')}!"
        print(txt)

    game_state = 1 if game_state == 0 else game_state
    xx1, yy1 = right_point
    yy1 = yy1 + 1 + np.random.choice([-1, 0, 1])
    xx2, yy2 = wrong_point
    yy2 = yy2 + 1 + np.random.choice([-1, 0, 1])
    xa, ya, _ = keypoints[active_point]
    # radius = int(16) + np.random.choice([-1, 0, 1])
    success_radius = radius
    img = cv2.circle(img, (xa, ya), 8, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)

    if game_state == -1:
        txt = f"PAUSE"
    elif abs(xx1 - xa) < success_radius and abs(yy1 - ya) < success_radius:
        game_state = 2
        txt = f"NO, IT'S NOT DANGEROUS"
        # print(f"DEBUG: {abs(xx1 - xa)}, {abs(yy1 - ya)} - radius {success_radius}")
        color1 = color_fail
    elif abs(xx2 - xa) < success_radius and abs(yy2 - ya) < success_radius:
        game_state = 4
        # print(f"DEBUG: {abs(xx1 - xa)}, {abs(yy1 - ya)} - radius {success_radius}")
        txt = f"WELL DONE!!!!"
        color2 = color_end
    else:
        color1 = color2 = color
        txt = f"SELECT RIGHT ANSWER!!"

    txt = f"{txt} {right_answer.upper()}"

    try:
        img = cv2.rectangle(img, (xx1 - 15, yy1 - 15), (xx1 + 290, yy1 + 215), color1, thickness=-1,
                            lineType=cv2.LINE_AA)
        img = cv2.rectangle(img, (xx2 - 15, yy2 - 15), (xx2 + 290, yy2 + 215), color2, thickness=-1,
                            lineType=cv2.LINE_AA)
        img[yy1:yy1 + 200, xx1:xx1 + 275] = cv2.resize(img_r, (275, 200))
        img[yy2:yy2 + 200, xx2:xx2 + 275] = cv2.resize(img_w, (275, 200))
    except ValueError as e:
        print(e)
        print(f"insert into ({xx1}:{xx1 + 275}, {yy1}:{yy1 + 200}), image shape: {img_r.shape}")

    img = draw_box_with_text(img, txt, edge_color=(255, 255, 255), border=4)

    right_point = (xx1, yy1)
    wrong_point = (xx2, yy2)

    return img, active_point, game_state, right_point, wrong_point, txt, right_answer, img_r, img_w


def main():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    shape = (cam.get(3), cam.get(4))
    scale = max(shape) / 400 * SCALE
    scaled_width = int(shape[0] * SCALE)

    kps = None
    game = 0
    score = 0
    point = None
    cpoint = None
    wpoint = None
    answer = ""
    radius = 185
    success = 0
    fail = 0
    delay = 1
    c1, c2, c3 = 84, 210, 253
    txt = ""
    frame = frame_ = None
    img_r = img_w = None
    predictor = MpipePredictor(detection_thr=.8, tracking_thr=.9)
    video_output = None

    while cv2.waitKey(1) != 27 or success == GAMES_NUMBER:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        frame_ = frame.copy()  # for final photo
        frame = imutils.resize(frame, width=scaled_width)

        if not success == GAMES_NUMBER:
            try:
                outputs = predictor.get_keypoints(frame)
                color = (int(c1), int(c2), int(c3))
                kps = get_updated_keypoints(outputs)
                # radius = int(radius)
                frame = visualize_keypoints(kps, frame, skeleton=2, dict_is_updated=True, threshold=.7, scale=scale)
                frame, point, game, cpoint, wpoint, txt, answer, img_r, img_w = mp_pose_game(img=frame,
                                                                                             keypoints=kps,
                                                                                             shape=shape,
                                                                                             game_state=game,
                                                                                             active_point=point,
                                                                                             right_point=cpoint,
                                                                                             wrong_point=wpoint,
                                                                                             radius=radius,
                                                                                             color=color,
                                                                                             right_answer=answer,
                                                                                             img_r=img_r,
                                                                                             img_w=img_w)
                # radius += np.random.choice([0, 1, 0])
                c1 -= .4
                c3 += .4
                c1, c3 = max(c1, 0), min(c3, 255)

                if fail == GAMES_NUMBER * 42:
                    txt = f"GAME OVER! YOUR SCORE {score}"
                    cam.release()
                    break

                # if game == 1:
                #     if delay < 4:  # ################################# #
                #         print("HOLD!!!!")
                #         delay += 1
                #     else:
                #         delay = 0

                if game == 4:
                    print("RIGHT ANSWER!")
                    success += 1
                    score += 500
                    game = 0
                    delay = 15

                if game == 2:
                    print("WRONG ANSWER!")
                    fail += 1
                    score -= 5
                    # game = 0
                    delay = 15

                if success == GAMES_NUMBER:
                    txt = f"YOU WIN!!! YOUR SCORE {score}"
                    cam.release()
                    break

            except Exception as e:
                print(f"{type(e)}: {e}")
                txt = "Try to stay visible for the camera"
                frame = draw_box_with_text(frame, txt, edge_color=(255, 255, 255), border=6)

            # -------------- cat -------------- #
            if game == 2:
                cat_path = get_cat_path()
                frame = draw_cat(img=frame, cat_path=cat_path, show_text=False)
            # -------------- --- -------------- #

            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(delay)
            delay = 1

            if SAVE_OUTPUT:
                if video_output is None and not frame.shape[0] == 900:  # open output file when 1st frame is received
                    frame_width, frame_height, _ = [int(num) for num in frame.shape]
                    # print(frame_height, frame_width)
                    video_output = cv2.VideoWriter(filename=SAVE_PATH,
                                                   fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=40.,
                                                   frameSize=(frame_height, frame_width), isColor=True, )
                if video_output is not None:
                    frame_ = frame.copy()
                    # print(f"frame shape: {frame.shape}")
                    frame_ = cv2.resize(frame_, (frame_height, frame_width))
                    video_output.write(frame)

    # if SAVE_OUTPUT and video_output is not None:
    #     video_output.release()

    cam.release()
    cv2.waitKey(5)

    if SAVE_OUTPUT and video_output is not None:
        video_output.release()

    if "WIN" in txt:
        frame_color = (242, 242, 242)
    elif "OVER" in txt:
        frame_color = (188, 188, 188)
    else:
        txt = f"YOU EXIT THE GAME. YOU SCORE {score}"
        frame_color = (100, 100, 200)

    print(txt)
    color = (int(c1), int(c2), int(c3))

    final_photo = frame if WITH_SKELETON else frame_  # use clean frame if WITH_SKELETON == False
    # final_overlay = final_photo.copy()
    # final_overlay = cv2.circle(final_overlay, wpoint, radius, color, -1, cv2.LINE_AA)
    # final_photo = cv2.addWeighted(final_photo, .7, final_overlay, .3, 1)
    final_photo = draw_box_with_text(final_photo, txt, edge_color=frame_color, border=6)

    final_photo = improve_photo(final_photo)
    final_photo = draw_cat(img=final_photo)

    cv2.imshow(WINDOW_NAME, final_photo)
    date_and_time = datetime.now().strftime("%d.%m.%Y_%H.%M")
    img_name = f"result_{date_and_time}.jpg"
    img_path = "gallery/" + img_name
    cv2.imwrite(img_path, final_photo)
    cv2.waitKey(0)

    while cv2.waitKey(1) != 27:
        pass
    cv2.destroyAllWindows()


def improve_photo(img):
    img_ = img.copy()
    img_ = cv2.detailEnhance(img_, sigma_s=20, sigma_r=0.15)
    img_ = cv2.edgePreservingFilter(img_, flags=1, sigma_s=60, sigma_r=0.15)
    img_ = cv2.stylization(img_, sigma_s=95, sigma_r=0.95)
    img = cv2.addWeighted(img, .8, img_, .2, 1)

    return img


def replace_alpha(img_path):
    # load image with alpha channel.  use IMREAD_UNCHANGED to ensure loading of alpha channel
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    try:
        # make mask of where the transparent bits are
        trans_mask = image[:, :, 3] == 0
        # replace areas of transparency with white and not transparent
        image[trans_mask] = [255, 255, 255, 255]

    except Exception as e:
        print(f"{type(e)}: {e}")

    # new image without alpha channel...
    new_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return new_img


def draw_cat(img=None, img_path=None, cat_path=None, show_text=True):

    if img is None and img_path is not None:
        img = cv2.imread(img_path)
    elif img is None and img_path is None:
        return
    else:
        print(img.shape)
    img_h, img_w = img.shape[0], img.shape[1]

    cat_h, cat_w = (270, 325) if cat_path is None else (360, 225)
    png = True if cat_path is not None else False
    cat_path = "./images/cat.jpg" if cat_path is None else cat_path

    if png:
        cat = replace_alpha(cat_path)
        cat = cv2.resize(cat, (cat_h, cat_w))
        img[img_h - cat_w:img_h, 100:cat_h + 100] = np.where(cat > 240, img[img_h - cat_w:img_h, 100:cat_h + 100], cat)
    else:
        cat = cv2.imread(cat_path)
        cat = cv2.resize(cat, (cat_h, cat_w))
        img[img_h - cat_w:img_h, img_w - cat_h:] = np.where(cat > 240, img[img_h - cat_w:img_h, img_w - cat_h:], cat)

    if show_text:
        text = cv2.imread("./txt/text1.jpg")
        text = cv2.resize(text, (300, 125))
        img[img_h - 205:img_h-80, 100:400] = np.where(text > 240, img[img_h - 205:img_h-80, 100:400], text)

    # cv2.imshow("", img)
    # cv2.waitKey(0)
    return img


if __name__ == "__main__":
    main()
    # draw_cat(img_path="./gallery/result_27.11.2021_06.58.jpg")
