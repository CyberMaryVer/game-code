import os
import cv2
# from PIL import Image
from time import time
import imutils
import numpy as np
from datetime import datetime

from src.mp_predictor import MpipePredictor, get_updated_keypoints
from src.visualization import visualize_keypoints
from src.geometry import get_distance
from game_configs.catalog import GameConfigLoader

WITH_SKELETON = False


# noinspection PyUnresolvedReferences
class ActiveGame:
    def __init__(self, config: GameConfigLoader):
        # import constants from config
        self.WINDOW_NAME = config.WINDOW_NAME
        self.DESCRIPTION = config.DESCRIPTION
        self.META_WIN = config.META_WIN
        self.META_FAIL = config.META_FAIL
        self.KEYPOINTS_FOR_GAME = config.KEYPOINTS_FOR_GAME
        self.GAMES_NUMBER = config.GAMES_NUMBER
        self.WINDOW_NAME = config.WINDOW_NAME
        self.SCALE = config.SCALE
        self.IMAGE_SIZE = config.IMAGE_SIZE
        self.IMAGE_PATH = config.IMAGE_PATH
        self.IMAGE_TAGS = config.IMAGE_TAGS
        self.IMAGE_CAT = config.IMAGE_CAT
        self.SAVE_OUTPUT = config.SAVE_OUTPUT
        self.SAVE_PATH = config.SAVE_PATH
        self.SPEED = config.SPEED
        self.ACCELERATION = config.ACCELERATION
        self.PAUSE_MAX = config.PAUSE_MAX
        self.RADIUS = 200 * self.SCALE * self.IMAGE_SIZE[0]
        self.COLOR = (0, 0, 0)

        # stateful parameters
        self.show_cat = False
        self.img_correct, self.img_wrong = self.find_images()
        self.START_TIME = time()
        self.W = None
        self.H = None
        self.kps = None
        self.game_state = 0
        self.score = 0
        self.active_point = None
        self.correct_point = None
        self.wrong_point = None
        self.answer = ""
        self.success = 0
        self.fail = 0
        self.score = 0
        self.pause = 0
        self.color = (84, 210, 253)
        self.txt = ""
        self.frame = None
        self.video_output = None
        self.stickers = ["crown", "dress", "random"]

        # set predictor
        self.predictor = MpipePredictor(detection_thr=.8, tracking_thr=.9)

    @staticmethod
    def test_device(source):
        """Checks if webcam exists and enabled"""
        cam = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if cam is None or not cam.isOpened():
            print(f'Warning: unable to open video source: {source}')
            return False
        cam.release()
        return True

    def draw_point(self, point_name="nose", color=(0, 0, 255)):
        if self.kps is None or len(self.stickers) == 0:
            return self.frame
        kps = get_updated_keypoints(self.kps) if len(self.kps) == 17 else self.kps

        center = (self.kps[point_name][0], self.kps[point_name][1])
        # radius = int(5 * self.SCALE)
        head_center = (kps["head_center"][0], kps["head_center"][1])
        radius = int(get_distance(center, head_center) / 2)

        img = cv2.circle(img=self.frame,
                         center=center,
                         radius=radius,
                         color=color,
                         thickness=-1,
                         lineType=cv2.LINE_AA)
        return img

    def draw_sticker(self, img, point_name="head_center", sticker_name="random", thresh=250):
        w, h = int(100 * self.SCALE), int(75 * self.SCALE)  # width and height of sticker
        if self.kps is None or len(self.stickers) == 0:
            return self.frame
        self.kps = get_updated_keypoints(self.kps) if len(self.kps) == 17 else self.kps
        # img_h, img_w = img.shape[0], img.shape[1]
        # print(self.kps.keys())
        w_center = int(self.kps[point_name][0])
        h_center = int(self.kps[point_name][1])

        # img = cv2.circle(img=img,
        #                  center=(w_center, h_center),
        #                  radius=4,
        #                  color=(255, 0, 0),
        #                  thickness=-1,
        #                  lineType=cv2.LINE_AA)

        back = img[h_center - int(h / 2):h_center + h - int(h / 2), w_center - int(w / 2):w_center + w - int(w / 2)]
        sticker_path = f"./txt/{sticker_name}.png" if sticker_name != "random" else "./txt/random.png"
        sticker = self._replace_alpha(sticker_path)
        sticker = cv2.resize(sticker, (w, h))

        try:
            inserted = np.where(sticker > thresh, back, sticker)
            img[h_center - int(h / 2):h_center + h - int(h / 2),
            w_center - int(w / 2):w_center + w - int(w / 2)] = inserted
        except Exception as e:
            print(f"{type(e)}: {e}")

        return img

    def find_images(self):
        path = self.IMAGE_PATH
        ok_tag, nok_tag = self.IMAGE_TAGS
        nok_ims = [os.path.join(path, p) for p in os.listdir(path) if nok_tag in p]
        ok_ims = [os.path.join(path, p) for p in os.listdir(path) if ok_tag in p]
        safe = cv2.imread(np.random.choice(nok_ims))
        danger = cv2.imread(np.random.choice(ok_ims))
        # safe = cv2.resize(safe, self.IMAGE_SIZE)
        # danger = cv2.resize(danger, self.IMAGE_SIZE)

        return safe, danger

    def get_cat_path(self):
        path = self.IMAGE_CAT
        paths = [os.path.join(path, p) for p in os.listdir(path)]
        # return np.random.choice(paths)
        return paths[4]

    @staticmethod
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
            img = cv2.putText(img, text, (x + 2, y + 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color,
                              font_thickness,
                              lineType=cv2.LINE_AA)

        img = cv2.copyMakeBorder(img, border, 42 * border, border, border, cv2.BORDER_ISOLATED, value=edge_color)

        return img

    def _update_game(self, img):
        width_frame, height_frame, _ = img.shape
        width_proportion, height_proportion = self.IMAGE_SIZE
        color_end = (0, 255, 0)
        color_fail = (0, 0, 255)
        width_img = int(width_frame * width_proportion / self.SCALE)
        height_img = int(width_img * height_proportion)
        # center_w, center_h = int(width_frame/2), int(height_frame/10)
        border = 15
        color1 = color2 = self.color

        if self.game_state == -1:
            # in case of fail
            self.pause += 1
            if self.pause == 1:
                self.img_correct = self.find_images()[1]
            if self.pause > self.PAUSE_MAX:
                self.game_state = 0
                self.pause = 0
                img = self.grayscale(self.frame)
                img = self.draw_box_with_text(img, self.txt, edge_color=(255, 255, 255), border=4)
                return img
            # img[center_h:height_img+center_h, center_w:width_img+center_w] = \
            #     cv2.resize(self.img_correct, (width_img, height_img))
            # img = self.draw_box_with_text(img, self.txt, edge_color=(255, 255, 255), border=4)
            # img = self.draw_box_with_text(img, self.txt, edge_color=(255, 255, 255), border=4)
            # return img

        if self.game_state == 0:
            # get random pair of images
            self.img_correct, self.img_wrong = self.find_images()

            # randomly choose active point
            self.active_point = np.random.choice(self.KEYPOINTS_FOR_GAME)

            # set coordinates for images
            xx1, yy1 = int(width_frame * .10), int(height_frame * .05)
            xx2, yy2 = int(width_frame - 2 * (width_frame * .10)), int(height_frame * .05)

            # randomly choose left/right placement for each image
            points = ((xx1, yy1), (xx2, yy2))
            right_idx = np.random.choice([0, 1])
            wrong_idx = 1 - right_idx
            self.correct_point = points[right_idx]
            self.wrong_point = points[wrong_idx]

            # update answer and inform developer about start of the game
            self.answer = f"right answer is {wrong_idx}"
            txt = f"GAME STARTED\nUSE YOUR {self.active_point.upper().replace('_', ' ')}!"
            print(txt)

        if self.game_state == 6:
            # in case of win
            self.pause += 1
            if self.pause == 1:
                self.show_cat = True
            if self.pause > self.PAUSE_MAX:
                self.game_state = 0
                self.pause = 0
                self.show_cat = False
            img = self.draw_box_with_text(img, self.txt, edge_color=(255, 255, 255), border=4)
            return img

        if self.game_state == 3:
            # in case of win
            self.pause += 1
            if self.pause == 1:
                self.img_wrong = self.find_images()[0]
            if self.pause > self.PAUSE_MAX:
                self.game_state = 0
                self.pause = 0
                img = self.draw_box_with_text(img, self.txt, edge_color=(255, 255, 255), border=4)
                return img

        # update game state
        self.game_state = 1 if self.game_state == 0 else self.game_state

        xx1, yy1 = self.correct_point
        yy1 = yy1 + self.SPEED + np.random.choice([-1, 0, 1])
        xx2, yy2 = self.wrong_point
        yy2 = yy2 + self.SPEED + np.random.choice([-1, 0, 1])
        xa, ya, _ = self.kps[self.active_point]

        if self.game_state == 1:
            # in case of play
            img = cv2.circle(img, (xa, ya), 8, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)

            if abs(xx1 - xa) < self.RADIUS and abs(yy1 - ya) < self.RADIUS:
                self.game_state = 2
                self.txt = f"NO, IT'S NOT DANGEROUS"
                # print(f"DEBUG: {abs(xx1 - xa)}, {abs(yy1 - ya)} - radius {success_radius}")
                color1 = color_fail
            elif abs(xx2 - xa) < self.RADIUS and abs(yy2 - ya) < self.RADIUS:
                self.game_state = 4
                # print(f"DEBUG: {abs(xx1 - xa)}, {abs(yy1 - ya)} - radius {success_radius}")
                self.txt = f"WELL DONE!!!!"
                color2 = color_end
            else:
                color1 = color2 = self.color
                self.txt = f"SELECT RIGHT ANSWER!!"

        txt = f"{self.txt} {self.answer.upper()}"

        # draw two descending images
        try:
            img = cv2.rectangle(img, (xx1 - border, yy1 - border),
                                (xx1 + width_img + border, yy1 + height_img + border),
                                color1, thickness=-1, lineType=cv2.LINE_AA)
            img = cv2.rectangle(img, (xx2 - border, yy2 - border),
                                (xx2 + width_img + border, yy2 + height_img + border),
                                color2, thickness=-1, lineType=cv2.LINE_AA)
            img[yy1:yy1 + height_img, xx1:xx1 + width_img] = cv2.resize(self.img_correct, (width_img, height_img))
            img[yy2:yy2 + height_img, xx2:xx2 + width_img] = cv2.resize(self.img_wrong, (width_img, height_img))
        except ValueError as e:
            self.fail += 100
            self.game_state = -1
            print(e)
            print(f"insert into ({xx1}:{xx1 + width_img}, {yy1}:{yy1 + height_img}), "
                  f"image shape: {self.img_correct.shape}")

        if self.game_state == -1:
            img = self.grayscale(img)

        img = self.draw_box_with_text(img, txt, edge_color=(255, 255, 255), border=4)

        self.correct_point = (xx1, yy1)
        self.wrong_point = (xx2, yy2)

        return img

    def play(self):
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        shape = (cam.get(3), cam.get(4))
        scale = max(shape) / 400 * self.SCALE
        scaled_width = int(shape[0] * self.SCALE)

        delay = 1
        txt = ""
        frame = None

        while cv2.waitKey(1) != 27 or self.success == self.GAMES_NUMBER:
            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)
            frame = imutils.resize(frame, width=scaled_width)
            self.frame = frame.copy()  # for final photo
            # frame = self.draw_point(color=(0, 24, 243))

            if not self.success == self.GAMES_NUMBER:
                try:
                    outputs = self.predictor.get_keypoints(frame)
                    self.kps = get_updated_keypoints(outputs)

                    frame = visualize_keypoints(self.kps, frame, skeleton=2, dict_is_updated=True, threshold=.7,
                                                scale=scale)
                    frame = self._update_game(img=frame)

                    if self.fail > self.GAMES_NUMBER * 42:
                        txt = f"GAME OVER! YOUR SCORE {self.score}"
                        self.game_state = 100
                        cam.release()
                        break

                    if self.game_state == 4:
                        print("RIGHT ANSWER!")
                        self.success += 1
                        self.score += 500
                        self.game_state = self.META_WIN
                        # self.SPEED += self.ACCELERATION
                        delay = 15

                    if self.game_state == 2:
                        print("WRONG ANSWER!")
                        self.fail += 1
                        self.score -= 500
                        self.game_state = self.META_FAIL
                        self.SPEED += self.ACCELERATION
                        delay = 15

                    if self.success == self.GAMES_NUMBER:
                        txt = f"YOU WIN!!! YOUR SCORE {self.score}"
                        self.game_state = 777
                        cam.release()
                        break

                except Exception as e:
                    print(f"{type(e)}: {e}")
                    txt = "Try to stay visible for the camera"
                    frame = self.draw_box_with_text(frame, txt, edge_color=(255, 255, 255), border=6)

                # -------------- cat -------------- #
                if self.show_cat:
                    cat_path = self.get_cat_path()
                    frame = self.draw_cat(img=frame, cat_path=cat_path, show_text=False)
                # -------------- --- -------------- #

                cv2.imshow(self.WINDOW_NAME, frame)
                cv2.waitKey(delay)
                delay = 1

                if self.SAVE_OUTPUT:
                    frame_ = frame.copy()

                    if self.video_output is None:  # open output file when 1st frame is received
                        self.W, self.H, _ = frame.shape
                        self.video_output = cv2.VideoWriter(filename=self.SAVE_PATH,
                                                            fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=40.,
                                                            frameSize=(self.H, self.W), isColor=True, )
                        self.video_output.write(frame_)

                    if self.video_output is not None:
                        frame_ = cv2.resize(frame_, (self.H, self.W))
                        self.video_output.write(frame_)

        cam.release()
        cv2.waitKey(5)

        if self.SAVE_OUTPUT and self.video_output is not None:
            self.video_output.release()

        if self.game_state == 777:
            frame_color = (242, 242, 242)
        elif self.game_state == 100:
            frame_color = (188, 188, 188)
        else:
            txt = f"YOU EXIT THE GAME. YOU SCORE {self.score}"
            frame_color = (100, 100, 200)

        # -------------- final photo -------------- #
        final_photo = self.get_final_photo(img=frame, raw=WITH_SKELETON, txt=txt, color=frame_color)
        cv2.imshow(self.WINDOW_NAME, final_photo)
        cv2.waitKey(0)
        # -------------- ----------- -------------- #

        while cv2.waitKey(1) != 27:
            pass
        cv2.destroyAllWindows()

    def get_final_photo(self, img=None, raw=False, txt=None, color=None):
        desc = self.DESCRIPTION["author"]
        final_photo = img if raw else self.frame  # use clean frame if WITH_SKELETON == False
        # final_photo = self.draw_sticker(final_photo)
        final_photo = self.draw_box_with_text(final_photo, text=txt, edge_color=color, border=6)

        final_photo = self.apply_filter(final_photo)
        final_photo = self.draw_cat(img=final_photo)
        final_photo = cv2.putText(final_photo, desc, (4, 4), cv2.FONT_HERSHEY_SIMPLEX, .4, self.COLOR, 1,
                                  lineType=cv2.LINE_AA)

        date_and_time = datetime.now().strftime("%d.%m.%Y_%H.%M")
        img_name = f"result_{date_and_time}.jpg"
        img_path = "gallery/" + img_name
        cv2.imwrite(img_path, final_photo)

        # print(f"Final photo was saved here: {img_path}")
        return final_photo

    def apply_filter(self, img):
        img_ = img.copy()
        # img_ = cv2.detailEnhance(img_, sigma_s=20, sigma_r=0.15)
        # img_ = cv2.edgePreservingFilter(img_, flags=1, sigma_s=60, sigma_r=0.15)
        img_ = self.increase_brightness(img_)
        img_ = self.extract_hue(img_)
        img_ = cv2.stylization(img_, sigma_s=95, sigma_r=0.95)
        img = cv2.addWeighted(img, .7, img_, .3, 1)

        return img

    @staticmethod
    def extract_hue(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # set the bounds for the red hue
        lower_red = np.array([160, 100, 50])
        upper_red = np.array([180, 255, 255])

        # create a mask using the bounds set
        mask = cv2.inRange(hsv, lower_red, upper_red)
        # create an inverse of the mask
        mask_inv = cv2.bitwise_not(mask)
        # Filter only the red colour from the original image using the mask(foreground)
        res = cv2.bitwise_and(img, img, mask=mask)
        # Filter the regions containing colours other than red from the grayscale image(background)
        background = cv2.bitwise_and(gray, gray, mask=mask_inv)
        # convert the one channelled grayscale background to a three channelled image
        background = np.stack((background,) * 3, axis=-1)
        # add the foreground and the background
        added_img = cv2.add(res, background)
        return added_img

    @staticmethod
    def grayscale(img):
        img_ = img.copy()
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        img_ = cv2.cvtColor(img_, cv2.COLOR_GRAY2RGB)

        # img = cv2.addWeighted(img, .5, img_, .5, 1)
        return img_

    @staticmethod
    def increase_brightness(img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    @staticmethod
    def _replace_alpha(img_path):
        """Loads image with alpha channel and replaces it with white background"""
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        try:
            # make mask of where the transparent bits are
            trans_mask = image[:, :, 3] == 0
            # replace areas of transparency with white and not transparent
            image[trans_mask] = [255, 255, 255, 255]

        except Exception as e:
            print(f"{type(e)}: {e}")

        # new image without alpha channel...
        new_img = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return new_img

    def draw_cat(self, img=None, img_path=None, cat_path=None, show_text=True):

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
            cat = self._replace_alpha(cat_path)
            cat = cv2.resize(cat, (cat_h, cat_w))
            img[img_h - cat_w:img_h, 100:cat_h + 100] = np.where(cat > 240, img[img_h - cat_w:img_h, 100:cat_h + 100],
                                                                 cat)
        else:
            cat = cv2.imread(cat_path)
            cat = cv2.resize(cat, (cat_h, cat_w))
            img[img_h - cat_w:img_h, img_w - cat_h:] = np.where(cat > 240, img[img_h - cat_w:img_h, img_w - cat_h:],
                                                                cat)

        if show_text:
            w, h = int(360 * self.SCALE), int(280 * self.SCALE)  # width and height of sticker
            ws, hs = 100, 10  # width shift and height shift
            text = cv2.imread("./txt/win.jpg") if self.game_state == 777 else cv2.imread("./txt/fail.jpg")
            text = cv2.resize(text, (w, h))
            img[img_h - (h + hs):img_h - hs, ws:ws + w] = np.where(text > 233,
                                                                   img[img_h - (h + hs):img_h - hs, ws:ws + w], text)

        return img


def test_sticker(img, thresh=240, scale=1, sticker_name="random", a: ActiveGame = None):
    w, h = int(200 * scale), int(150 * scale)  # width and height of sticker

    img_h, img_w = img.shape[0], img.shape[1]
    w_center, h_center = int(img_w / 2), int(img_h / 3)

    ws, hs = w_center - int(w / 2), h_center - int(h / 2)  # width shift and height shift
    sticker_path = f"./txt/{sticker_name}.png" if sticker_name != "random" else "./txt/random.png"
    if a is not None:
        sticker = a._replace_alpha(sticker_path)
    else:
        sticker = cv2.imread(sticker_path)
    sticker = cv2.resize(sticker, (w, h))
    # sticker = cv2.cvtColor(sticker, cv2.COLOR_RGB2BGR)

    try:
        print(f"{img_h - (h + hs)}:{img_h - hs}, {ws}:{ws + w}")
        inserted = np.where(sticker > thresh, img[img_h - (h + hs):img_h - hs, ws:ws + w], sticker)
        print(f"inserted: {inserted.shape}")
        h = min(h, inserted.shape[0])
        print(f"h={h}")
        img[img_h - (h + hs):img_h - hs, ws:ws + w] = inserted
    except Exception as e:
        print(f"{type(e)}: {e}")

    return img


def test_opencv(test_brightness=True, test_add_sticker=False, test_hue=True):
    img = cv2.imread("./gallery/result_30.11.2021_19.05.jpg", cv2.IMREAD_UNCHANGED)
    a = ActiveGame(config=GameConfigLoader.princess())

    if test_brightness:
        img = a.increase_brightness(img)
        cv2.imshow("", img)
        cv2.waitKey(0)
    if test_add_sticker:
        img = test_sticker(img, thresh=245, a=a, sticker_name="random")
        print(img.shape)
        cv2.imshow("", img)
        cv2.waitKey(0)
    if test_hue:
        img = a.extract_hue(img)
        cv2.imshow("", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    game_config = GameConfigLoader.princess()
    game = ActiveGame(config=game_config)
    game.play()
    # test_opencv()
