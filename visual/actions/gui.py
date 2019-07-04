from pathlib import Path as P
from uuid import uuid4
import base64
import itertools
import os
import time

from scipy.spatial.distance import cdist
import cv2
import numpy as np
import pyautogui
import requests

from .base import Action
from exceptions import (
    ConfigurationError,
    FileDoesNotExist,
    NotImplementedStrategy,
    QueryImageNotFound,
    TextNotFound,
)


class MATCH_STRATEGY:
    SIFT = "sift"
    TEMPLATE_MATCHING = "template_matching"
    DEFAULT = "template_matching"


def take_screenshot():
    image = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image


# Macs are at scale in respect to resolution
SCALE = take_screenshot().shape[0] / pyautogui.size()[1]


def nominate_misfit(points):
    """Nominate points which do not fit to the rest because of the distance
    """
    distances = cdist(points, points)
    # pick point which has the highest distance to all the rest points
    index = np.mean(distances.T, axis=0).argmax()
    return index


class ScreenShot(Action):
    instruction_type = "screenshot"
    required = ["name", "path"]
    optional = ["override"]

    def __init__(self, name, path, override=False):
        self.path = path
        self.override = override
        super().__init__(name)

    def run(self):
        image = take_screenshot()
        path = P(self.path)
        if path.exists() and not self.override:
            path = path.dirname / f"{path.stem}-{uuid4()}{path.suffix}"

        cv2.imwrite(str(path), image)


class TypeWriteAction(Action):
    instruction_type = "type_write"
    required = ["name", "text"]
    optional = ["interval"]

    def __init__(self, name, text, interval=0.05):
        self.text = text
        self.interval = interval
        super().__init__(name)

    def run(self):
        pyautogui.typewrite(self.text, interval=self.interval)


class TemplateAction(Action):
    def __init__(self, name, path, strategy):
        if not os.path.exists(path):
            raise FileDoesNotExist(f"File {path} does not exist.")

        self.path = path

        if strategy not in (MATCH_STRATEGY.TEMPLATE_MATCHING, MATCH_STRATEGY.SIFT):
            raise NotImplementedStrategy(f"{strategy} does not exist.")

        self.strategy = strategy
        super().__init__(name)

    def find_image(self, *args, **kwargs):
        if self.strategy == MATCH_STRATEGY.SIFT:
            return self._find_image_SIFT(*args, **kwargs)

        elif self.strategy == MATCH_STRATEGY.TEMPLATE_MATCHING:
            return self._find_image_template_matching(*args, **kwargs)

        else:
            raise NotImplementedError(
                f'There strategy "{self.strategy}" is not implemented.'
            )

    @staticmethod
    def _find_image_template_matching(query_image, image, threshold=0.98, show=False):
        template_high, template_width = query_image.shape[:2]
        res = cv2.matchTemplate(image, query_image, cv2.TM_CCORR_NORMED)

        (_, max_val, _, max_loc) = cv2.minMaxLoc(res)
        if max_val < threshold:
            raise QueryImageNotFound("Query image was not found in the image.")

        start_x, start_y = int(max_loc[0]), int(max_loc[1])
        end_x, end_y = int(max_loc[0] + template_width), int(max_loc[1] + template_high)

        if show:
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
            image[start_y:end_y, start_x:end_x] = query_image

            winname = "Debug Image Template Matching"
            cv2.imshow(
                winname,
                cv2.resize(image, (0, 0), fx=1 / (SCALE * 1.1), fy=1 / (SCALE * 1.1)),
            )
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # centroid
        return (start_x + end_x) / 2, (start_y + end_y) / 2

    @staticmethod
    def _find_image_SIFT(query_image, image, show=False):
        FLANN_INDEX_KDTREE = 1
        index_params = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}
        sift = cv2.xfeatures2d.SIFT_create()

        kp1, des1 = sift.detectAndCompute(query_image, None)

        # FLANN parameters
        search_params = {"checks": 50}

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        kp2, des2 = sift.detectAndCompute(image, None)

        try:
            matches = flann.knnMatch(des1, des2, k=2)
        except Exception:
            return image

        matchesMask = [[0, 0] for _ in matches]

        good = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.70 * n.distance:
                matchesMask[i] = [1, 0]
                good.append(m)

        points = [kp2[m.trainIdx].pt for m in good]
        index = nominate_misfit(np.array(points))
        del points[index]

        if show:
            for point in points:
                image = cv2.circle(
                    image, (int(point[0]), int(point[1])), 5, (0, 255, 0), thickness=-1
                )
            cv2.imshow(
                "Debug Image SIFT",
                cv2.resize(image, (0, 0), fx=1 / (SCALE * 1.1), fy=1 / (SCALE * 1.1)),
            )
            cv2.waitKey(0)

        # centroid
        return np.int32(np.mean(np.array(points), axis=0))


class ClickAction(TemplateAction):
    instruction_type = "click"
    required = ["name", "path"]
    optional = ["strategy", "sleep", "show"]

    def __init__(
        self, name, path, strategy=MATCH_STRATEGY.DEFAULT, sleep=0, show=False
    ):
        self.sleep = sleep
        self.show = show
        super().__init__(name, path, strategy)

    def run(self):
        query_image = cv2.imread(self.path)
        try:
            point = self.find_image(query_image, take_screenshot(), show=self.show)
        except QueryImageNotFound:
            raise QueryImageNotFound(f"Image {self.path} was not found on the screen")

        pyautogui.moveTo(point[0] / SCALE, point[1] / SCALE, 1, pyautogui.easeOutQuad)
        time.sleep(self.sleep)
        pyautogui.click()


class WaitableAction(TemplateAction):
    required = ["name", "path"]
    optional = ["check_interval", "timeout"]

    def __init__(
        self,
        name,
        path,
        strategy=MATCH_STRATEGY.DEFAULT,
        check_interval=1,
        timeout=60 * 10,
    ):
        self.check_interval = check_interval
        self.timeout = timeout
        super().__init__(name, path, strategy)


class WaitUntilVisible(WaitableAction):
    instruction_type = "wait_until_visible"

    def run(self):
        query_image = cv2.imread(self.path)

        time_start = time.time()
        for i in itertools.count(1):
            try:
                self.find_image(query_image, take_screenshot())
            except QueryImageNotFound:
                print(
                    f"\t{self.path} not visible, took: {time.time() - time_start:.2f}s"
                )
                return
            else:
                print(f"\ttrying {i}. time, {self.path} is still visible...")

            if time.time() > time_start + self.timeout:
                raise TimeoutError(
                    f"{self.path} is still visible after {self.timeout} timeout"
                )

            time.sleep(self.check_interval)


class WaitForAppear(WaitableAction):
    instruction_type = "wait_for_appear"

    def run(self):
        query_image = cv2.imread(self.path)

        time_start = time.time()
        for i in itertools.count(1):
            try:
                self.find_image(query_image, take_screenshot())
            except QueryImageNotFound:
                print(f"\ttrying {i}. time, {self.path} is still not visible...")

            else:
                print(
                    f"\t{self.path} is visible, took: {time.time() - time_start:.2f}s"
                )
                return

            if time.time() > time_start + self.timeout:
                raise TimeoutError(
                    f"{self.path} is still not visible after {self.timeout} timeout"
                )

            time.sleep(self.check_interval)


class ScrollForAppear(WaitableAction):
    instruction_type = "scroll_for_appear"
    optional = WaitableAction.optional + ["distance"]

    def __init__(self, *args, distance=-10, **kwargs):
        self.distance = distance
        super().__init__(*args, **kwargs)

    def run(self):
        query_image = cv2.imread(self.path)

        time_start = time.time()
        for i in itertools.count(1):
            try:
                self.find_image(query_image, take_screenshot())
            except QueryImageNotFound:
                print(
                    f"\ttrying {i}. time, {self.path} is still not visible, scrolling..."
                )
                pyautogui.scroll(self.distance)

            else:
                print(
                    f"\t{self.path} is visible, took: {time.time() - time_start:.2f}s"
                )
                return

            if time.time() > time_start + self.timeout:
                raise TimeoutError(
                    f"{self.path} is still not visible after {self.timeout} timeout"
                )

            time.sleep(self.check_interval)


class KeySequenceAction(Action):
    instruction_type = "key_sequence"
    required = ["name", "keys"]
    optional = ["sleep"]

    def __init__(self, name, keys, sleep=0):
        self.keys = keys
        self.sleep = sleep
        super().__init__(name)

    def run(self):
        for key in self.keys:
            pyautogui.keyDown(key)

        time.sleep(self.sleep)
        for key in self.keys[::-1]:
            pyautogui.keyUp(key)


def convert_image(image: np.array, image_type: str) -> bytes:
    if image_type == "png":
        convert_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]

    elif image_type == "jpeg":
        convert_params = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            95,
            int(cv2.IMWRITE_JPEG_OPTIMIZE),
            True,
        ]
    else:
        convert_params = []

    _, image_encoded = cv2.imencode("." + image_type, image, convert_params)
    return image_encoded.tobytes()


def encode_image(image):
    image_bytes = convert_image(image, "jpeg")
    return base64.b64encode(image_bytes).decode("utf-8")


session = requests.Session()


class FindTextAction(Action):
    instruction_type = "find_text"
    required = ["name", "text"]
    optional = ["show"]

    def __init__(self, name, text, show=False):
        import warnings

        warnings.warn("Find Text function is currently in Beta")
        self.text = text
        self.show = show
        super().__init__(name)

    @property
    def gcv_endpoint_url(self):
        return os.getenv(
            "AUTOMAMA_GCV_ANNOTATE_ENDPOINT",
            "https://vision.googleapis.com/v1/images:annotate",
        )

    @property
    def google_key(self):
        key = os.getenv("AUTOMAMA_GOOGLE_API_KEY")
        if key is None:
            raise ConfigurationError("AUTOMAMA_GOOGLE_API_KEY env variable is not set.")
        return key

    def run(self):
        image = take_screenshot()
        annotations = self.make_gcv_request(encode_image(image))["textAnnotations"]

        # at 0 index is all text read in the image
        annotations[:] = annotations[1:]

        found_indexes = []
        words = self.text.split()
        for i, annotation in enumerate(annotations):
            for word in words:
                if word == annotation["description"]:
                    found_indexes.append(i)
                    break

        if self.show:
            colors = {fi: (0, 255, 0) for fi in found_indexes}  # green
            for i, annotation in enumerate(annotations):
                corners = annotation["boundingPoly"]["vertices"]

                cv2.rectangle(
                    image,
                    (corners[0]["x"], corners[0]["y"]),
                    (corners[2]["x"], corners[2]["y"]),
                    colors.get(i, (0, 0, 255)),  # red
                    2,
                )
            winname = "Debug Image GCV"
            cv2.imshow(
                winname,
                cv2.resize(image, (0, 0), fx=1 / (SCALE * 1.1), fy=1 / (SCALE * 1.1)),
            )
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        ret = []
        for index in found_indexes:
            corners = annotations[index]["boundingPoly"]["vertices"]
            ret.append(
                ((corners[0]["x"], corners[0]["y"]), (corners[2]["x"], corners[2]["y"]))
            )

        if not ret:
            raise TextNotFound(f'"{self.text} was not found on the image')

        return ret

    def make_gcv_request(self, encoded_image):
        request_json = {
            "requests": [
                {
                    "image": {"content": encoded_image},
                    "features": [{"type": "TEXT_DETECTION"}],
                }
            ]
        }

        response = session.post(
            self.gcv_endpoint_url, json=request_json, params={"key": self.google_key}
        )
        return response.json()["responses"][0]
