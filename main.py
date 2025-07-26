import time
import glob
import logging
import numpy as np
import cv2
import pynput
from PIL import ImageGrab

# Configurable Parameters
IMAGEPATH = "images"
DELAY = 0.1 
CONFIDENCE = 0.9
TOGGLE_KEY = pynput.keyboard.Key.f2  # Don't Change

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class MonopolyBot:
    def __init__(self) -> None:
        self.images_path = IMAGEPATH
        self.cache = {}
        self.running = True

        # Initialize keyboard listener
        self.keyboard_listener = pynput.keyboard.Listener(on_press=self.on_key_press)
        self.keyboard_listener.start()

        self.print_banner()

        while True:
            if self.running:
                self.process_images()
            time.sleep(DELAY)

    def print_banner(self) -> None:
        print("Monopoly Go! Bot")
        print(f"\nPress {TOGGLE_KEY} to toggle running.\n")

    def on_key_press(self, key) -> None:
        if key == TOGGLE_KEY:
            self.toggle_running()

    def toggle_running(self) -> None:
        self.running = not self.running
        status = "Started" if self.running else "Stopped"
        logger.info(f"Bot {status.lower()}.")

    def process_images(self) -> None:
        for path in self.get_sorted_images():
            if not self.running:
                break
            if self.process_image(path):
                break

    def get_sorted_images(self) -> list:
        return sorted(glob.glob(f"{self.images_path}/*.png"))

    def load_image(self, path: str) -> np.ndarray:
        if path not in self.cache:
            img = cv2.imread(path)
            self.cache[path] = img
        return self.cache[path]

    def find(self, image: np.ndarray) -> tuple | None:
        screen = np.array(ImageGrab.grab())  # Take a screenshot of the screen
        screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use template matching to find the image
        res = cv2.matchTemplate(screen_gray, image_gray, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= CONFIDENCE)

        # If a match is found, return the center point of the match
        if len(loc[0]) > 0 and len(loc[1]) > 0:
            top_left = (loc[1][0], loc[0][0])  # Top-left corner of the match
            h, w = image.shape[:2]  # Height and width of the image
            center = (top_left[0] + w // 2, top_left[1] + h // 2)  # Calculate the center point
            return center
        return None

    def process_image(self, path: str) -> bool:
        image = self.load_image(path)
        point = self.find(image)

        if point:
            # Move the mouse to the found point and click using OpenCV
            self.click_at(point)
            path = path.split("\\")[1]
            print(f"\nClicked on {path} at ({point[0]}, {point[1]})")
            return True
        else :
            print(".", end='',flush=True)
        return False

    def click_at(self, point: tuple) -> None:
        # Use pynput to move the mouse to the detected point and click
        from pynput.mouse import Controller
        mouse = Controller()
        currentPosition = mouse.position
        mouse.position = point
        mouse.click(pynput.mouse.Button.left)
        mouse.position = currentPosition

def main():
    MonopolyBot()

if __name__ == "__main__":
    main()
