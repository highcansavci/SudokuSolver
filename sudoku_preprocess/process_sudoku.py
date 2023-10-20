import cv2
import torch
import numpy as np

from sudoku_ocr.inference import ocr_inference
from sudoku_preprocess.preprocess_sudoku import SudokuPreprocessor

sudoku_preprocessor = SudokuPreprocessor()


class SudokuProcessor:
    def create_grid_mask(self, vertical, horizontal):
        grid = cv2.add(horizontal, vertical)
        grid = cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 235, 2)
        grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
        pts = cv2.HoughLines(grid, .3, np.pi/90, 200)
        lines = sudoku_preprocessor.draw_lines(grid, pts)
        mask = cv2.bitwise_not(lines)
        return mask

    def get_grid_lines(self, img, length=10):
        horizontal = sudoku_preprocessor.grid_line_helper(img, 1, length)
        vertical = sudoku_preprocessor.grid_line_helper(img, 0, length)
        return vertical, horizontal

    def find_contours(self, img, original):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        polygon = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, closed=True)
            approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, closed=True)
            num_corners = len(approx)

            if num_corners == 4 and area > 1000:
                polygon = cnt
                break

        if polygon is not None:
            top_left = sudoku_preprocessor.find_extreme_corners(polygon, min, np.add)
            top_right = sudoku_preprocessor.find_extreme_corners(polygon, max, np.subtract)
            bot_left = sudoku_preprocessor.find_extreme_corners(polygon, min, np.subtract)
            bot_right = sudoku_preprocessor.find_extreme_corners(polygon, max, np.add)

            if bot_right[1] - top_right[1] == 0:
                return []
            if not (0.95 < ((top_right[0] - top_left[0]) / (bot_right[1] - top_right[1])) < 1.05):
                return []

            cv2.drawContours(original, polygon, 0, (0, 0, 255), 3)
            [sudoku_preprocessor.draw_extreme_corners(x, original) for x in [top_left, top_right, bot_right, bot_left]]

            return [top_left, top_right, bot_right, bot_left]

        return []

    def warp_image(self, corners, original):
        corners = np.array(corners, dtype=np.float32)
        top_left, top_right, bot_right, bot_left = corners

        width = int(max([
            np.linalg.norm(top_right - bot_right),
            np.linalg.norm(top_left - bot_left),
            np.linalg.norm(bot_right - bot_left),
            np.linalg.norm(top_left - top_right)
        ]))

        mapping = np.array([[0, 0], [width - 1, 0], [width - 1, width - 1], [0, width - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(corners, mapping)
        return cv2.warpPerspective(original, matrix, (width, width)), matrix


    def split_into_squares(self, warped_img):
        squares = []
        width = warped_img.shape[0] // 9

        for j in range(9):
            for i in range(9):
                p1 = (i * width, j * width)
                p2 = ((i + 1) * width, (j + 1) * width)
                squares.append(warped_img[p1[1]:p2[1], p1[0]:p2[0]])

        return squares


    def clean_squares(self, squares):
        cleaned_squares = []
        i = 0

        for square in squares:
            new_img, is_number = sudoku_preprocessor.clean_(square)
            if is_number:
                cleaned_squares.append(new_img)
                i += 1
            else:
                cleaned_squares.append(0)

        return cleaned_squares

    def recognize_digits(self, squares_processed, model):
        s = ""
        formatted_squares = []
        location_of_zeroes = set()

        blank_image = np.zeros_like(cv2.resize(squares_processed[0], (28, 28)))

        for i in range(len(squares_processed)):
            if type(squares_processed[i]) == int:
                location_of_zeroes.add(i)
                formatted_squares.append(blank_image)
            else:
                img = cv2.resize(squares_processed[i], (28, 28))
                formatted_squares.append(img)

        formatted_squares = np.array(formatted_squares)
        all_preds = ocr_inference(formatted_squares, model)
        for i in range(len(all_preds)):
            if i in location_of_zeroes:
                s += "0"
            else:
                s += str(all_preds[i])

        return s

    def draw_digits_on_warped(self, warped_img, solved_puzzle, squared_process):
        width = warped_img.shape[0] // 9
        img_w_text = np.zeros_like(warped_img)

        index = 0
        for j in range(9):
            for i in range(9):
                if type(squared_process[index]) == int:
                    p1 = (i * width, j * width)
                    p2 = ((i + 1) * width, (j + 1) * width)
                    center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                    text_size = cv2.getTextSize(str(solved_puzzle[index]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 4)
                    text_origin = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)
                    cv2.putText(warped_img, str(solved_puzzle[index]),
                                text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                index += 1

        return img_w_text

    def unwarp_image(self, img_src, img_dest, pts, time):
        pts = np.array(pts)
        height, width = img_src.shape[0], img_src.shape[1]
        pts_source = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, width - 1]], dtype=np.float32)
        h, status = cv2.findHomography(pts_source, pts)
        warped = cv2.warpPerspective(img_src, h, (img_dest.shape[1], img_dest.shape[0]))
        cv2.fillConvexPoly(img_dest, pts, 0, 16)

        dst_img = cv2.add(img_dest, warped)
        dst_img_height, dst_img_width = dst_img.shape[0], dst_img.shape[1]
        cv2.putText(dst_img, time, (dst_img_width - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        return dst_img
