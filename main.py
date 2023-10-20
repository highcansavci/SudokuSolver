import torch
import numpy as np
import time as t
import cv2
from sudoku_ocr.ocr_model import DigitOCR
from sudoku_preprocess.preprocess_sudoku import SudokuPreprocessor
from sudoku_preprocess.process_sudoku import SudokuProcessor
from sudoku_solver.solver import SudokuSolver

FRAME_WIDTH = 960
FRAME_HEIGHT = 720

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    frame_rate = 30

    # width is id number 3, height is id 4
    cap.set(3, FRAME_WIDTH)
    cap.set(4, FRAME_HEIGHT)

    # change brightness to 150
    cap.set(10, 150)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DigitOCR()
    model.to(device)
    model.load_state_dict(torch.load("sudoku_ocr/ocr_digit.pth", map_location=device))

    sudoku_preprocessor = SudokuPreprocessor()
    sudoku_processor = SudokuProcessor()
    sudoku_solver = SudokuSolver()
    prev = 0
    seen = dict()

    while True:
        time_elapsed = t.time() - prev

        success, img = cap.read()

        if time_elapsed > 1. / frame_rate:
            prev = t.time()
            img_result = img.copy()
            img_corners = img.copy()

            processed_img = sudoku_preprocessor.preprocess(img)
            corners = sudoku_processor.find_contours(processed_img, img_corners)

            if corners:
                warped, matrix = sudoku_processor.warp_image(corners, img)
                warped_processed = sudoku_preprocessor.preprocess(warped)

                vertical_lines, horizontal_lines = sudoku_processor.get_grid_lines(warped_processed)
                mask = sudoku_processor.create_grid_mask(vertical_lines, horizontal_lines)
                numbers = cv2.bitwise_and(warped_processed, mask)
                cv2.imwrite("result.png", numbers)

                squares = sudoku_processor.split_into_squares(numbers)
                squares_processed = sudoku_processor.clean_squares(squares)

                squares_guesses = sudoku_processor.recognize_digits(squares_processed, model)
                print(squares_guesses)

                # if it is impossible, continue
                if squares_guesses in seen and seen[squares_guesses] is False:
                    continue

                # if we already solved this puzzle, just fetch the solution
                if squares_guesses in seen:
                    sudoku_processor.draw_digits_on_warped(warped, seen[squares_guesses][0], squares_processed)
                    img_result = sudoku_processor.unwarp_image(warped, img_result, corners, seen[squares_guesses][1])

                else:
                    solved_puzzle, time = sudoku_solver.solve_wrapper(squares_guesses)
                    if solved_puzzle is not None:
                        sudoku_processor.draw_digits_on_warped(warped, solved_puzzle, squares_processed)
                        img_result = sudoku_processor.unwarp_image(warped, img_result, corners, time)
                        seen[squares_guesses] = [solved_puzzle, time]

                    else:
                        seen[squares_guesses] = False

        cv2.imshow('window', img_result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
