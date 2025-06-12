#!/usr/bin/env python3

import cv2
import cv2.aruco as aruco
import numpy as np
import argparse
import os

def generate_aruco_board(markers_x, markers_y, marker_length_pixels, marker_separation_pixels, dictionary_name, output_image_file, image_size_pixels):
    """
    Generates an ArUco board image.

    Args:
        markers_x (int): Number of markers in X direction.
        markers_y (int): Number of markers in Y direction.
        marker_length_pixels (int): Size of the markers in pixels.
        marker_separation_pixels (int): Separation between markers in pixels.
        dictionary_name (str): Name of the ArUco dictionary (e.g., "DICT_6X6_250").
        output_image_file (str): Path to save the generated board image.
        image_size_pixels (tuple): (width, height) of the output image in pixels.
    """
    try:
        aruco_dict_id = getattr(aruco, dictionary_name)
    except AttributeError:
        print(f"Error: ArUco dictionary '{dictionary_name}' not found. Available dictionaries can be found in cv2.aruco.DICT_*")
        return

    dictionary = aruco.getPredefinedDictionary(aruco_dict_id)
    board = aruco.GridBoard(
        size=(markers_x, markers_y),
        markerLength=float(marker_length_pixels),
        markerSeparation=float(marker_separation_pixels),
        dictionary=dictionary
    )

    img = board.generateImage(image_size_pixels)
    cv2.imwrite(output_image_file, img)
    print(f"ArUco board saved to {output_image_file}")

def generate_charuco_board(squares_x, squares_y, square_length_pixels, marker_length_pixels, dictionary_name, output_image_file, image_size_pixels):
    """
    Generates a ChArUco board image.

    Args:
        squares_x (int): Number of chessboard squares in X direction.
        squares_y (int): Number of chessboard squares in Y direction.
        square_length_pixels (int): Size of the chessboard squares in pixels.
        marker_length_pixels (int): Size of the ArUco markers within the squares in pixels.
        dictionary_name (str): Name of the ArUco dictionary (e.g., "DICT_6X6_250").
        output_image_file (str): Path to save the generated board image.
        image_size_pixels (tuple): (width, height) of the output image in pixels.
    """
    try:
        aruco_dict_id = getattr(aruco, dictionary_name)
    except AttributeError:
        print(f"Error: ArUco dictionary '{dictionary_name}' not found. Available dictionaries can be found in cv2.aruco.DICT_*")
        return

    dictionary = aruco.getPredefinedDictionary(aruco_dict_id)
    board = aruco.CharucoBoard(
        size=(squares_x, squares_y),
        squareLength=float(square_length_pixels),
        markerLength=float(marker_length_pixels),
        dictionary=dictionary
    )

    # Calculate image size if not fully specified, ensuring it fits the board
    # The CharucoBoard_create uses squareLength for its internal drawing scale,
    # so the image needs to be at least board_width_pixels x board_height_pixels
    board_width_pixels = squares_x * square_length_pixels
    board_height_pixels = squares_y * square_length_pixels
    
    if image_size_pixels[0] < board_width_pixels or image_size_pixels[1] < board_height_pixels:
        print(f"Warning: Specified image_size_pixels {image_size_pixels} is smaller than the board dimensions "
              f"({board_width_pixels}x{board_height_pixels}). Adjusting to fit board.")
        img_w = max(image_size_pixels[0], board_width_pixels)
        img_h = max(image_size_pixels[1], board_height_pixels)
        final_image_size = (img_w, img_h)
    else:
        final_image_size = image_size_pixels

    img = board.generateImage(final_image_size)
    cv2.imwrite(output_image_file, img)
    print(f"ChArUco board saved to {output_image_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate ArUco or ChArUco boards.")
    parser.add_argument("board_type", type=str, choices=["aruco", "charuco"], help="Type of board to generate.")
    parser.add_argument("-o", "--output", type=str, default="board.png", help="Output image file name (e.g., board.png).")
    parser.add_argument("--dict", type=str, default="DICT_6X6_250", help="ArUco dictionary to use (e.g., DICT_6X6_250).")
    parser.add_argument("--img_width", type=int, default=2000, help="Width of the output image in pixels.")
    parser.add_argument("--img_height", type=int, default=2000, help="Height of the output image in pixels.")

    # ArUco specific arguments
    parser.add_argument("--markers_x", type=int, default=5, help="[ArUco] Number of markers in X direction.")
    parser.add_argument("--markers_y", type=int, default=7, help="[ArUco] Number of markers in Y direction.")
    parser.add_argument("--marker_len_px_aruco", type=int, default=200, help="[ArUco] Marker length in pixels.")
    parser.add_argument("--marker_sep_px", type=int, default=50, help="[ArUco] Marker separation in pixels.")

    # ChArUco specific arguments
    parser.add_argument("--squares_x", type=int, default=5, help="[ChArUco] Number of squares in X direction.")
    parser.add_argument("--squares_y", type=int, default=7, help="[ChArUco] Number of squares in Y direction.")
    parser.add_argument("--square_len_px", type=int, default=200, help="[ChArUco] Square length in pixels.")
    parser.add_argument("--marker_len_px_charuco", type=int, default=120, help="[ChArUco] Marker length in pixels (must be smaller than square_len_px).")

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    image_size = (args.img_width, args.img_height)

    if args.board_type == "aruco":
        generate_aruco_board(
            args.markers_x,
            args.markers_y,
            args.marker_len_px_aruco,
            args.marker_sep_px,
            args.dict,
            args.output,
            image_size
        )
    elif args.board_type == "charuco":
        if args.marker_len_px_charuco >= args.square_len_px:
            print("Error: For ChArUco boards, marker length must be smaller than square length.")
            return
        generate_charuco_board(
            args.squares_x,
            args.squares_y,
            args.square_len_px,
            args.marker_len_px_charuco,
            args.dict,
            args.output,
            image_size
        )

if __name__ == "__main__":
    main()
