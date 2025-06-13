#!/usr/bin/env python3

import cv2
import cv2.aruco as aruco
import numpy as np
import argparse
import os

INCHES_PER_METER = 39.3701

def generate_aruco_board(markers_x, markers_y, marker_length_m, marker_separation_m, dictionary_name, dpi, margin_pixels):
    """
    Generates an ArUco board image with specified physical dimensions.

    Args:
        markers_x (int): Number of markers in X direction.
        markers_y (int): Number of markers in Y direction.
        marker_length_m (float): Size of the markers in meters.
        marker_separation_m (float): Separation between markers in meters.
        dictionary_name (str): Name of the ArUco dictionary (e.g., "DICT_6X6_250").
        dpi (int): Dots per inch for the output image.
        margin_pixels (int): Margin around the board in pixels.
    """
    marker_length_pixels = int(marker_length_m * INCHES_PER_METER * dpi)
    marker_separation_pixels = int(marker_separation_m * INCHES_PER_METER * dpi)

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

    # Calculate image size based on board dimensions and DPI
    board_width_pixels = markers_x * marker_length_pixels + (markers_x - 1) * marker_separation_pixels
    board_height_pixels = markers_y * marker_length_pixels + (markers_y - 1) * marker_separation_pixels
    
    img_width = board_width_pixels + 2 * margin_pixels
    img_height = board_height_pixels + 2 * margin_pixels
    image_size_pixels = (img_width, img_height)

    img = board.generateImage(image_size_pixels, marginSize=margin_pixels) # marginSize might need adjustment or manual padding
    output_image_file = f"{markers_x}x_{markers_y}_{dictionary_name}_dict_{marker_length_m}len_{marker_separation_m}sep_{margin_pixels}marg_{dpi}dpi.png"
    cv2.imwrite(output_image_file, img)
    print(f"ArUco board saved to {output_image_file} ({img_width}x{img_height} pixels at {dpi} DPI)")
    print(f"  Marker length: {marker_length_m}m ({marker_length_pixels}px), Separation: {marker_separation_m}m ({marker_separation_pixels}px)")


def generate_charuco_board(squares_x, squares_y, square_length_m, marker_length_m, dictionary_name, dpi, margin_pixels):
    """
    Generates a ChArUco board image with specified physical dimensions.

    Args:
        squares_x (int): Number of chessboard squares in X direction.
        squares_y (int): Number of chessboard squares in Y direction.
        square_length_m (float): Size of the chessboard squares in meters.
        marker_length_m (float): Size of the ArUco markers within the squares in meters.
        dictionary_name (str): Name of the ArUco dictionary (e.g., "DICT_6X6_250").
        dpi (int): Dots per inch for the output image.
        margin_pixels (int): Margin around the board in pixels.
    """
    square_length_pixels = int(square_length_m * INCHES_PER_METER * dpi)
    marker_length_pixels = int(marker_length_m * INCHES_PER_METER * dpi)

    if marker_length_pixels >= square_length_pixels:
        print(f"Error: Marker length in pixels ({marker_length_pixels}) must be smaller than square length in pixels ({square_length_pixels}).")
        print(f"  This usually means marker_length_m ({marker_length_m}) is too large relative to square_length_m ({square_length_m}) for the given DPI.")
        return

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

    # Calculate image size based on board dimensions and DPI
    board_width_pixels = squares_x * square_length_pixels
    board_height_pixels = squares_y * square_length_pixels

    img_width = board_width_pixels + 2 * margin_pixels
    img_height = board_height_pixels + 2 * margin_pixels
    final_image_size = (img_width, img_height)
    
    img = board.generateImage(final_image_size, marginSize=margin_pixels, borderBits=1) # marginSize and borderBits might need adjustment

    output_image_file = f"{squares_x}x_{squares_y}_{dictionary_name}_dict_{square_length_m}slen_{marker_length_m}len_{margin_pixels}marg_{dpi}dpi.png"
    cv2.imwrite(output_image_file, img)
    print(f"ChArUco board saved to {output_image_file} ({img_width}x{img_height} pixels at {dpi} DPI)")
    print(f"  Square length: {square_length_m}m ({square_length_pixels}px), Marker length: {marker_length_m}m ({marker_length_pixels}px)")

def main():
    parser = argparse.ArgumentParser(description="Generate ArUco or ChArUco boards for printing with specific physical dimensions.")
    parser.add_argument("board_type", type=str, choices=["aruco", "charuco"], help="Type of board to generate.")
    parser.add_argument("--dict", type=str, default="DICT_6X6_250", help="ArUco dictionary to use (e.g., DICT_6X6_250).")
    parser.add_argument("--dpi", type=int, default=300, help="Dots Per Inch for the output image resolution.")
    parser.add_argument("--margin_m", type=float, default=0.01, help="Margin around the board in meters (e.g., 0.01 for 1cm).")


    # ArUco specific arguments
    parser.add_argument("--markers_x", type=int, default=5, help="[ArUco] Number of markers in X direction.")
    parser.add_argument("--markers_y", type=int, default=7, help="[ArUco] Number of markers in Y direction.")
    parser.add_argument("--marker_len_m_aruco", type=float, default=0.04, help="[ArUco] Marker length in meters (e.g., 0.04 for 4cm).")
    parser.add_argument("--marker_sep_m", type=float, default=0.01, help="[ArUco] Marker separation in meters (e.g., 0.01 for 1cm).")

    # ChArUco specific arguments
    parser.add_argument("--squares_x", type=int, default=5, help="[ChArUco] Number of squares in X direction.")
    parser.add_argument("--squares_y", type=int, default=7, help="[ChArUco] Number of squares in Y direction.")
    parser.add_argument("--square_len_m", type=float, default=0.04, help="[ChArUco] Square length in meters (e.g., 0.04 for 4cm).")
    parser.add_argument("--marker_len_m_charuco", type=float, default=0.025, help="[ChArUco] Marker length in meters (e.g., 0.025 for 2.5cm, must be smaller than square_len_m).")

    args = parser.parse_args()

    margin_pixels = int(args.margin_m * INCHES_PER_METER * args.dpi)

    if args.board_type == "aruco":
        generate_aruco_board(
            args.markers_x,
            args.markers_y,
            args.marker_len_m_aruco,
            args.marker_sep_m,
            args.dict,
            args.dpi,
            margin_pixels
        )
    elif args.board_type == "charuco":
        if args.marker_len_m_charuco >= args.square_len_m:
            print("Error: For ChArUco boards, marker length (meters) must be smaller than square length (meters).")
            return
        generate_charuco_board(
            args.squares_x,
            args.squares_y,
            args.square_len_m,
            args.marker_len_m_charuco,
            args.dict,
            args.dpi,
            margin_pixels
        )

if __name__ == "__main__":
    main()
