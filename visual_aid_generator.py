import matplotlib
# Try to set a non-interactive backend in case of display issues in sandbox
try:
    matplotlib.use('Agg') # Use Agg backend for writing to file, no GUI needed
    MATPLOTLIB_BACKEND_SET = True
except Exception as e:
    print(f"Could not set Matplotlib backend to Agg: {e}")
    MATPLOTLIB_BACKEND_SET = False

import matplotlib.pyplot as plt
import numpy as np
import os

def generate_simple_bar_chart(data_dict, title="Simple Bar Chart", xlabel="Categories", ylabel="Values", output_filename="chart.png"):
    """
    Generates a simple bar chart from a dictionary and saves it as a PNG file.
    Example data_dict: {"A": 10, "B": 20, "C": 15}
    Returns the path to the saved chart or None if error.
    """
    if not MATPLOTLIB_BACKEND_SET:
        print("Matplotlib backend not set, chart generation might fail or try to use GUI.")
        # Fallback, try to generate anyway, might work or give different error

    if not data_dict:
        print("No data provided for chart generation.")
        return None

    categories = list(data_dict.keys())
    values = list(data_dict.values())

    try:
        fig, ax = plt.subplots()
        ax.bar(categories, values)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # Ensure the directory for output_filename exists if it's nested
        # For now, assuming output_filename is in the current directory

        plt.savefig(output_filename)
        plt.close(fig) # Close the figure to free memory

        print(f"Chart '{output_filename}' generated successfully.")
        return output_filename
    except Exception as e:
        print(f"Error generating chart with Matplotlib: {e}")
        # This could be due to Matplotlib not being installed, backend issues, etc.
        return None

if __name__ == "__main__":
    print("--- Visual Aid Generator Test (Matplotlib) ---")

    sample_data = {"Apples": 5, "Bananas": 12, "Oranges": 8, "Grapes": 15}
    chart_file = generate_simple_bar_chart(sample_data, title="Fruit Counts", output_filename="fruit_counts_chart.png")

    if chart_file and os.path.exists(chart_file):
        print(f"Test chart saved to: {chart_file}")
        print(f"Please check the file '{chart_file}' to verify.")
        # In a real application, you might display this image or pass its path.
        # For example, using OpenCV to display if in a GUI environment:
        # try:
        #     import cv2
        #     img = cv2.imread(chart_file)
        #     if img is not None:
        #         cv2.imshow("Generated Chart", img)
        #         cv2.waitKey(0)
        #         cv2.destroyAllWindows()
        #     else:
        #         print(f"Could not read generated chart with OpenCV from {chart_file}")
        # except Exception as e_cv:
        #     print(f"OpenCV display error: {e_cv}")
    elif chart_file: # Generated but not found (should not happen if generate_simple_bar_chart is correct)
        print(f"Chart generation reported success but file '{chart_file}' not found.")
    else:
        print("Chart generation failed.")
        print("This could be due to Matplotlib not being installed correctly (from previous disk space issues) or other environment limitations.")

    print("\n--- Test with empty data ---")
    empty_chart_file = generate_simple_bar_chart({}, title="Empty Data Chart", output_filename="empty_chart.png")
    if empty_chart_file:
        print(f"Empty chart test reported success (should ideally not create a file or report error): {empty_chart_file}")
    else:
        print("Empty chart test correctly reported an issue or did not generate a file.")

    print("\n--- Visual Aid Generator Test Complete ---")
