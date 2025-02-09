from pathlib import Path
import sys


file_path = Path(__file__).resolve()

root_path = file_path.parent

def get_path_model_object():
    return root_path + "/best.pt"
def get_path_model_char():
    return root_path + "/yoyolov8n_char_new.pt"
