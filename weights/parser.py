from pathlib import Path
import sys


file_path = Path(__file__).resolve()

root_path = str(file_path.parent)

def get_path_model_object():
    return root_path + "/plateYolo.pt"
def get_path_model_char():
    return root_path + "/CharsYolo.pt"
