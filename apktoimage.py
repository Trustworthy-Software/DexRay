import sys
import math
from androguard.core.bytecodes.apk import APK
from PIL import Image

def get_dex_bytes(apk: APK) -> bytes:
    for f in apk.get_files():
        if f.endswith(".dex"):
            yield apk.get_file(f)


def generate_png(apk: APK, filename: str, folder: str):
    stream = bytes()
    for s in get_dex_bytes(apk):
        stream += s
    current_len = len(stream)
    image = Image.frombytes(mode='L', size=(1, current_len), data=stream)
    image.save(f"{folder}/{filename}.png")
    


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise Exception("[!] Usage: python3 apktoimage.py APK DESTINATION")
    else:
        filename = sys.argv[1]
        destination_folder = sys.argv[2]
    try:
        apk = APK(filename)
        generate_png(apk, filename, destination_folder)
        print(f"Image successfully generated from {filename}")
    except Exception as e:
        print("[!] An exception occured with: {}".format(filename))
        print("Exception: {}".format(e))
