import sys
sys.path.insert(0, 'verification_ocr_engine')
import tesserocr
from PIL import Image
import cv2
import numpy as np
import sign_ocr_engine
from check_traffic_ocr import synthetic_signs
from traffic_sign_worker import SignRecognizer

api = tesserocr.PyTessBaseAPI(path='verification_ocr_engine', lang='eng')
def read(image, config=''):
    psm = int(config.split('--psm ')[1].split()[0])
    letters = config.split('whitelist=')[1].split()[0]
    image, config = sign_ocr_engine.prepare_text(image, config)
    letters = config.split('whitelist=')[1].split()[0]
    api.SetPageSegMode(psm)
    api.SetVariable('tessedit_char_whitelist', letters)
    api.SetImage(Image.fromarray(image))
    result = api.GetUTF8Text()
    cv2.imwrite('verification_ocr_engine/roi_'+letters+'.png', image)
    api.SetVariable('tessedit_char_whitelist', '')
    api.SetImage(Image.fromarray(image))
    print('unrestricted',repr(api.GetUTF8Text()))
    print('OCR', image.shape, psm, repr(result))
    return result
sign_ocr_engine.image_to_string = read
if __name__ == '__main__':
    recognizer = SignRecognizer()
    for i in (1, 2):
        r = recognizer.process(synthetic_signs(), 1000+i*.1, 0, i)
        print({k:r[k] for k in ('stop_text','park_text','p_shape','stop_confirmed','park_confirmed')})
