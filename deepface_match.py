import cv2
import time
import numpy as np
from deepface.detectors import YuNet
from deepface.modules import detection
from deepface.basemodels import Facenet
from deepface.modules import preprocessing
from typing import Tuple, Union, Dict, Any, List
from tensorflow.keras.preprocessing import image
from deepface.detectors.DetectorWrapper import rotate_facial_area
from deepface.models.Detector import DetectedFace, FacialAreaRegion
from deepface.modules.verification import find_distance, find_threshold

# globals
model_name = 'Facenet512'
img_path = 'test_mage.jpg'
detector_backend = 'yunet'
enforce_detection = False
grayscale = False
human_readable=False
normalization='base'
# load model
rec = Facenet.FaceNet128dClient()
target_size = rec.input_shape
# detector
det = YuNet.YuNetClient()
distance_metric = 'cosine'

def verify2(emb1, img2_path, threshold=0.7):
    emb2 = predict_one(img2_path)
    # find the face pair with minimum distance
    # threshold = find_threshold(model_name, distance_metric)
    distance = find_distance(emb1, emb2, distance_metric)
    resp_obj = {
        "verified": distance <= threshold,
        "distance": distance,
        "threshold": threshold,
        "model": model_name,
        "detector_backend": detector_backend,
        "similarity_metric": 'distance_metric',
    }
    return resp_obj

def verify(img1_path, img2_path):
    emb1 = predict_one(img1_path)
    emb2 = predict_one(img2_path)
    # find the face pair with minimum distance
    threshold = find_threshold(model_name, distance_metric)
    distance = find_distance(emb1, emb2, distance_metric)
    resp_obj = {
        "verified": distance <= threshold,
        "distance": distance,
        "threshold": threshold,
        "model": model_name,
        "detector_backend": detector_backend,
        "similarity_metric": 'distance_metric',
    }
    return resp_obj

def predict_one(img_path):
    embeddings = []
    # load image
    img, img_name = preprocessing.load_image(img_path)
    if img is None:
        raise ValueError(f"Exception while loading {img_name}")
    facial_areas = det.detect_faces(img=img)
    results = []
    base_region = FacialAreaRegion(x=0, y=0, w=img.shape[1], h=img.shape[0], confidence=0)
    for facial_area in facial_areas:
        x = facial_area.x
        y = facial_area.y
        w = facial_area.w
        h = facial_area.h
        left_eye = facial_area.left_eye
        right_eye = facial_area.right_eye
        confidence = facial_area.confidence
        # extract detected face unaligned
        detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]
        aligned_img, angle = detection.align_face(
            img=img, left_eye=left_eye, right_eye=right_eye
        )
        rotated_x1, rotated_y1, rotated_x2, rotated_y2 = rotate_facial_area(
            facial_area=(x, y, x + w, y + h),
            angle=angle,
            size=(img.shape[0], img.shape[1])
        )
        detected_face = aligned_img[
            int(rotated_y1) : int(rotated_y2),
            int(rotated_x1) : int(rotated_x2)]

        result = DetectedFace(
            img=detected_face,
            facial_area=FacialAreaRegion(
                x=x, y=y, h=h, w=w, confidence=confidence, left_eye=left_eye, right_eye=right_eye
            ),
            confidence=confidence,
        )
        results.append(result)
    face_objs = results.copy()
    resp_objs = []
    # in case of no face found
    if len(face_objs) == 0 and enforce_detection is True:
        if img_name is not None:
            raise ValueError(
                f"Face could not be detected in {img_name}."
                "Please confirm that the picture is a face photo "
                "or consider to set enforce_detection param to False."
            )
        else:
            raise ValueError(
                "Face could not be detected. Please confirm that the picture is a face photo "
                "or consider to set enforce_detection param to False."
            )

    if len(face_objs) == 0 and enforce_detection is False:
        face_objs = [DetectedFace(img=img, facial_area=base_region, confidence=0)]

    for face_obj in face_objs:
        current_img = face_obj.img
        current_region = face_obj.facial_area

        if current_img.shape[0] == 0 or current_img.shape[1] == 0:
            continue

        if grayscale is True:
            current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

        # resize and padding
        if target_size is not None:
            factor_0 = target_size[0] / current_img.shape[0]
            factor_1 = target_size[1] / current_img.shape[1]
            factor = min(factor_0, factor_1)

            dsize = (
                int(current_img.shape[1] * factor),
                int(current_img.shape[0] * factor),
            )
            current_img = cv2.resize(current_img, dsize)

            diff_0 = target_size[0] - current_img.shape[0]
            diff_1 = target_size[1] - current_img.shape[1]
            if grayscale is False:
                # Put the base image in the middle of the padded image
                current_img = np.pad(
                    current_img,
                    (
                        (diff_0 // 2, diff_0 - diff_0 // 2),
                        (diff_1 // 2, diff_1 - diff_1 // 2),
                        (0, 0),
                    ),
                    "constant",
                )
            else:
                current_img = np.pad(
                    current_img,
                    (
                        (diff_0 // 2, diff_0 - diff_0 // 2),
                        (diff_1 // 2, diff_1 - diff_1 // 2),
                    ),
                    "constant",
                )

            # double check: if target image is not still the same size with target.
            if current_img.shape[0:2] != target_size:
                current_img = cv2.resize(current_img, target_size)

        # normalizing the image pixels
        # what this line doing? must?
        img_pixels = image.img_to_array(current_img)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255  # normalize input in [0, 1]
        # discard expanded dimension
        if human_readable is True and len(img_pixels.shape) == 4:
            img_pixels = img_pixels[0]

        resp_objs.append(
            {
                "face": img_pixels[:, :, ::-1] if human_readable is True else img_pixels,
                "facial_area": {
                    "x": int(current_region.x),
                    "y": int(current_region.y),
                    "w": int(current_region.w),
                    "h": int(current_region.h),
                    "left_eye": current_region.left_eye,
                    "right_eye": current_region.right_eye,
                },
                "confidence": round(current_region.confidence, 2),
            }
        )

    img_objs = resp_objs.copy()
    img_embedding_obj = []
    for img_obj in img_objs:
        img = img_obj["face"]
        region = img_obj["facial_area"]
        confidence = img_obj["confidence"]
        # custom normalization
        img = preprocessing.normalize_input(img=img, normalization=normalization)
        embedding = rec.find_embeddings(img)
        resp_obj = {}
        resp_obj["embedding"] = embedding
        resp_obj["facial_area"] = region
        resp_obj["face_confidence"] = confidence
        img_embedding_obj.append(resp_obj)
        img_embedding = img_embedding_obj[0]["embedding"]
        embeddings.append(img_embedding)

    return embeddings[0]

if __name__ == '__main__':
    t1 = time.perf_counter()
    for i in range(50):
        result = verify(img1_path = "test_mage.jpg", img2_path = "test_mage.jpg")
    t2 = time.perf_counter()
    print(f'total time: {t2 - t1}')
