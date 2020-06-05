"""
Test suite for face detection

There is an image stored in the file test_image.npy that
contains 18 faces. This image is used to test different
cases while detecting a face
"""

import cv2
import numpy as np

import face_detection_mtcnn

# Test image stored in a numpy array for easy access
# This allows the test to load the image without having
# to install OpenCV to read the file
with open("tests/test_image.npy", "rb") as f:
    TEST_IMAGE = np.load(f)

FACES_IN_IMAGE = 18


def test_number_detections():
    """
    Test to check if the correct number of detections was
    done using the detect_face function
    """

    image_rgb = cv2.cvtColor(TEST_IMAGE, cv2.COLOR_BGR2RGB)
    boxes, probs, faces = face_detection_mtcnn.detect_face(image_rgb)

    assert len(boxes) == FACES_IN_IMAGE
    assert len(probs) == FACES_IN_IMAGE
    assert len(faces) == FACES_IN_IMAGE


def test_shape_one_image():
    """
    Test to check if dimensions when using one image are
    correct. The detect face should remove one dimension
    when only working with one image
    """

    image_rgb = cv2.cvtColor(TEST_IMAGE, cv2.COLOR_BGR2RGB)
    boxes, probs, faces = face_detection_mtcnn.detect_face(image_rgb)

    boxes_shape = np.array(boxes).shape
    probs_shape = np.array(probs).shape
    faces_shape = np.array(faces).shape

    # There should be 18 faces with 4 points marking the
    # bounding box for the face
    assert boxes_shape == (18, 4)

    # There should be 18 probs to indicate the probability
    # of each bounding box to belong to a face
    assert probs_shape == (18,  )

    # There should be 18 faces with 5 points. Each point should
    # have to values that represent the location of the 5 landmarks
    # in the face
    assert faces_shape == (18, 5, 2)


def test_shape_list_images():
    """
    Test to check the dimensions when using a list of
    images instead of a single image.

    This is used with analyzing batches of images with
    similar dimensions
    """

    image_rgb = cv2.cvtColor(TEST_IMAGE, cv2.COLOR_BGR2RGB)

    # Creating a list with 3 repeated images
    images = [image_rgb, image_rgb, image_rgb]

    boxes, probs, faces = face_detection_mtcnn.detect_face(images)

    boxes_shape = np.array(boxes).shape
    probs_shape = np.array(probs).shape
    faces_shape = np.array(faces).shape

    # There should be 3 images with 18 faces with 4 points marking the
    # bounding box for the face
    assert boxes_shape == (3, 18, 4)

    # There should be 3 images with 18 probs to indicate the probability
    # of each bounding box to belong to a face
    assert probs_shape == (3, 18)

    # There should be 3 images with 18 faces with 5 points. Each point should
    # have to values that represent the location of the 5 landmarks
    # in the face
    assert faces_shape == (3, 18, 5, 2)
