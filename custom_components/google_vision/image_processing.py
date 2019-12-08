"""
Perform image processing with Google Vision
"""
import base64
import json
import logging
import time
import io
import os
from datetime import timedelta
from typing import Union, List, Set, Dict

from google.cloud import vision
from google.cloud.vision import types
from google.oauth2 import service_account

from PIL import Image, ImageDraw

import voluptuous as vol

import homeassistant.util.dt as dt_util
from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import split_entity_id
import homeassistant.helpers.config_validation as cv
from homeassistant.components.image_processing import (
    PLATFORM_SCHEMA,
    ImageProcessingEntity,
    ATTR_CONFIDENCE,
    CONF_SOURCE,
    CONF_ENTITY_ID,
    CONF_NAME,
    draw_box,
)


_LOGGER = logging.getLogger(__name__)

SCAN_INTERVAL = timedelta(days=365)  # Effectively disable scan.

CONF_API_KEY_FILE = "api_key_file"
CONF_SAVE_FILE_FOLDER = "save_file_folder"
CONF_TARGET = "target"
DEFAULT_TARGET = "face"
EVENT_OBJECT_DETECTED = "image_processing.object_detected"
EVENT_FACE_DETECTED = "image_processing.face_detected"
EVENT_FILE_SAVED = "image_processing.file_saved"
FILE = "file"
OBJECT = "object"
FACE = "face"

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_API_KEY_FILE): cv.string,
        vol.Optional(CONF_TARGET, default=DEFAULT_TARGET): cv.string,
        vol.Optional(CONF_SAVE_FILE_FOLDER): cv.isdir,
    }
)


def setup_platform(hass, config, add_devices, discovery_info=None):
    """Set up platform."""

    save_file_folder = config.get(CONF_SAVE_FILE_FOLDER)
    if save_file_folder:
        save_file_folder = os.path.join(save_file_folder, "")  # If no trailing / add it

    entities = []
    for camera in config[CONF_SOURCE]:
        entities.append(
            Gvision(
                config.get(CONF_TARGET),
                config.get(CONF_API_KEY_FILE),
                config.get(ATTR_CONFIDENCE),
                save_file_folder,
                camera[CONF_ENTITY_ID],
                camera.get(CONF_NAME),
            )
        )
    add_devices(entities)


def format_confidence(confidence: Union[str, float]) -> float:
    """Takes a confidence from the API like 
       0.55623 and returns 55.6 (%).
    """
    return round(float(confidence) * 100, 1)


def get_box(normalized_vertices: List):
    """
    Return the relative bounxing box coordinates
    defined by the tuple (y_min, x_min, y_max, x_max)
    where the coordinates are floats in the range [0.0, 1.0] and
    relative to the width and height of the image.
    """
    y = []
    x = []
    for box in normalized_vertices:
        y.append(box.y)
        x.append(box.x)

    box = [min(set(y)), min(set(x)), max(set(y)), max(set(x))]

    rounding_decimals = 5
    box = [round(coord, rounding_decimals) for coord in box]
    return box


def get_objects(objects: List[types.LocalizedObjectAnnotation]) -> List[str]:
    """
    Get a list of the unique objects predicted.
    """
    labels = [obj.name.lower() for obj in objects]
    return list(set(labels))


def get_object_confidences(objects: List[types.LocalizedObjectAnnotation], target: str):
    """
    Return the list of confidences of instances of target label.
    """
    confidences = [
        format_confidence(obj.score) for obj in objects if obj.name.lower() == target
    ]
    return confidences


def get_objects_summary(objects: List[types.LocalizedObjectAnnotation]):
    """
    Get a summary of the objects detected.
    """
    objects_labels = get_objects(objects)
    return {
        target: len(get_object_confidences(objects, target))
        for target in objects_labels
    }


class Vision(object):
    """Interact with Google Vision."""

    def __init__(self, api_key_file):
        credentials = service_account.Credentials.from_service_account_file(
            api_key_file
        )
        scoped_credentials = credentials.with_scopes(
            ["https://www.googleapis.com/auth/cloud-platform"]
        )
        self._client = vision.ImageAnnotatorClient(credentials=scoped_credentials)

    def object_localization(self, image_bytes):
        """Return the list of objects in an image from the imge bytes."""
        return self._client.object_localization(image=types.Image(content=image_bytes))

    def face_detection(self, image_bytes):
        """Return the images in an image from the image bytes."""
        return self._client.face_detection(image=types.Image(content=image_bytes))


class Gvision(ImageProcessingEntity):
    """Perform object recognition with Google Vision."""

    def __init__(
        self,
        target,
        api_key_file,
        confidence,
        save_file_folder,
        camera_entity,
        name=None,
    ):
        """Init with the client."""
        self._target = target
        self._api = Vision(api_key_file)
        self._confidence = confidence  # the confidence threshold
        if name:  # Since name is optional.
            self._name = name
        else:
            entity_name = split_entity_id(camera_entity)[1]
            self._name = "{} {} {}".format("google vision", target, entity_name)
        self._camera_entity = camera_entity
        self._state = None  # The number of instances of interest
        self._summary = {}
        self._last_detection = None
        if save_file_folder:
            self._save_file_folder = save_file_folder

    def process_image(self, image):
        self._state = None
        self._summary = {}

        response = self._api.face_detection(image)
        faces = response.face_annotations

        if len(faces) <= 0:
            return
        # print('faces: ',str(len(faces)))
        self._state = len(faces)
        self.fire_face_detected_event(faces)

        if self._state > 0:
            self._last_detection = dt_util.now()

        if hasattr(self, "_save_file_folder") and self._state > 0:
            self.save_image(image, faces, self._target, self._save_file_folder)

    def save_image(self, image, faces, target, directory):
        """Save a timestamped image with bounding boxes around targets."""

        img = Image.open(io.BytesIO(bytearray(image))).convert("RGB")
        draw = ImageDraw.Draw(img)

        for face in faces:
            # box = get_box(face.bounding_poly.vertices)
            box = [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
            # draw_box(draw, box, img.width, img.height)
            draw.line(box + [box[0]], width=5, fill="#00ff00")
        latest_save_path = directory + "google_vision_latest_faces.jpg"
        img.save(latest_save_path)

    def fire_face_detected_event(self, faces):
        for face in faces:
            self.hass.bus.fire(
                EVENT_FACE_DETECTED,
                {
                    ATTR_ENTITY_ID: self.entity_id,
                    # FACE: face.name.lower(),
                    # ATTR_CONFIDENCE: obj_confidence,
                },
            )

    @property
    def camera_entity(self):
        """Return camera entity id from process pictures."""
        return self._camera_entity

    @property
    def state(self):
        """Return the state of the entity."""
        return self._state

    @property
    def name(self):
        """Return the name of the sensor."""
        return self._name

    @property
    def unit_of_measurement(self):
        """Return the unit of measurement."""
        target = self._target
        if self._state != None and self._state > 1:
            target += "s"
        return target

    @property
    def device_state_attributes(self):
        """Return device specific state attributes."""
        attr = {}
        attr["target"] = self._target
        attr["summary"] = self._summary
        if self._last_detection:
            attr[
                "last_{}_detection".format(self._target)
            ] = self._last_detection.strftime("%Y-%m-%d %H:%M:%S")
        return attr
