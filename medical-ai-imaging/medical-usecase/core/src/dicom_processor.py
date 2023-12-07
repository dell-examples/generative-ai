# Created by Scalers AI for Dell Inc.

import logging
import os
import shutil
import time
import uuid

import cv2
import numpy as np
import pydicom
import requests


class DicomProcessor:
    def __init__(
        self,
        orthanc_url,
        username,
        password,
        output_directory,
        infer_server,
        logger,
    ):
        """
        Initialize the DicomProcessor.

        Parameters:
        - orthanc_url (str): URL of the Orthanc server.
        - username (str): Username for authentication.
        - password (str): Password for authentication.
        - output_directory (str): Directory path to save downloaded and processed DICOM files.
        - infer_server: Inference server object.
        - logger: Logger object for logging.

        Returns:
        - None
        """
        self.orthanc_url = orthanc_url
        self.username = username
        self.password = password
        self.output_directory = output_directory
        self.infer_server = infer_server
        self.logger = logger

    def request_get(self, url, auth=None):
        """
        Make a GET request to the specified URL.

        Parameters:
        - url (str): The URL to make the GET request.
        - auth (tuple): Tuple containing username and password for authentication.

        Returns:
        - response: Response object if successful, None if request fails.
        """
        max_retries = 3
        for i in range(max_retries):
            try:
                response = requests.get(url, auth=auth)
                if response.status_code == 200:
                    return response
                elif response.status_code == 401:
                    self.logger.error(
                        "Authentication failed. Check your username and password."
                    )
                    return None
            except requests.exceptions.RequestException as e:
                if i < max_retries - 1:
                    logging.error(f"Connection error: {e}. Retrying...")
                    time.sleep(2)  # Add a delay before retrying
                else:
                    logging.error(
                        f"Failed to establish connection after multiple retries: {e}"
                    )
        return None

    def get_patient_ids(self):
        """
        Retrieve a list of patient IDs from the Orthanc server.

        Returns:
        - patient_ids (list): List of patient IDs.
        """
        # Query Orthanc for the list of patients
        changes_url = f"{self.orthanc_url}/changes"
        try:
            # Make a GET request to the changes API
            response = self.request_get(
                changes_url, auth=(self.username, self.password)
            )

            if response is not None:
                changes_data = response.json()["Changes"]

                # Extract the patient IDs from the changes data
                patient_ids = [
                    change["ID"]
                    for change in changes_data
                    if change["ChangeType"] == "NewPatient"
                ]
                self.logger.info(f"patients {patient_ids}")

                # Clear the changes by making a DELETE request
                clear_changes_response = requests.delete(
                    changes_url, auth=(self.username, self.password)
                )

                if clear_changes_response.status_code == 200:
                    self.logger.info("Changes cleared successfully.")
                else:
                    self.logger.warning("Failed to clear changes.")
                return patient_ids
            else:
                self.logger.error("Failed to retrieve the list of patients.")
                return []

        except requests.exceptions.RequestException as e:
            self.logger.error(f"An error occurred: {e}")
            return []

    def process_dicom(self, patient_id, output_directory):
        """
        Process DICOM images, add text, and upload them to the Orthanc server.

        Parameters:
        - patient_id (str): ID of the patient.
        - output_directory (str): Directory to save the processed DICOM files.

        Returns:
        - None
        """

        def get_dicom_info(url):
            response = self.request_get(
                url, auth=(self.username, self.password)
            )
            if response is None:
                return None
            return response.json()

        def save_dicom_file(dicom_url, file_path):
            response = self.request_get(
                dicom_url, auth=(self.username, self.password)
            )
            if response is not None:
                with open(file_path, "wb") as f:
                    f.write(response.content)
            else:
                self.logger.error(
                    f"Failed to download DICOM file from {dicom_url}. Status code: {response.status_code}"
                )

        def process_image_and_upload(dcm, output_path):
            pixel_data = dcm.pixel_array
            image_raw = pixel_data
            image_raw = cv2.resize(image_raw, (224, 224))
            image_raw = image_raw.astype("float32") / 255.0

            # Add a batch dimension to the image
            image_raw = np.expand_dims(image_raw, axis=-1)
            image_raw = np.expand_dims(image_raw, axis=0)

            start = time.time()
            res = self.infer_server.infer(image_raw)
            end = time.time()
            self.logger.info(res)
            interval = end - start
            interval_text = f"Inference time: {interval:.2f} second(s)"

            # Add text using OpenCV
            if res == "Pneumonia":
                text = "Pneumonia diagnosed"
            else:
                text = res
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            font_color = (255, 255, 255)
            thickness = 5
            text_position = (30, 70)
            image_with_text = cv2.putText(
                pixel_data,
                text,
                text_position,
                font,
                font_scale,
                font_color,
                thickness,
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (255, 255, 255)
            thickness = 5
            text_position = (30, 120)
            image_with_text = cv2.putText(
                pixel_data,
                interval_text,
                text_position,
                font,
                font_scale,
                font_color,
                thickness,
            )

            # Save the processed image as a new DICOM file
            new_dcm = dcm.copy()
            new_dcm.PixelData = image_with_text.tobytes()
            new_dcm.file_meta.TransferSyntaxUID = (
                pydicom.uid.ExplicitVRLittleEndian
            )
            new_dcm.Rows, new_dcm.Columns = (
                image_with_text.shape[0],
                image_with_text.shape[1],
            )
            new_dcm.SOPInstanceUID = str(uuid.uuid4())
            new_dcm.save_as(output_path)

            # Upload the new DICOM instance to Orthanc
            upload_dicom(output_path)

            # Get the Study directory
            parent_dir = os.path.dirname(output_path)

            # Delete the Study directory
            if os.path.exists(parent_dir):
                shutil.rmtree(parent_dir)
                print(f"Study directory {parent_dir} deleted.")
            else:
                print(
                    f"Study directory {parent_dir} does not exist or could not be deleted."
                )

        def upload_dicom(dicom_path):
            with open(dicom_path, "rb") as f:
                new_dcm_bytes = f.read()

            response = requests.post(
                f"{self.orthanc_url}/instances",
                data=new_dcm_bytes,
                headers={"content-type": "application/dicom"},
                auth=(self.username, self.password),
            )

            if response.status_code == 200:
                self.logger.info(
                    f"Processed DICOM image uploaded successfully."
                )
            else:
                self.logger.error(
                    f"Error uploading processed DICOM image. Status code: {response.status_code}"
                )

        studies_url = f"{self.orthanc_url}/patients/{patient_id}/studies"
        studies_data = get_dicom_info(studies_url)

        if not studies_data:
            self.logger.info(f"No study found for this patient {patient_id}")
            return

        for study_info in studies_data:
            study_instance_uid = study_info["ID"]
            study_directory = os.path.join(
                output_directory, study_instance_uid
            )
            try:
                os.makedirs(study_directory, exist_ok=True)
                self.logger.info(
                    f"Study directory created: {output_directory}"
                )
            except OSError as e:
                self.logger.error(f"Error creating directory: {e}")

            series_url = (
                f"{self.orthanc_url}/studies/{study_instance_uid}/series"
            )
            series_data = get_dicom_info(series_url)

            if not series_data:
                continue

            first_series = series_data[0]
            first_series_instance_uid = first_series["ID"]
            instances_url = f"{self.orthanc_url}/series/{first_series_instance_uid}/instances"
            instances_data = get_dicom_info(instances_url)

            if not instances_data:
                continue

            first_instance = instances_data[0]
            instance_uid = first_instance["ID"]
            dicom_file_url = (
                f"{self.orthanc_url}/instances/{instance_uid}/file"
            )
            dicom_file_path = os.path.join(
                study_directory, f"{instance_uid}.dcm"
            )
            save_dicom_file(dicom_file_url, dicom_file_path)

            dcm = pydicom.dcmread(dicom_file_path)
            output_dicom_path = os.path.join(
                study_directory, f"{instance_uid}_processed.dcm"
            )

            # Process the DICOM image and upload
            process_image_and_upload(dcm, output_dicom_path)
