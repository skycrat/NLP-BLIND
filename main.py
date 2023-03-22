import json
import cv2
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import os
import azure.cognitiveservices.speech as speechsdk

# loading json
CONFIG = json.load(open(".venv\config.json"))
vision_creds = CONFIG["credentials"]["vision"]
# Set up Azure Cognitive Services client
subscription_key = vision_creds["subskey"]
endpoint = vision_creds["endpoint"]
computervision_client = ComputerVisionClient(
    endpoint, CognitiveServicesCredentials(subscription_key))

# Set up Azure Text to Speech client
speech_creds = CONFIG["credentials"]["speech"]
speech_config = speechsdk.SpeechConfig(
    subscription= speech_creds["subskey"], region= speech_creds["region"])
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

# Create a video capture object
cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Display the captured frame in a window
    cv2.imshow("test", frame)
    k = cv2.waitKey(1)

    # Break the loop if the user presses the "ESC" key
    if k % 256 == 27:
        print("Closing the app")
        break

    # Take a screenshot and analyze the spatial environment if the user presses the space bar
    elif k % 256 == 32:
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("Screenshot taken")
        img_counter += 1

        # Analyze the image using the Azure Computer Vision API
        with open(img_name, "rb") as image_stream:
            response = computervision_client.analyze_image_in_stream(
                image_stream, visual_features=['Categories', 'Description', 'Objects'])
            description = response.description.captions[0].text
            print(description)

            # Convert the description to speech using the Azure Text to Speech API
            result = speech_synthesizer.speak_text_async(description).get()

            # Save the speech output to a file
            speech_file = "speech_{}.wav".format(img_counter)
            with open(speech_file, "wb") as file:
                file.write(result.audio_data)

            # Play the speech output
            os.system("start " + speech_file)

# Release the video capture object and destroy the window
cam.release()
cv2.destroyAllWindows()

#prueba exitosa, el programa detecto y describió que tenía un gorro negro y que era una persona