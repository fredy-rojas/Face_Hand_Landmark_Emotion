import os
import cv2
import numpy as np
import mediapipe as mp
from src.pretrain_models.yunet import YuNet
from src.pretrain_models.facial_fer_model import FacialExpressionRecog


def face_detection_track_face_hands_emotion(model_path_yunet,
                                            model_path_FER_2022july,
                                            boundingBoxScaleFactor=1.05,
                                            ):
    """
    Description:
    ------------
        This function open cam, and track hand and face landmark + a prediction of emotions: 
            0:'angry',
            1:'disgust',
            2:'fearful',
            3:'happy',
            4:'neutral',
            5:'sad',
            6:'surprised',

    Parameteres
    -----------
        model_path_yunet: face_detection_yunet_2023mar.onnx path
        model_path_FER_2022july: facial_expression_recognition_mobilefacenet_2022july.onnx path 
        boundingBoxScaleFactor: bounding box increase or deacrese from the output model face_detection_yunet_2023mar.onnx

    """


    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI files
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480)) 

    net = YuNet(
        modelPath=model_path_yunet,
        inputSize=[320, 320],
        confThreshold=0.85, #.9
        nmsThreshold=0.3,
        topK=100,
        backendId=cv2.dnn.DNN_BACKEND_OPENCV,
        targetId=cv2.dnn.DNN_TARGET_CPU,
        )
    

    fer_model = FacialExpressionRecog(modelPath=model_path_FER_2022july,
                                    backendId=cv2.dnn.DNN_BACKEND_OPENCV,
                                    targetId=cv2.dnn.DNN_TARGET_CPU
                                    )

    boundingBoxScaleFactor = boundingBoxScaleFactor #1.05 # to increase face bounding box by 5%

    """
    This model 'fer_model' was trained to predict following classes 
    _default_emotion_of_fer_model = {0:'angry',
                                    1:'disgust',
                                    2:'fearful',
                                    3:'happy',
                                    4:'neutral',
                                    5:'sad',
                                    6:'surprised',
                                    }
    """

    #======================================================
    # Face with vertices
    mp_face_mesh = mp.solutions.face_mesh
    faceMesh = mp_face_mesh.FaceMesh()
    mp_drawing_styles = mp.solutions.drawing_styles  # For custom drawing styles

    #======================================================
    ## Face land marks and vertices with hands 
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands



    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(1)

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
        while cap.isOpened():
            ret, frame = cap.read()

            #===========================================================
            # Hands Detections section

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Flip on horizontal
            image = cv2.flip(image, 1)
            # Set flag
            image.flags.writeable = False
            # Detections
            results = hands.process(image)
            # Set flag to true
            image.flags.writeable = True
            # RGB 2 BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Rendering results
            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                            )

            #================================================================================================
            #================================================================================================
            # Dynamically set the input size based on the frame size
            h, w, img_chns = image.shape
            net.setInputSize([w, h])
            # Perform face detection
            # Documentation "https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html"
            results0 = net.infer(image) # shape (1, 15) > (faces_detected, 15)
            # shape description 
            # bounding box {x1, y1, w, h,}
            # 5 face landmark {x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm}
            # 1 confidence interval index 14

            #===========================================================
            # Processing faces 
            if len(results0) > 0:
                # print(results)
                for dect in results0: 
                    # Bounding box 
                    x, y, width, height = dect[0:4].astype(int)
                    
                    # Draw bounding box
                    cv2.rectangle(image, 
                                (x, y), 
                                (x + int(width*boundingBoxScaleFactor), y + int(height*boundingBoxScaleFactor)), 
                                (0, 255, 0), 
                                2)

                    #-------------
                    # EMOTION DETECTION
                    emo_code = fer_model.infer(image, dect[:-1]) 
                    emo_label_pred = FacialExpressionRecog.getDesc(emo_code.item())
        
                    label_emo = f"Emo: {emo_label_pred}"
        
                    emotion_colors = {
                                "angry": (0, 0, 255),       # Red
                                "disgust": (0, 128, 0),     # Dark Green
                                "fearful": (128, 0, 128),      # Purple
                                "happy": (0, 255, 255),     # Yellow
                                "neutral": (0, 255, 0),  # green
                                "sad": (255, 0, 0),         # Blue
                                "surprised": (0, 165, 255),  # Orange
                                
                            }
        
                    cv2.putText(image, label_emo, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, emotion_colors[emo_label_pred], 2)

                    label_re = f"RightEyes: x={int(dect[4])}, y={int(dect[5])}"
                    cv2.putText(image, label_re, (x, y + int(height*boundingBoxScaleFactor) + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, .5, emotion_colors[emo_label_pred], 2)
                    
            #================================================================================================
            #================================================================================================
            results = faceMesh.process(image)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw landmarks and connect them with lines
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,  # Connect landmarks
                        landmark_drawing_spec=None,  # Disable points (optional)
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()  # Line style
                    )


            #========================================================================
            # Write the processed frame to the video file
            # out.write(image)

            #=========================================================================
            cv2.imshow('EMO_HANDS_FACE_LandMarks', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    # out.release()  # Save the video file
    cv2.destroyAllWindows()