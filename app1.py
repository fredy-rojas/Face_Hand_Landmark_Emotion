import os
from src.track_landmark.track import face_detection_track_face_hands_emotion

#==========================================================================
class cfg:
    """Configuration class for model paths."""
    BASE_MODEL_PATH = os.path.join("model")
    YUNET_2023mar = os.path.join(BASE_MODEL_PATH, 
                                 "face_detection", 
                                 "pretrain",
                                 "face_detection_yunet_2023mar.onnx")
    FER_2022july = os.path.join(BASE_MODEL_PATH, 
                                "facial_expresion_recognition", 
                                "pretrain", 
                                "facial_expression_recognition_mobilefacenet_2022july.onnx")

#==========================================================================
def main(model_path_yunet=cfg.YUNET_2023mar,
         model_path_FER_2022july=cfg.FER_2022july,
         boundingBoxScaleFactor=1.05,
         cam_id=1,
         ):
    face_detection_track_face_hands_emotion(model_path_yunet=model_path_yunet,
                                            model_path_FER_2022july=model_path_FER_2022july,
                                            boundingBoxScaleFactor=boundingBoxScaleFactor,
                                            cam_id=cam_id,
                                            )

if __name__ == "__main__":
    main()