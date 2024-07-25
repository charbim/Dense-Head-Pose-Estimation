import argparse
import service
import cv2
import simpleaudio as sa


def main(args, color=(224, 255, 255)):
    fd = service.UltraLightFaceDetecion("weights/RFB-320.tflite",
                                        conf_threshold=0.8)

    if args.mode in ["sparse", "pose"]:
        fa = service.DepthFacialLandmarks("weights/sparse_face.tflite")
    else:
        fa = service.DenseFaceReconstruction("weights/dense_face.tflite")
        if args.mode == "mesh":
            color = service.TrianglesMeshRender("asset/render.so",
                                                "asset/triangles.npy")

    handler = getattr(service, args.mode)

    # video capture
    cap = cv2.VideoCapture(0)
    wave_object = sa.WaveObject.from_wave_file('man_scream.wav')
    play_object = wave_object.play()


    # new window
    width = 1920   
    height = 1080   
    image_size = [width, height]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # create window
    cv2.namedWindow('Engagement Detector', flags=cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Engagement Detector', (width,height))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # face detection
        boxes, scores = fd.inference(frame)

        # raw copy for reconstruction
        feed = frame.copy()

        for results in fa.get_landmarks(feed, boxes):
            pitch, yaw = handler(frame, results, color)

            # engagement tracker
            if abs(yaw) >= 20 or pitch >= 7:
                cv2.putText(frame, f"OFFTASK", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 255), 13)
                if not play_object.is_playing():
                   wave_object.play()
            else: 
                sa.stop_all()


        # cv2.imwrite(f'draft/gif/trans/img{counter:0>4}.jpg', frame)

        cv2.imshow("Engagement Detector", frame)
        if cv2.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video demo script.")
    parser.add_argument("-m", "--mode", type=str, default="sparse",
                        choices=["sparse", "dense", "mesh", "pose"])

    args = parser.parse_args()
    main(args)
