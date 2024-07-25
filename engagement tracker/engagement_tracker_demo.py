import argparse
import service
import cv2
import mediapipe as mp
import numpy as np
import datetime
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose



# classes (you can tell that someone who primarily codes in java wrote this)

# main sensitivity info
class User_Settings:
    def __init__(self, prompt_time, sensitivity):
        self.prompt_time = prompt_time
        self.sensitivity = sensitivity

# sliding time window (could not pick a consistent name for this, sorry if all the variables are labeled diff)
class Prompt_Timer:
    def __init__(self, start, current):
        self.start = start
        self.current = current
    
    # updates the timeline (period checking for activity)
    def update_timeline(cls, user):
        cls.current = datetime.datetime.now()
        if diff(cls.start, cls.current) >= user.prompt_time:
            cls.start = cls.current - datetime.timedelta(0, user.prompt_time)
    
    # resets the current activity period
    def reset_timer(cls):
        cls.start = datetime.datetime.now()
        cls.current = datetime.datetime.now() + datetime.timedelta(0, 1)

class Off_Task:
    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time






# methods
    
def diff(start_time, end_time):
    return (end_time - start_time).total_seconds()

# sums value all relevant off-task objs
def total_offtask_time(activity):
    sum = 0
    if len(activity) != 0:
        for time in activity:
            sum += diff(time.start_time, time.end_time) 
        return sum

# updates off-task data to fit within the timeline
def outdated_time(timeline, activity):
    if len(activity) != 0:
        if activity[0].start_time < timeline.start:
            activity[0].start_time = timeline.start

        if (activity[0].start_time >= activity[0].end_time): 
            activity.pop(0) 

# comparing ratio of off-task time in sesh to sensitivity
def has_been_offtask(user, activity):
    if len(activity) != 0:
        percent_offtask = total_offtask_time(activity) / user.prompt_time
        if (percent_offtask > user.sensitivity):
            return True
        return False

# updating off-task obj during the off-task period
def update_offtask(offtask_start_time, activity):
    offtask_end_time = datetime.datetime.now()
    if diff(offtask_start_time, offtask_end_time) >= 1:
        if len(activity) == 0 or activity[len(activity) - 1].start_time != offtask_start_time:
            activity.append(Off_Task(offtask_start_time, offtask_end_time))
        else:
            activity[len(activity) - 1].end_time = offtask_end_time



def create_window(cap):
     # new window
    width = 1920   
    height = 1080   
    image_size = [width, height]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # create window
    cv2.namedWindow('Engagement Detector', flags=cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Engagement Detector', (width,height))





# MAIN
def main(args, color=(224, 255, 255)):

    # face detection stuff
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

    # add user input for settings / preset data
    user = User_Settings(15, 0.4)
    timer = Prompt_Timer(datetime.datetime.now(), datetime.datetime.now()) # HERE
    activity = [] # stores all off-task data
    is_offtask = False
    offtask_start_time = 0
    blossom_activated = False

    # video capture
    cap = cv2.VideoCapture(0)

    # create window
    create_window(cap)

    # confidence mp
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:






        # main loop
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                continue


            # timer stuff (updating)
            timer.update_timeline(user)
            outdated_time(timer, activity)

            # check if user on-task
            if has_been_offtask(user, activity):
                blossom_activated = True
                prompt_start = datetime.datetime.now()
                # hook up blossom code and interaction after this

                # clearing all data
                timer.reset_timer()
                activity = []

            # pausing program until blossom prompting animation finishes
            if blossom_activated:
                prompt_end = datetime.datetime.now()
                cv2.putText(frame, f"GET BACK TO", (250, 500), cv2.FONT_HERSHEY_SIMPLEX, 7, (30, 27, 153), 15)
                cv2.putText(frame, f"WORK!!", (600, 700), cv2.FONT_HERSHEY_SIMPLEX, 7, (30, 27, 153), 15)
                if diff(prompt_start, prompt_end) >= 10:
                    blossom_activated = False


            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(frame)

            # Draw the pose annotation on the image.
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                frame,
                results_pose.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # face detection
            boxes, scores = fd.inference(frame)

            # raw copy for reconstruction
            feed = frame.copy()




            # check for person face
            has_landmark = False
            for results_face in fa.get_landmarks(feed, boxes):
                has_landmark = True
                pitch, yaw = handler(frame, results_face, color)

                # engagement tracker
                if (abs(yaw) >= 20 or pitch >= 9) and not blossom_activated:
                    cv2.putText(frame, f"OFFTASK", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 204, 255), 13)
                    if not is_offtask:
                        offtask_start_time = datetime.datetime.now()
                        is_offtask = True
                    else:
                        update_offtask(offtask_start_time, activity)
                else:
                    is_offtask = False
                    if offtask_start_time != 0:
                        update_offtask(offtask_start_time, activity)
                    offtask_start_time = 0



            # check if person even there
            if not has_landmark and not blossom_activated:
                if results_pose.pose_landmarks != None:
                    cv2.putText(frame, f"OFFTASK", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 204, 255), 13)
                    if not is_offtask:
                            offtask_start_time = datetime.datetime.now()
                            is_offtask = True
                    else:
                        update_offtask(offtask_start_time, activity)
                else:
                    is_offtask = False
                    if offtask_start_time != 0: 
                        update_offtask(offtask_start_time, activity)
                    offtask_start_time = 0

            cv2.imshow("Engagement Detector", frame)
            if cv2.waitKey(1) == ord("q"):
                break
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video demo script.")
    parser.add_argument("-m", "--mode", type=str, default="sparse",
                        choices=["sparse", "dense", "mesh", "pose"])

    args = parser.parse_args()
    main(args)
