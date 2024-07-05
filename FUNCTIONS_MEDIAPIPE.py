import cv2
import mediapipe as mp
import time
import numpy as np
import os
from tqdm import tqdm
from PIL import Image

def extract_keypointsHOLISTIC(results, extract_pose=True, extract_face=True, extract_lh=True, extract_rh=True):
    ## pose
    if extract_pose:
        pose_array = np.zeros((23, 4))
        # take all the keypoints until the pelvis keypoints not included
        if results.pose_landmarks:
            for i in range(0, 23):
                res = results.pose_landmarks.landmark[i]
                #if res.visibility>=0.5:
                pose_array[i, :] = [res.x, res.y, res.z, res.visibility]
                #else:
                #    pose_array[i, :] = [np.nan,np.nan,np.nan,res.visibility] #!! if low visibility do not consider reliable the output
            pose = pose_array.flatten()
        else:
            pose = np.full(23 * 4, np.nan)
    else:
        pose = np.full(23 * 4, np.nan)

    ## face
    if extract_face:
        face = np.array([[res.x, res.y, res.z] for res in
                         results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.full(468 * 3, np.nan)
    else:
        face = np.full(468 * 3, np.nan)

    ## left hand
    if extract_lh:
        lh = np.array([[res.x, res.y, res.z] for res in
                       results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.full(21 * 3, np.nan)
    else:
        lh = np.full(21 * 3, np.nan)

    ## right hand
    if extract_rh:
        rh = np.array([[res.x, res.y, res.z] for res in
                       results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.full(21 * 3, np.nan)
    else:
        rh = np.full(21 * 3, np.nan)

    return np.concatenate([pose, face, lh, rh]) # []

def draw_styled_landmarks(image, results):  # this does the same thing as previous but with more style
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )

    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )

def HOLISTIC_from_array(filename,complexity=0,show=False,annotations=False,make_video=False,im_size=[]):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    KEYPOINTS = []
    #KEYPOINTS_MP = []
    latency_HOLISTIC = []
    latency_ITERATION = []

    video = np.load(filename)
    # Define the codec and create VideoWriter object

    with mp_holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=complexity,
            smooth_landmarks=True,
            enable_segmentation=True
    ) as holistic:
        frame=0
        all_frames = np.shape(video)[0]
        while frame<all_frames:
            START_all = time.perf_counter_ns()
            image = np.squeeze(video[frame,:,:,:,])
            image = Image.fromarray(image)
            resized_image = image.resize(im_size)
            image = np.array(resized_image)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image.flags.writeable = False
            ### DA VEDERE QUESTO ... devo cambiare l'ordine di RGB?? --> prova holistic normale e vedi cos'è Image
            START = time.perf_counter_ns()
            results = holistic.process(image)
            END = (time.perf_counter_ns() - START)  # nsec

            #fps = round(1 / (END / 10e9)) # frames/sec

            latency_HOLISTIC.append(END)

            if annotations:
                cv2.putText(image, f"lat (landmarks) [ms]:{END / 10e6}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)

            # Draw landmark annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if annotations:
                draw_styled_landmarks(image, results)
            END_all = (time.perf_counter_ns() - START_all)  # nsec to sec

            if annotations:
                cv2.putText(image, f"lat (all) [ms]:{END_all / 10e6}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)

            if show:
                cv2.imshow('MediaPipe Holistic', image)

            KEYPOINTS.append(extract_keypointsHOLISTIC(results).tolist())
            latency_ITERATION.append(END_all)
            frame = frame + 1 ###!!
            if cv2.waitKey(5) == ord('q'):
                break

        cv2.destroyAllWindows()

        output = {
            'keypoints_array': KEYPOINTS,
            'latency_holistic_model': latency_HOLISTIC,
            'latency_end_interation': latency_ITERATION
        }

        return output

def make_HOLISTIC_photo_on_white_canva():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    cap = cv2.VideoCapture(0)
    KEYPOINTS = []
    #KEYPOINTS_MP = []
    latency_HOLISTIC = []
    latency_ITERATION = []
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object

    with mp_holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=True
    ) as holistic:
        while cap.isOpened():
            white_canva = np.zeros((frame_height, frame_width, 3))
            START_all = time.perf_counter_ns()

            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            START = time.perf_counter_ns()
            results = holistic.process(image)
            END = (time.perf_counter_ns() - START)  # nsec

            latency_HOLISTIC.append(END)

            # Draw landmark annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            draw_styled_landmarks(image, results)
            draw_styled_landmarks(white_canva, results)

            END_all = (time.perf_counter_ns() - START_all)  # nsec to sec

            cv2.imshow('MediaPipe Holistic', image)
            cv2.imshow('Holistic on Canva', white_canva)

            KEYPOINTS.append(extract_keypointsHOLISTIC(results).tolist())
            latency_ITERATION.append(END_all)

            if cv2.waitKey(5) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def HOLISTIC(video,complexity,root,show=False,annotations=False,make_video=False):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic

    cap = cv2.VideoCapture(video)
    KEYPOINTS = []
    #KEYPOINTS_MP = []
    latency_HOLISTIC = []
    latency_ITERATION = []
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    if make_video:
        filename_video = os.path.basename(video)[:-4]
        output_video=cv2.VideoWriter(root+'/VIDEO/'+filename_video+'.avi',
                        cv2.VideoWriter_fourcc(*'MJPG'),
                        10, (frame_width, frame_height))

    with mp_holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=complexity,
            smooth_landmarks=True,
            enable_segmentation=True
    ) as holistic:
        while cap.isOpened():
            START_all = time.perf_counter_ns()

            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            START = time.perf_counter_ns()
            results = holistic.process(image)
            END = (time.perf_counter_ns() - START)  # nsec

            #fps = round(1 / (END / 10e9)) # frames/sec

            latency_HOLISTIC.append(END)

            if annotations:
                cv2.putText(image, f"lat (landmarks) [ms]:{END / 10e6}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)

            # Draw landmark annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if annotations:
                draw_styled_landmarks(image, results)
            END_all = (time.perf_counter_ns() - START_all)  # nsec to sec

            if annotations:
                cv2.putText(image, f"lat (all) [ms]:{END_all / 10e6}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)

            if show:
                cv2.imshow('MediaPipe Holistic', image)

            KEYPOINTS.append(extract_keypointsHOLISTIC(results).tolist())
            latency_ITERATION.append(END_all)

            #draw_styled_landmarks(image, results)
            if make_video:
                output_video.write(image)

            if cv2.waitKey(5) == ord('q'):
                break
        if make_video:
            output_video.release()
        cap.release()
        cv2.destroyAllWindows()

        output = {
            'keypoints_array': KEYPOINTS,
            'latency_holistic_model': latency_HOLISTIC,
            'latency_end_interation': latency_ITERATION
        }


        return output

def POSE(video,complexity):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=complexity,
            smooth_landmarks=True,
            enable_segmentation=True
    ) as pose:
        while cap.isOpened():
            START_all = time.perf_counter_ns()

            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            START = time.perf_counter_ns()
            results = pose.process(image)
            END = (time.perf_counter_ns() - START)  # nsec to sec

            fps = round(1 / (END / 10e9))

            cv2.putText(image, f"lat (landmarks) [ms]:{END / 10e6}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

            # Draw landmark annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                      )

            END_all = (time.perf_counter_ns() - START_all)  # nsec to sec

            cv2.putText(image, f"lat (all) [ms]:{END_all / 10e6}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('MediaPipe Holistic', image)
            if cv2.waitKey(5) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        #output = {
        #    'keypoints_array': KEYPOINTS,
        #    'latency_holistic_model': latency_HOLISTIC,
        #    'latency_end_interation': latency_ITERATION
        #}

def HANDS(video,complexity,annotations = False, show = False):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    KEYPOINTS = []

    cap = cv2.VideoCapture(video)
    with mp_hands.Hands(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=complexity
    ) as hands:
        while cap.isOpened():
            START_all = time.perf_counter_ns()

            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            START = time.perf_counter_ns()
            results_hands = hands.process(image)
            END = (time.perf_counter_ns() - START)  # nsec to sec

            fps = round(1 / (END / 10e9))

            cv2.putText(image, f"lat (landmarks) [ms]:{END / 10e6}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

            # Draw landmark annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            H = np.full((2, 21 * 3), np.nan)
            if annotations:
                i = 0
                if results_hands.multi_hand_landmarks:
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2,
                                                                         circle_radius=4),
                                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2,
                                                                         circle_radius=2)
                                                  )
                        if i < 2:
                            H[i, :] = np.array([[res.x, res.y, res.z] for res in
                                                hand_landmarks.landmark]).flatten() if hand_landmarks else np.full(
                                21 * 3, np.nan)
                            i = i + 1

                H = H.reshape(21 * 6)

                KEYPOINTS.append(H)

            END_all = (time.perf_counter_ns() - START_all)  # nsec to sec

            cv2.putText(image, f"lat (all) [ms]:{END_all / 10e6}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

            if show:
                cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        output = {
            'keypoints_array': KEYPOINTS
        }

        return output

def HANDS_from_array(filename,complexity,annotations = False, show = False,im_size = (640,480)):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    KEYPOINTS = []

    video = np.load(filename)
    with mp_hands.Hands(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=complexity
    ) as hands:
        frame=0
        all_frames = np.shape(video)[0]
        while frame<all_frames:
            START_all = time.perf_counter_ns()

            image = np.squeeze(video[frame,:,:,:,])
            image = Image.fromarray(image)
            resized_image = image.resize(im_size)
            image = np.array(resized_image)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            START = time.perf_counter_ns()
            results_hands = hands.process(image)
            END = (time.perf_counter_ns() - START)  # nsec to sec

            cv2.putText(image, f"lat (landmarks) [ms]:{END / 10e6}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

            # Draw landmark annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            H = np.full((2, 21 * 3), np.nan)
            if annotations:
                i = 0
                if results_hands.multi_hand_landmarks:
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2,
                                                                         circle_radius=4),
                                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2,
                                                                         circle_radius=2)
                                                  )
                        if i < 2:
                            H[i, :] = np.array([[res.x, res.y, res.z] for res in
                                                hand_landmarks.landmark]).flatten() if hand_landmarks else np.full(
                                21 * 3, np.nan)
                            i = i + 1

                H = H.reshape(21 * 6)

                KEYPOINTS.append(H)

            END_all = (time.perf_counter_ns() - START_all)  # nsec to sec

            cv2.putText(image, f"lat (all) [ms]:{END_all / 10e6}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)
            frame = frame + 1
            if show:
                cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) == ord('q'):
                break
        cv2.destroyAllWindows()
        output = {
            'keypoints_array': KEYPOINTS
        }

        return output


def POSEnHANDS(video,complexityH=0,complexityP=1,show=False, annotations=False):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video)

    KEYPOINTS = []
    latency_iteration =[]
    latency_model = []
    n=0
    with mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=complexityP,
            smooth_landmarks=True,
            enable_segmentation=True
    ) as pose:

        with mp_hands.Hands(
                static_image_mode=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=complexityH
        ) as hands:
            while cap.isOpened():
                START_all = time.perf_counter_ns()

                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    break

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                START = time.perf_counter_ns()
                results_hands = hands.process(image)
                results_pose = pose.process(image)
                END = (time.perf_counter_ns() - START)  # nsec to sec
                latency_model.append(END)

                #fps = round(1 / (END / 1e9))
                if annotations:
                    cv2.putText(image, f"lat (landmarks) [ms]:{END / 1e6}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 2, cv2.LINE_AA)

                # Draw landmark annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                H = np.full((2,21*3),np.nan)
                if annotations:
                    i = 0
                    if results_hands.multi_hand_landmarks:
                        for hand_landmarks in results_hands.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                      mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2,
                                                                             circle_radius=4),
                                                      mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2,
                                                                             circle_radius=2)
                                                      )
                            if i<2:
                                H[i,:] = np.array([[res.x, res.y, res.z] for res in
                                               hand_landmarks.landmark]).flatten() if hand_landmarks else np.full(21 * 3, np.nan)
                                i = i+1

                    H = H.reshape(21*6)
                    keypoints = extract_keypointsHOLISTIC(results_pose, extract_pose=True, extract_face=False, extract_lh=False,
                                              extract_rh=False)
                    mp_drawing.draw_landmarks(image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                              )
                    keypoints[23*4+468*3:1622] = H

                    KEYPOINTS.append(keypoints.tolist())

                END_all = (time.perf_counter_ns() - START_all)  # nsec to sec
                latency_iteration.append(END_all)
                if annotations:
                    cv2.putText(image, f"lat (all) [ms]:{END_all / 1e6}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 2, cv2.LINE_AA)

                if show:
                    cv2.imshow('MediaPipe POSE HANDS', image)
                if cv2.waitKey(5) == ord('q'):
                    break
                n=n+1
            cap.release()
            cv2.destroyAllWindows()

            output = {
                'latency_model': latency_model,
                'latency_end_interation': latency_iteration,
                'keypoints_array': KEYPOINTS
            }

            return output