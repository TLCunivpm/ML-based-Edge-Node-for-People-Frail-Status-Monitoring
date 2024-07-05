###ESXPLORATORY DATA ANALYSIS
import numpy as np
import matplotlib.animation as animation
from itertools import combinations
import sweetviz as sv
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import cv2
import seaborn as sns



def visualize_face_simple(FACE,ax,alpha=0.5):
    ax.plot(FACE[:,0],FACE[:,1],color = 'purple',marker='.', markersize = 1,alpha=alpha)

def countour_GEN(data,XTAG,YTAG,meanFACE,name):
    fig, ax = plt.subplots()

    sns.kdeplot(
        data=data, x=XTAG, y=YTAG, fill=True,
    )
    visualize_face_simple(meanFACE, ax, alpha=0.5)
    ax.set_xlim(2, -2)
    ax.set_ylim(2, -2)
    fig.savefig(name +'.jpg')
    plt.close('all')
def visualize_face(FACE,ax,markersize,lw = 2, alpha=0.5):
    silhouette= FACE[[10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,10],0:2]

    lipsUpperOuter= FACE[[61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],0:2]
    lipsLowerOuter= FACE[[146, 91, 181, 84, 17, 314, 405, 321, 375, 291],0:2]

    rightEyeUpper0= FACE[[246, 161, 160, 159, 158, 157, 173],0:2]
    rightEyeLower0= FACE[[33, 7, 163, 144, 145, 153, 154, 155, 133],0:2]

    leftEyeUpper0= FACE[[466, 388, 387, 386, 385, 384, 398],0:2]
    leftEyeLower0= FACE[[263, 249, 390, 373, 374, 380, 381, 382, 362],0:2]

    nose = FACE[[1,2,98,327],0:2]

    ax.plot(silhouette[:,0],silhouette[:,1],color = 'purple',marker='.', markersize = markersize,alpha=alpha,linewidth=lw,label='FACE')
    ax.plot(lipsUpperOuter[:,0],lipsUpperOuter[:,1],color = 'purple',marker='.', markersize = markersize,alpha=alpha,linewidth=lw)
    ax.plot(lipsLowerOuter[:,0],lipsLowerOuter[:,1],marker='.',markersize = markersize,color = 'purple',alpha=alpha,linewidth=lw)
    ax.plot(rightEyeUpper0[:,0],rightEyeUpper0[:,1],marker='.',markersize = markersize,color = 'purple',alpha=alpha,linewidth=lw)
    ax.plot(rightEyeLower0[:,0],rightEyeLower0[:,1],marker='.',markersize = markersize,color = 'purple',alpha=alpha,linewidth=lw)
    ax.plot(leftEyeUpper0[:,0],leftEyeUpper0[:,1],marker='.',markersize = markersize,color = 'purple',alpha=alpha,linewidth=lw)
    ax.plot(leftEyeLower0[:,0],leftEyeLower0[:,1],marker='.',markersize = markersize,color = 'purple',alpha=alpha,linewidth=lw)
    ax.plot(nose[:,0],nose[:,1],marker='.',markersize = markersize,color = 'purple',alpha=alpha,linewidth=lw)
def visualize_hand_data(HAND,ax,markersize,lw = 2,alpha=0.5):
    connect_thumb = HAND[0:5,0:2]
    connect_index = HAND[5:9,0:2]
    connect_midfing = HAND[9:13, 0:2]
    connect_ringfing = HAND[13:17, 0:2]
    connect_pinky = HAND[17:21, 0:2]
    connect_palm = HAND[[0,5,9,13,17],0:2]

    ax.plot(connect_thumb[:,0],connect_thumb[:,1],color = 'red',marker='.', markersize = markersize ,alpha=alpha,linewidth=lw,label='HAND')
    ax.plot(connect_index[:,0],connect_index[:,1],color = 'red',marker='.', markersize = markersize ,alpha=alpha,linewidth=lw)
    ax.plot(connect_midfing[:,0],connect_midfing[:,1],marker='.',markersize = markersize,color = 'red',alpha=alpha,linewidth=lw)
    ax.plot(connect_ringfing[:,0],connect_ringfing[:,1],marker='.',markersize = markersize,color = 'red',alpha=alpha,linewidth=lw)
    ax.plot(connect_pinky[:,0],connect_pinky[:,1],marker='.',markersize = markersize,color = 'red',alpha=alpha,linewidth=lw)
    ax.plot(connect_palm[:,0],connect_palm[:,1],marker='.',markersize = markersize,color = 'red',alpha=alpha,linewidth=lw)

def visualize_pose_data(keypoints,ax,markersize,lw = 2,alpha=0.5):
    # SINGLE FRAME SNAPSHOT VISUALISATION
    # faceXY,lt_sh,rt_sh,lt_elb,rt_elb,lt_wirst,rt_wirst,...
    connect_face_lt = keypoints[[0,1,2,3,7],0:2]
    connect_face_rt = keypoints[[0,4,5,6,8],0:2]
    connect_mouth = keypoints[[9,10],0:2]
    connect_rt_arm = keypoints[[12,14,16,18,20,22,16],0:2]
    connect_lt_arm = keypoints[[11,13,15,17,19,21,15],0:2]
    connect_shoulder = keypoints[[11,12],0:2]

    #plt.scatter(connect_lt_arm[:,0],connect_lt_arm[:,1],color = 'orange')
    #plt.scatter(connect_rt_arm[:,0],connect_rt_arm[:,1],color = 'blue')

    ax.plot(connect_lt_arm[:,0],connect_lt_arm[:,1],color = 'blue',marker='.', markersize = markersize,alpha=alpha,linewidth=lw,label = 'POSE')
    ax.plot(connect_rt_arm[:,0],connect_rt_arm[:,1],color = 'blue',marker='.', markersize = markersize,alpha=alpha,linewidth=lw)
    ax.plot(connect_shoulder[:,0],connect_shoulder[:,1],color = 'blue',alpha=alpha)
    ax.plot(connect_face_lt[:,0],connect_face_lt[:,1],marker='.',markersize = markersize,color = 'blue',alpha=alpha,linewidth=lw)
    ax.plot(connect_face_rt[:,0],connect_face_rt[:,1],marker='.',markersize = markersize,color = 'blue',alpha=alpha,linewidth=lw)
    ax.plot(connect_mouth[:,0],connect_mouth[:,1],marker='.',markersize = markersize,color = 'blue',alpha=alpha,linewidth=lw)

    #plt.show()

def make_time_evolution_plots(filenames,TAG):
    actions = np.array(['bt_GOOD', 'fixinghair_GOOD', 'no_action_GOOD', 'wsf_GOOD'])

    for i in tqdm(range(len(filenames)), desc='Progress', unit='iter'):
        NAME = filenames[i]
        filename = os.path.basename(NAME)
        with open(NAME, 'rb') as file:
            # Load the object from the pickle file
            output = pickle.load(file)

        keypoints = np.squeeze(output['keypoints_array'])
        noseXY = np.squeeze(keypoints[:,0,:])
        noseX = noseXY[:,0]
        noseY = noseXY[:,1]
        handRT_x = np.mean(keypoints[:,[16,18,20,22],0],axis=1)-noseX
        handLT_x = np.mean(keypoints[:,[15,17,19,21],0],axis=1)- noseX

        handRT_y = np.mean(keypoints[:,[16,18,20,22],1],axis=1)- noseY
        handLT_y = np.mean(keypoints[:,[15,17,19,21],1],axis=1)- noseY

        plt.subplot(2,2,1)
        plt.plot(handRT_x)
        plt.title('handRT_x')
        plt.subplot(2, 2, 2)
        plt.plot(handRT_y)
        plt.title('handRT_y')
        plt.subplot(2, 2, 3)
        plt.plot(handLT_x)
        plt.title('handLT_x')
        plt.subplot(2, 2, 4)
        plt.plot(handLT_y)
        plt.title('handLT_y')

        root_path = "C:/Users/Utente/Desktop/Scripts_VIOLA/TIMEPLOTS/onlypose"+TAG
        directory_path= root_path +"/" + actions[int(output['label'])]

        if not(os.path.exists(directory_path)):
            os.makedirs(root_path)

        if os.path.exists(directory_path):
            plt.savefig(directory_path + "/" +str(filename[:-4])+".png")
        else:
            os.makedirs(directory_path)
            plt.savefig(directory_path + "/" +str(filename[:-4])+ ".png")

        plt.close('all')

def HEATMAP_gen(newX,TAG):
    fig, ax = plt.subplots()
    for i in tqdm(range(np.shape(newX)[0]), desc="Processing", unit="iteration"):
        keypoints = np.squeeze(newX[i,:,:])
        pose = keypoints[0:23,:]

        lthand = keypoints[23:23+21,:]
        rthand = keypoints[23+21:23+21+21,:]

        face = keypoints[23+21+21:-1,:]

        visualize_pose_data(pose, ax,alpha=0.01)

        visualize_hand_data(lthand, ax,alpha=0.01)
        visualize_hand_data(rthand, ax,alpha=0.01)

        visualize_face_simple(face, ax,alpha=0.01)

    # Set limits for x-axis and y-axis
    ax.set_xlim(2, -2)
    ax.set_ylim(2, -2)
    ax.set_title(f'HEATMAP PLOT ACTION')

    # Save animation as video
    root_path = "C:/Users/Utente/Desktop/Scripts_VIOLA/ANIMATIONS/pose_hand_faceHEATMAP" + TAG
    if not (os.path.exists(root_path)):
        os.makedirs(root_path)

    directory_path = root_path

    if os.path.exists(directory_path):
        fig.savefig(directory_path + "/" + "HEATMAP" + ".png")
    else:
        os.makedirs(directory_path)
        fig.savefig(directory_path + "/" + "HEATMAP" + ".png")

    plt.close('all')

def make_animation_plots_single_video_PREP(NAME,TAG,preprocessing,labelsPRED=[],labelsTRUE=[]):

    actions = np.array(['bt_GOOD', 'fixinghair_GOOD', 'no_action_GOOD', 'wsf_GOOD'])

    filename = os.path.basename(NAME)
    with open(NAME, 'rb') as file:
        # Load the object from the pickle file
        output = pickle.load(file)
    fig, ax = plt.subplots()
    keypoints = np.array(output['keypoints_array'])
    keypoints = preprocessing(keypoints)
    pose = np.squeeze(keypoints[:,:,0:23,:])

    lthand = np.squeeze(keypoints[:,:,23:23+21,:])
    rthand = np.squeeze(keypoints[:,:,23+21:23+21+21,:])

    face = np.squeeze(keypoints[:,:,23+21+21:-1,:])

    keypointsX = keypoints[:,:,:,0].reshape(-1,1)
    keypointsY = keypoints[:,:,:,1].reshape(-1,1)

    # keypointsX = pose[:, :, 0].reshape(-1, 1)
    # keypointsY = pose[:, :, 1].reshape(-1, 1)
    # LIMITS = np.array([np.nanmax(keypointsX), np.nanmin(keypointsX), np.nanmax(keypointsY), np.nanmin(keypointsX)])
    def update(frame):
        if frame < len(keypoints[:, 0]):
            ax.clear()
            if len(labelsPRED) > 0:
                plt.text(1.9, -1.9, "predicted_label:" + str(labelsPRED[frame]), fontsize=12, color='black')
                plt.text(1.9, -1.7, "actual_label:" + str(labelsTRUE[frame]), fontsize=12, color='black')

            visualize_pose_data(pose[frame, :, :], ax)

            visualize_hand_data(lthand[frame, :, :], ax)
            visualize_hand_data(rthand[frame, :, :], ax)

            visualize_face_simple(face[frame, :, :], ax)
            # Set limits for x-axis and y-axis
            ax.set_xlim(2, -2)
            ax.set_ylim(2, -2)
            ax.set_title(f'Plot (Frame {frame})')

    # Create animation
    ani = animation.FuncAnimation(fig=fig, func=update, frames=np.shape(pose)[0], interval=1000)

    # Set up writer for saving animation as video
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

    # Save animation as video
    root_path = "C:/Users/Utente/Desktop/Scripts_VIOLA/ANIMATIONS/pose_hand_face" + TAG
    if not (os.path.exists(root_path)):
        os.makedirs(root_path)

    directory_path = root_path + "/" + actions[int(output['label'])]

    if os.path.exists(directory_path):
        ani.save(directory_path + "/" + str(filename[:-4]) + ".mp4", writer=writer)
    else:
        os.makedirs(directory_path)
        ani.save(directory_path + "/" + str(filename[:-4]) + ".mp4", writer=writer)

    plt.close('all')


def make_animation_plots_single_video(NAME,TAG,labelsPRED=[],labelsTRUE=[]):
    actions = np.array(['bt_GOOD', 'fixinghair_GOOD', 'no_action_GOOD', 'wsf_GOOD'])

    filename = os.path.basename(NAME)
    with open(NAME, 'rb') as file:
        # Load the object from the pickle file
        output = pickle.load(file)
    fig, ax = plt.subplots()
    keypoints = np.array(output['keypoints_array'])
    pose = keypoints[:, 0:23 * 4].reshape(-1, 23, 4)
    pose = pose[:, :, 0:2]

    face = keypoints[:, 23 * 4:23 * 4 + 468 * 3].reshape(-1, 468, 3)
    face = face[:, :, 0:2]

    lthand = keypoints[:, 23 * 4 + 468 * 3:23 * 4 + 468 * 3 + 21 * 3].reshape(-1, 21, 3)
    lthand = lthand[:, :, 0:2]

    rthand = keypoints[:, 23 * 4 + 468 * 3 + 21 * 3:1622].reshape(-1, 21, 3)
    rthand = rthand[:, :, 0:2]

    # keypointsX = pose[:, :, 0].reshape(-1, 1)
    # keypointsY = pose[:, :, 1].reshape(-1, 1)
    # LIMITS = np.array([np.nanmax(keypointsX), np.nanmin(keypointsX), np.nanmax(keypointsY), np.nanmin(keypointsX)])
    def update(frame):
        if frame < len(keypoints[:, 0]):
            ax.clear()
            if len(labelsPRED) > 0:
                plt.text(0.9, 0.9, "predicted_label:" + str(labelsPRED[frame]), fontsize=12, color='black')
                plt.text(0.9, 0.7, "actual_label:" + str(labelsTRUE[frame]), fontsize=12, color='black')

            visualize_pose_data(pose[frame, :, :], ax)

            visualize_hand_data(lthand[frame, :, :], ax)
            visualize_hand_data(rthand[frame, :, :], ax)

            visualize_face(face[frame, :, :], ax)
            # Set limits for x-axis and y-axis
            ax.set_xlim(1, 0)
            ax.set_ylim(1, 0)
            ax.set_title(f'Plot (Frame {frame})')

    # Create animation
    ani = animation.FuncAnimation(fig=fig, func=update, frames=np.shape(pose)[0], interval=1000)

    # Set up writer for saving animation as video
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

    # Save animation as video
    root_path = "C:/Users/Utente/Desktop/Scripts_VIOLA/ANIMATIONS/pose_hand_face" + TAG
    if not (os.path.exists(root_path)):
        os.makedirs(root_path)

    directory_path = root_path + "/" + actions[int(output['label'])]

    if os.path.exists(directory_path):
        ani.save(directory_path + "/" + str(filename[:-4]) + ".mp4", writer=writer)
    else:
        os.makedirs(directory_path)
        ani.save(directory_path + "/" + str(filename[:-4]) + ".mp4", writer=writer)

    plt.close('all')

def SWEETVIZ_ANALYSYS(df):
    report = sv.analyze(df)

    # Show the report (can also save it to a file using report.show_html('report.html'))
    report.show_html("sweetviz_report.html")

    ## per class
    actions = np.array(['bt_GOOD', 'fixinghair_GOOD', 'no_action_GOOD', 'wsf_GOOD'])
    for i in range(0,4):
        new_df_class = df[df['label']==i]
        my_report = sv.analyze((new_df_class,actions[i]), pairwise_analysis='on')

        my_report.show_html("SWEETVIZREPORT" + actions[i] + ".html")


#def add_annotations_to_video(video,keypoints_array,labels):



def make_animation_plots(filenames,TAG,labels=[]):

    for i in tqdm(range(len(filenames)), desc='Progress', unit='iter'):
        NAME = filenames[i]
        make_animation_plots_single_video(NAME,TAG,labels)
