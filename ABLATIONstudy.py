import pickle
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from preprocessing_functions import posehands3D_wrtNose
from FILE_MANAGEMENT import list_files_with_extension,create_folder_if_not_exists,find_indices,create_folder_if_not_exists
import os
from tqdm import tqdm
from model_evaluation import ALLMETRICS

def save_results(all_videos,all_labels,names,filepath):
    create_folder_if_not_exists(filepath)

    for i in range(0,len(all_videos)):
        output = {}
        video = all_videos[i]
        label = all_labels[i]

        output['keypoints_array'] = video
        output['label'] = label
        name = names[i]

        with open(filepath+"\\"+name,"wb") as f:
            pickle.dump(output,f)

def ablate(allvideos,all_labels,names,TAG,bestmodel,directory,pose=True,handsXY=True,handsZ=True,poseface = True, posehands=True, posearms = True):
    #inputer = bestmodel.named_steps['imputer']
    #scaler = bestmodel.named_steps['scaler']
    newallvideos = []
    for video in allvideos:
        #video = inputer.transform(video)
        #video = scaler.transform(video)
        newvideo = video.copy()
        if not(poseface):
            newvideo[:,0:10*2]=np.nan

        if not(posearms):
            newvideo[:,10*2:16*2]=np.nan

        if not(posehands):
            newvideo[:,16*2:22*2]=np.nan

        if not(pose):
            newvideo[:,0:22*2]=np.nan

        if not(handsXY):
            newvideo[:,22*2:22*2+21*2+21*2]=np.nan

        if not(handsZ):
            newvideo[:,22*2+21*2+21*2:22*2+21*2+21*2+21+21]=np.nan

        newallvideos.append(newvideo)
    create_folder_if_not_exists(directory)
    save_results(newallvideos, all_labels, names, directory+"\\"+TAG)

    return None

def readvideos(allfiles,prep_func):

    all_videos = []
    all_labels = []
    names = []

    for testfile in allfiles:
        with open(testfile, 'rb') as file:
        # Load the object from the pickle file
            output = pickle.load(file)
        Xi = np.squeeze((np.array(output['keypoints_array'])))
        yi = output['label']
        Xi = np.squeeze(prep_func(Xi))
        namei = os.path.basename(testfile)

        all_videos.append(Xi)
        all_labels.append(yi)
        names.append(namei)

    return all_videos,all_labels,names

def MAKE_and_SAVE_ABLATIONs():

    arrayTAG = ["H0", "H1", "H2", "P0HA0", "P0HA1", "P1HA0", "P1HA1", "P2HA0", "P2HA1"]

    with open('bestmodel.pkl', 'rb') as file:
        best_model = pickle.load(file)

    for tag_i in range(0,len(arrayTAG)):
        TAG = arrayTAG[tag_i]
        directoryEXPERIMENTALDAT = "experimental_dat\\"+TAG+"\\640x480"
        TESTFILES = list_files_with_extension(directoryEXPERIMENTALDAT,'.pkl')
        directory_where_to_save = "ablated_videos"+TAG

        ##### ABLATING THE FEATURES ##### and ##### SAVING THE ABLATION #####
        allvideos,all_labels,names  = readvideos(TESTFILES,posehands3D_wrtNose)

        print('ablation ...\n\n')
        ablate(allvideos,all_labels,names,"noablation",best_model,directory_where_to_save)

        print('---> no pose')
        ablate(allvideos,all_labels,names,"nopose",best_model,directory_where_to_save, pose=False)

        print('---> no hands')
        ablate(allvideos,all_labels,names, "nohands",best_model,directory_where_to_save,handsXY=False, handsZ=False)

        print('---> no hands XY')
        ablate(allvideos,all_labels,names, "nohandsXY",best_model,directory_where_to_save,handsXY=False)

        print('---> no hands Z')
        ablate(allvideos,all_labels,names,"nohandsZ",best_model,directory_where_to_save,handsZ=False)

        print('---> no poseface')
        ablate(allvideos,all_labels,names, "noposeface",best_model,directory_where_to_save,poseface=False)

        print('---> no posearms')
        ablate(allvideos,all_labels,names,"noposearms",best_model,directory_where_to_save,posearms=False)

        print('---> no posehands')
        ablate(allvideos,all_labels,names,"noposehands",best_model,directory_where_to_save,posehands=False)

def TEST_and_SAVE_RESULTS():
    with open('bestmodel.pkl', 'rb') as file:
        best_model = pickle.load(file)

    classifier = best_model


    arrayTAG = ["H0", "H1", "H2", "P0HA0", "P0HA1", "P1HA0", "P1HA1", "P2HA0", "P2HA1"]

    for tag_i in range(0,len(arrayTAG)):
        TAG = arrayTAG[tag_i]
        allablations = os.listdir('ablated_videos'+TAG)
        for i in range(0,len(allablations)):
            ablationfolder = 'ablated_videos'+TAG+'\\'+allablations[i]
            ablationfiles = list_files_with_extension(ablationfolder,'.pkl')
            allnames = []
            allpred = []
            alltrue = []
            for file in ablationfiles:
                with open(file,'rb') as f:
                    output = pickle.load(f)

                Xi = output['keypoints_array']
                namei = os.path.basename(file)[:-4]
                yTRUE_array = np.ones((np.shape(Xi)[0])) * output['label']

                # predict
                yhat_i = classifier.predict(Xi)
                allpred.append(yhat_i.tolist())
                alltrue.append(yTRUE_array.tolist())
                allnames.append(namei)

            path = "RESULTS\\EXPERIMENTAL_DAT\\ablation\\"+ablationfolder
            create_folder_if_not_exists(path)
            results = {'yPREDtot':allpred,'yTRUEtot':alltrue,'names':allnames}
            results = ALLMETRICS(results,times=False, steps=True)
            with open(path+"\\results.pkl", 'wb') as f:
                pickle.dump(results,f)

#MAKE_and_SAVE_ABLATIONs()
#TEST_and_SAVE_RESULTS()

################### excel results
def give_me_orientation(files):
    def find_all_underscores(string):
        """
        Find the positions of all underscore characters "_" in a string.

        Args:
        string (str): The string to search in.

        Returns:
        list: A list containing the positions of all underscore characters.
        """
        positions = []
        for i in range(len(string)):
            if string[i] == '_':
                positions.append(i)
        return positions
    orientations = np.array(["0","45R","45L"])

    files0 = []
    files45R = []
    files45L = []
    import os
    for file in files:
        filename = os.path.basename(file)
        underscore_pos = find_all_underscores(filename)
        orientation = filename[underscore_pos[0]+1:underscore_pos[1]]

        if orientation == orientations[0]:
            files0.append(file)
        if orientation == orientations[1]:
            files45R.append(file)
        if orientation == orientations[2]:
            files45L.append(file)

    return files0,files45L,files45R

def give_me_distances(files):
    def find_all_underscores(string):
        """
        Find the positions of all underscore characters "_" in a string.

        Args:
        string (str): The string to search in.

        Returns:
        list: A list containing the positions of all underscore characters.
        """
        positions = []
        for i in range(len(string)):
            if string[i] == '_':
                positions.append(i)
        return positions
    distances = np.array(["50cm","70cm","75cm","100cm","130cm"])

    files50cm = []
    files70cm = []
    files100cm = []
    files130cm = []

    import os
    for file in files:
        filename = os.path.basename(file)
        underscore_pos = find_all_underscores(filename)
        distance = filename[underscore_pos[1]+1:underscore_pos[2]]

        if distance == distances[0]:
            files50cm.append(file)
        if distance == distances[1] or distance == distances[2]:
            files70cm.append(file)
        if distance == distances[3]:
            files100cm.append(file)
        if distance == distances[4]:
            files130cm.append(file)

    return files50cm, files70cm, files100cm, files130cm

arrayTAG = ["H0", "H1", "H2", "P0HA0", "P0HA1", "P1HA0", "P1HA1", "P2HA0", "P2HA1"]

for tag_i in range(8,len(arrayTAG)):
    TAG = arrayTAG[tag_i]
    print(TAG)
    directory = 'RESULTS\\EXPERIMENTAL_DAT\\ablation\\ablated_videos' + TAG
    array_ablationTAG = ["noablation","nohands","nohandsXY","nohandsZ","nopose","noposearms","noposeface","noposehands"]

    DICTS = []
    DICTS0 = []
    DICTS45R = []
    DICTS45L = []
    DICTS50cm = []
    DICTS70cm = []
    DICTS100cm = []
    DICTS130cm = []

    CMtags = ["allframe","perframe","step10","step15","step30"]
    orientations = ["0", "45R", "45L"]
    distances = ["50cm", "70cm", "100cm", "130cm"]
    for CMtag in CMtags:
        exec("CM_ablation"+CMtag+" = np.zeros((len(array_ablationTAG),4,4))")
        exec("CM_orientations_ablation"+CMtag+" = np.zeros((len(orientations),len(array_ablationTAG),4,4))")
        exec("CM_distances_ablation"+CMtag+" = np.zeros((len(distances),len(array_ablationTAG),4,4))")


    for TAG_i in range(0,len(array_ablationTAG)):
        ablationTAG = array_ablationTAG[TAG_i]
        with open(directory+'\\'+ablationTAG+'\\results.pkl', 'rb') as file:
            # Load the object from the pickle file
            results = pickle.load(file)

        all_names = results['names']
        yPREDtot = results['yPREDtot']
        yTRUEtot = results['yTRUEtot']

        #results = ALLMETRICS(results,times=False)
        entries_to_exclude = ['names', 'yPREDtot', 'yTRUEtot']
        substring = 'CV'
        results_filt = {key: value for key, value in results.items() if ((key not in entries_to_exclude) and (substring not in key))}
        results_filt['TAG'] = ablationTAG
        DICTS.append(results_filt)

        all_names_50cm, all_names_70cm, all_names_100cm, all_names_130cm = give_me_distances(all_names)
        all_names_0, all_names_45L, all_names_45R = give_me_orientation(all_names)


        for CMtag in CMtags:
            string_exec = "CM_ablation"+CMtag+"[TAG_i, :, :] = results['CV_'+CMtag]"
            exec(string_exec)

        for distance_i in range(0,len(distances)):
            distance = distances[distance_i]

            exec("idx0 = find_indices(all_names, all_names_"+distance+")")

            yPREDtot_0 = []
            yTRUEtot_0 = []
            for i in range(0, len(yPREDtot)):
                if i in idx0:
                    yPREDtot_0.append(yPREDtot[i])
                    yTRUEtot_0.append(yTRUEtot[i])

            results_0 = {'yPREDtot': yPREDtot_0, 'yTRUEtot': yTRUEtot_0}
            results_0 = ALLMETRICS(results_0,times=False,steps=True)
            entries_to_exclude = ['names', 'yPREDtot', 'yTRUEtot']
            substring = 'CV'

            results_filt = {key: value for key, value in results_0.items() if
                            ((key not in entries_to_exclude) and (substring not in key))}
            results_filt['TAG'] = ablationTAG

            exec("DICTS"+distance+".append(results_filt)")

            for CMtag in CMtags:
                string_exec = "CM_distances_ablation" + CMtag + "[distance_i,TAG_i,:,:] = results_0['CV_'+CMtag]"
                exec(string_exec)

        for orientation_i in range(0,len(orientations)):
            orientation = orientations[orientation_i]

            exec("idx0 = find_indices(all_names, all_names_"+orientation+")")

            yPREDtot_0 = []
            yTRUEtot_0 = []
            for i in range(0, len(yPREDtot)):
                if i in idx0:
                    yPREDtot_0.append(yPREDtot[i])
                    yTRUEtot_0.append(yTRUEtot[i])

            results_0 = {'yPREDtot': yPREDtot_0, 'yTRUEtot': yTRUEtot_0}
            results_0 = ALLMETRICS(results_0,times=False,steps=True)
            entries_to_exclude = ['names', 'yPREDtot', 'yTRUEtot']
            substring = 'CV'

            results_filt = {key: value for key, value in results_0.items() if
                            ((key not in entries_to_exclude) and (substring not in key))}
            results_filt['TAG'] = ablationTAG

            exec("DICTS"+orientation+".append(results_filt)")

            for CMtag in CMtags:
                string_exec = "CM_orientations_ablation" + CMtag + "[orientation_i,TAG_i,:,:] = results_0['CV_'+CMtag]"
                exec(string_exec)

    import pandas as pd

    dfs = [pd.DataFrame(d, index=[0]) for d in DICTS]
    df_concatenated = pd.concat(dfs, ignore_index=True)
    df_concatenated.to_excel(directory+'\\allresults_experimental_dat.xlsx')

    for orientation in orientations:
        exec("dfs = [pd.DataFrame(d, index=[0]) for d in DICTS"+orientation+"]")
        df_concatenated = pd.concat(dfs, ignore_index=True)
        df_concatenated.to_excel(directory+'\\allresults_experimental_dat_orientation'+orientation+'.xlsx')

    for distance in distances:
        exec("dfs = [pd.DataFrame(d, index=[0]) for d in DICTS"+distance+"]")
        df_concatenated = pd.concat(dfs, ignore_index=True)
        df_concatenated.to_excel(directory+'\\allresults_experimental_dat_distance'+distance+'.xlsx')

    for CMtag in CMtags:
        string_exec = "cms = CM_orientations_ablation" + CMtag
        exec(string_exec)

        newCM = np.zeros((4*len(array_ablationTAG),4*len(orientations)))

        for i in range(0, np.shape(cms)[0]): # orientation
            for j in range(0, np.shape(cms)[1]): # complexity
                cm = cms[i, j, :, :]

                #orientation
                col_start = 4*i
                col_fin = col_start + 4

                # complexity
                row_start = 4*j
                row_fin = row_start + 4

                newCM[row_start:row_fin,col_start:col_fin] = cm

        df_cms = pd.DataFrame(newCM)
        df_cms.to_excel(directory+'\\cms_orientations_ablation'+CMtag+'.xlsx')

    for CMtag in CMtags:
        string_exec = "cms = CM_distances_ablation" + CMtag
        exec(string_exec)

        newCM = np.zeros((4*len(array_ablationTAG),4*len(distances)))

        for i in range(0, np.shape(cms)[0]): # distances
            for j in range(0, np.shape(cms)[1]): # complexity
                cm = cms[i, j, :, :]

                #orientation
                col_start = 4*i
                col_fin = col_start + 4

                # complexity
                row_start = 4*j
                row_fin = row_start + 4

                newCM[row_start:row_fin,col_start:col_fin] = cm

        df_cms = pd.DataFrame(newCM)
        df_cms.to_excel(directory+'\\cms_distances_ablation'+CMtag+'.xlsx')

    for CMtag in CMtags:
        string_exec = "cms = CM_ablation" + CMtag
        exec(string_exec)

        newCM = np.zeros((4*len(array_ablationTAG),4))

        for j in range(0, np.shape(cms)[0]):
            cm = cms[j, :, :]

            row_start = 4 * j
            row_fin = row_start + 4

            newCM[row_start:row_fin, :] = cm

        df_cms = pd.DataFrame(newCM)
        df_cms.to_excel(directory+'\\cms'+CMtag+'.xlsx')
