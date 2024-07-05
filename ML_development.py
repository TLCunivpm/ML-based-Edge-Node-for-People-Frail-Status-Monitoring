import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
#from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from FILE_MANAGEMENT import find_indices

'''
def create_consecutive_frames_dat(X,consecutive_frames,shift):
    X_i = X_i[0:tot_frames - (tot_frames % nframes), :]
    X_i = X_i.reshape(-1, nframes, feature_size)
    
    X_i_shifted = X_i[shift:tot_frames - shift - ((tot_frames-shift) % nframes), :]

    return X_new
    '''

def loadDATA(files,nframes,prep_func):

    X = []
    y = []
    allfile = []
    for i in range(0,len(files)):
        NAME = files[i]
        with open(NAME, 'rb') as file:
            # Load the object from the pickle file
            output = pickle.load(file)
            X_i = prep_func(np.array(output['keypoints_array']))

            tot_frames = np.shape(X_i)[0]
            feature_size = np.shape(X_i)[1]
            #X_i = X_i[0:tot_frames-(tot_frames % nframes),:]
            #X_i = X_i.reshape(-1,nframes,feature_size) # array of the consecutive frames
            X_i_frames = []
            if tot_frames>=nframes:
                if nframes!=1:
                    for j in range(0,tot_frames-nframes+1,round(nframes/2)):
                        X_i_frames.append(X_i[j:j+nframes,:].reshape(1,nframes,feature_size))
                    X_i_frames = np.concatenate(X_i_frames,axis=0)
                    y_i = output['label'] * np.ones((np.shape(X_i_frames)[0],1)) # output is saved for each video as a number

                    filename = os.path.basename(NAME)
                    file_i = np.tile(filename[:-4],(np.shape(X_i_frames)[0],1))

                    X.append(X_i_frames)
                    y.append(y_i)
                    allfile.append(file_i)
                else:
                    X.append(X_i)
                    y_i = output['label'] * np.ones((np.shape(X_i)[0], 1))
                    y.append(y_i)
                    filename = os.path.basename(NAME)
                    allfile.append(filename)

    X = np.vstack(X)
    y = np.vstack(y)
    allfile = np.vstack(allfile)

    return X,y,allfile

def loadDATAraw(files,nframes):

    X = []
    y = []
    allfile = []
    for i in range(0,len(files)):
        NAME = files[i]
        with open(NAME, 'rb') as file:
            # Load the object from the pickle file
            output = pickle.load(file)
            X_i = np.array(output['keypoints_array'])

            tot_frames = np.shape(X_i)[0]
            feature_size = np.shape(X_i)[1]
            #X_i = X_i[0:tot_frames-(tot_frames % nframes),:]
            #X_i = X_i.reshape(-1,nframes,feature_size) # array of the consecutive frames
            X_i_frames = []
            if tot_frames>=nframes:
                if nframes!=1:
                    for j in range(0,tot_frames-nframes+1,10):
                        X_i_frames.append(X_i[j:j+nframes,:].reshape(1,nframes,feature_size))
                    X_i_frames = np.concatenate(X_i_frames,axis=0)
                    y_i = output['label'] * np.ones((np.shape(X_i_frames)[0],1)) # output is saved for each video as a number

                    filename = os.path.basename(NAME)
                    file_i = np.tile(filename[:-4],(np.shape(X_i_frames)[0],1))

                    X.append(X_i_frames)
                    y.append(y_i)
                    allfile.append(file_i)
                else:
                    X.append(X_i)
                    y_i = output['label'] * np.ones((np.shape(X_i)[0], 1))
                    y.append(y_i)
                    filename = os.path.basename(NAME)
                    allfile.append(filename)

    X = np.vstack(X)
    y = np.vstack(y)
    allfile = np.vstack(allfile)

    return X,y,allfile

def offlineLAZYFIT(train_filenames,val_filenames,consecutive_frames,preprocess,tag):
    if strcmp(cv_type,'holdout'):
        train_filenames, test_val_filenames = train_test_split(
            all_files, test_size=0.2, random_state=42
        )

        val_filenames, test_filenames = train_test_split(
            test_val_filenames, test_val_labels, test_size=0.5, random_state=42
        )
    print('LOADING DATA ... ')
    X_train,y_train,_ = loadDATAraw(train_filenames, consecutive_frames)
    X_test,y_test,_ = loadDATAraw(val_filenames, consecutive_frames)
    print('preprocessing DATA ... ')

    X_train=preprocess(X_train)
    X_test=preprocess(X_test)

    X_train = X_train.reshape(np.shape(X_train)[0],-1)
    X_test = X_test.reshape(np.shape(X_test)[0],-1)

    print('LAZY CLASSIFICATION ...')
    clf = LazyClassifier(verbose=1,ignore_warnings=True,custom_metric=None)
    scores,_ = clf.fit(X_train,X_test,y_train,y_test)
    fitted_models = clf.provide_models(X_train,X_test,y_train,y_test)

    with open('SCORES\LAZY_SCORES_'+tag+'.pkl', 'wb') as f:
        pickle.dump(scores, f)

    with open('MODELS\LAZY_FITTED_MODELS_'+tag+'.pkl', 'wb') as f:
        pickle.dump(fitted_models, f)

    return fitted_models, scores

def offlineFITnVALIDATION(train_filenames,val_filenames,models,consecutive_frames,preprocess,TAG,input_scal=False,input_zero=False,input_mean=False):
    '''
    if strcmp(cv_type,'holdout'):
        train_filenames, test_val_filenames = train_test_split(
            all_files, test_size=0.2, random_state=42
        )

        val_filenames, test_filenames = train_test_split(
            test_val_filenames, test_val_labels, test_size=0.5, random_state=42
        )
'''
    XTRAIN,yTRAIN,_ = loadDATAraw(train_filenames, consecutive_frames)
    XVAL,yVAL,_ = loadDATAraw(val_filenames, consecutive_frames)
    XTRAIN = preprocess(XTRAIN,consecutive_frames)
    XVAL = preprocess(XVAL,consecutive_frames)

    XTRAIN = XTRAIN.reshape(np.shape(XTRAIN)[0],-1)
    XVAL = XVAL.reshape(np.shape(XVAL)[0],-1)
    print(np.shape(XTRAIN))

    # Fit and evaluate each model
    accuracies = []
    f1scores = []
    precisions = []
    recalls = []
    CMs = []
    model_name = []

    #PREPROCESSING
    if input_scal:
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
            # MEAN SUBTRACTION AND SCALING
        )
    if input_mean:
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="mean"))]
            # MEAN SUBTRACTION AND SCALING
        )
    if input_zero:
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="constant",fill_value=0))]
            # MEAN SUBTRACTION AND SCALING
        )

    if isinstance(XTRAIN, np.ndarray):
        XTRAIN = pd.DataFrame(XTRAIN)
        XVAL = pd.DataFrame(XVAL)

    numeric_features = XTRAIN.select_dtypes(include=[np.number]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),  # name , transformer, data
        ]
    )

    fitted_models = []
    #labels = [0,1,2,3] # 0: 'bt_GOOD', 1: 'fixinghair_GOOD', 2: 'no_action_GOOD', 3: 'wsf_GOOD'
    for name,model in models:
        print('FITTING ' + name )
        pipe = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", model)]
        )
        pipe.fit(XTRAIN, yTRAIN)
        fitted_models.append(pipe)
        model_name.append(name)
        y_pred = pipe.predict(XVAL)
        #precision, recall, fscore, support = precision_recall_fscore_support(np.squeeze(yVAL), y_pred, labels=labels,average=None)
        p_weighted, r_weighted, f_weighted, support = precision_recall_fscore_support(yVAL, y_pred,average='weighted')
        accuracy = accuracy_score(yVAL, y_pred)
        CMVAL = confusion_matrix(yVAL, y_pred)

        CM_array = np.array(CMVAL).reshape(1,-1)
        f1scores.append(f_weighted)
        precisions.append(p_weighted)
        recalls.append(r_weighted)
        CMs.append(np.squeeze(CM_array).tolist())
        accuracies.append(accuracy)
        print(accuracy)

    scores = pd.DataFrame({
        "Model":model_name,
        "Accuracy":accuracies,
        "Precision":precisions,
        "Recall":recalls,
        "F1score":f1scores
    })

    scores.to_excel("RESULTS\\KINETICS_VAL\\CV\\scores"+TAG+".xlsx")

    print(scores)

    return fitted_models, scores

def evaluate_model_accuracy(model, validation_filenames,TAG,preprocess,filepath):
    """
    Evaluate the accuracy of a model against a validation set.

    Parameters:
        model: The trained model to be evaluated
        validation_filenames: filenames associated with validation set

    Returns:
        float: metrics
    """
    # Use the model to predict labels for the validation data
    def compute_mode(arr):
        unique_elements, counts = np.unique(arr, return_counts=True)
        max_count_index = np.argmax(counts)
        mode_value = unique_elements[max_count_index]
        return mode_value

    print("Starting evaluation of performance metrics "+TAG)
    yTRUE = []
    yTRUEtot = []
    yPRED = []
    yPREDtot = []
    names = []

    from tqdm import tqdm

    for i in tqdm(range(0,len(validation_filenames)), desc='evaluation - model predictions', unit = 'file'):
        filename = validation_filenames[i]
        NAME = os.path.basename(filename)

        with open(filename, 'rb') as file:
            # Load the object from the pickle file
            output = pickle.load(file)
            temp = np.array(output['keypoints_array'])
            ## evalutation of visibility --> if visibility not present
        X = preprocess(temp)
        X = X.reshape(np.shape(X)[0],-1)
        yTRUE_i = output['label']
        yTRUE.append(yTRUE_i.tolist())

        yTRUE_array = np.ones((np.shape(X)[0])) * output['label']

        yPRED_array = model.predict(X)
        yTRUEtot.append(yTRUE_array.tolist())
        yPREDtot.append(yPRED_array.tolist())
        names.append(NAME)

        yPRED_i= compute_mode(yPRED_array)
        yPRED.append(yPRED_i.tolist())


    yPRED = np.array(yPRED)
    yTRUE = np.array(yTRUE)
    precisionVAL,recallVAL,fscoreVAL,_ = precision_recall_fscore_support(np.squeeze(yTRUE), yPRED,average='macro')

    # print('Evaluate Metrics Completed')
    # Calculate accuracy by comparing predicted labels with true labels

    results = {
        'yPREDtot': yPREDtot,
        'yTRUEtot': yTRUEtot,
        'names': names
    }

    with open(filepath, 'wb') as f:
        # Serialize and save the object to the file
        pickle.dump(results, f)

    return results

def onlineFIT(train_filenames):

    model = SGDClassifier(loss='hinge',early_stopping=False, validation_fraction=0.1,n_iter_no_change=5)

    for repetitions in range(0,100):
        indices = np.random.choice(len(train_filenames), size=100, replace=False)
        XTRAIN,yTRAIN = loadDATA(train_filenames[indices], 1)

        model.partial_fit(XTRAIN, yTRAIN)


    return model

