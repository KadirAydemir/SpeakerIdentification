from flask import Flask,render_template,request
import os, glob
from sklearn.neural_network import MLPClassifier
from pydub import AudioSegment
from threading import Thread
import pandas as pd
from zipfile import ZipFile
import librosa
import soundfile
import numpy as np

model = None
emodel = None

ALLOWED_EXTENSIONS = {'wav','mp3'}
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(app.instance_path, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def extract_feature(file_name, mfcc, chroma, mel, tempoGram, zeroCross):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if(np.ndim(X) == 2):
            X = np.asfortranarray(X[:,0])
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
            result = np.hstack((result, mfccs))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        if tempoGram:
            tempo = np.mean(
                librosa.feature.tempogram(y=X, sr=22050, onset_envelope=None, hop_length=512, win_length=384,
                                          center=True, window='hann'))
            result = np.hstack((result, tempo))
        if zeroCross:
            zero = np.mean(librosa.feature.zero_crossing_rate(y=X, frame_length=2048, hop_length=512, center=True))
            result = np.hstack((result, zero))

    return result

def get_model():
    global model
    x_train = pd.read_csv(os.path.join(app.instance_path, "features.csv"), error_bad_lines=False, encoding='utf-8').to_numpy()
    y_train = pd.read_csv(os.path.join(app.instance_path, "users.csv"), error_bad_lines=False, encoding='utf-8').to_numpy()
    model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(500,),learning_rate='adaptive', max_iter=1500)
    model.fit(x_train, y_train)

def add_data_to_csv(newuser):
    #Take ML Data'
    x = pd.read_csv(os.path.join(app.instance_path, "features.csv"), error_bad_lines=False, encoding='utf-8').values.tolist()
    y = pd.read_csv(os.path.join(app.instance_path, "users.csv"), error_bad_lines=False, encoding='utf-8').values.tolist()
    #Add new user features
    for file in glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], 'newuser','*.wav')):
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True, tempoGram=False, zeroCross=False)
        x.append(feature)
        y.append(newuser[0])
        os.remove(file)

    #Save new ML Data
    df = pd.DataFrame(x)
    dfy = pd.DataFrame(y)
    df.to_csv (os.path.join(app.instance_path, "features.csv"), index = None, header=True) #Don't forget to add '.csv' at the end of the path
    dfy.to_csv(os.path.join(app.instance_path, "users.csv"), index=None, header=True)  # Don't forget to add '.csv' at the end of the path


def allowed_file(filename):
    #Return format without dot if its in ALLOWED EXTENSIONS
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/emotion')
def eindex():
    return render_template('eindex.html')


@app.route('/getUser',methods=['Post'])
def getUser():

    #Take record from request
    record = request.files['record']

    #Check is mp3 or wav
    if record and allowed_file(record.filename):
        recordname = record.filename.rsplit('.', 1)[1].lower()
        recordpath = os.path.join(app.config['UPLOAD_FOLDER'], record.filename)
        record.save(recordpath) #Save in uploads folder

        #If file is mp3 convert to wav
        if recordname == 'mp3':
            sound = AudioSegment.from_mp3(os.path.join(app.config['UPLOAD_FOLDER'], record.filename))
            recordpath = os.path.join(app.config['UPLOAD_FOLDER'],record.filename.rsplit('.', 1)[0].lower()) + '.wav'
            sound.export(recordpath,format('wav'))
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], record.filename)) #Delete mp3 file

        feature = extract_feature(recordpath, mfcc=True, chroma=True, mel=True, tempoGram=False,zeroCross=False)
        os.remove(recordpath)
        x_test = feature.reshape(1, -1)
        y_pred = model.predict(x_test)
        return y_pred[0]

    else:
        return 'Wrong Format'


@app.route("/addUser", methods=['POST'])
def addUser():
    uploadedzip = request.files['zipfile']
    zippath = os.path.join(app.config['UPLOAD_FOLDER'], "newuser", uploadedzip.filename)
    uploadedzip.save(zippath)  # Save in uploads folder

    #Extract file in zip and remove zip
    with ZipFile(uploadedzip, 'r') as zip:
        zip.extractall(os.path.join(app.config['UPLOAD_FOLDER'], 'newuser'))
    os.remove(zippath)

    #If mp3 is exist conver to wav
    for file in glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], 'newuser', '*.mp3')):
        sound = AudioSegment.from_mp3(file)
        filepath = os.path.join(file.rsplit('.', 1)[0].lower()) + '.wav'
        sound.export(filepath, format('wav'))
        os.remove(file)

    #Take user info from csv file
    newuser = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'newuser','newuser.csv'),
                          error_bad_lines=False, encoding='utf-8').values.tolist()
    add_data_to_csv(newuser)
    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], 'newuser','newuser.csv'))

    #Update ML model
    Thread(target=get_model()).start()
    return "Success"


def get_emodel():
    global emodel
    x_train = pd.read_csv(os.path.join(app.instance_path, "efeatures.csv"), error_bad_lines=False,
                          encoding='utf-8').to_numpy()
    y_train = pd.read_csv(os.path.join(app.instance_path, "emotions.csv"), error_bad_lines=False,
                          encoding='utf-8').to_numpy()
    emodel = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(500,),
                          learning_rate='adaptive', max_iter=1500)
    emodel.fit(x_train, y_train)

@app.route('/getEmotion',methods=['Post'])
def getEmotion():

    #Take record from request
    record = request.files['record']

    #Check is mp3 or wav
    if record and allowed_file(record.filename):
        recordname = record.filename.rsplit('.', 1)[1].lower()
        recordpath = os.path.join(app.config['UPLOAD_FOLDER'], record.filename)
        record.save(recordpath) #Save in uploads folder

        #If file is mp3 convert to wav
        if recordname == 'mp3':
            sound = AudioSegment.from_mp3(os.path.join(app.config['UPLOAD_FOLDER'], record.filename))
            recordpath = os.path.join(app.config['UPLOAD_FOLDER'],record.filename.rsplit('.', 1)[0].lower()) + '.wav'
            sound.export(recordpath,format('wav'))
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], record.filename)) #Delete mp3 file

        feature = extract_feature(recordpath, mfcc=True, chroma=True, mel=True, tempoGram=False,zeroCross=False)
        os.remove(recordpath)
        x_test = feature.reshape(1, -1)
        y_pred = emodel.predict(x_test)
        return y_pred[0]

    else:
        return 'Wrong Format'



if __name__ == '__main__':
    Thread(target=get_model()).start()
    Thread(target=get_emodel()).start()
    app.run(debug = True)

