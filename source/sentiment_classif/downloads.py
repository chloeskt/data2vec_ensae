from pathlib import Path
import gdown
import os
D2VEC_IMDB = '1aR2Aq-MXMiEA7JBzBJQ65W3l-OXt4nD5'
D2VEC_AIRLINE = ''
BERT_IMDB = '1DHxPf31Q_ezYSmsVokPXjkhk_b69a1z8'
BERT_AIRLINE = '1rPISyDucv9_QXBRJOGKmfsPbG3ve5P5s'



def download_from_drive(file_id, data_dir, filename):
    DRIVE_URL = 'https://drive.google.com/uc?id='
    url = DRIVE_URL + file_id
    data_dir = Path(data_dir)
    filename = Path(filename)
    path = data_dir/filename
    if not data_dir.exists():
        os.mkdir(data_dir)
    if not path.is_file():
        print("Downloading ...")
        with open(path, 'wb') as f:
            gdown.download(url, f, quiet=False)


def get_weights():
    data_dir = Path('weights')
    download_from_drive(D2VEC_IMDB, data_dir, 'd2vec_imdb')
    download_from_drive(D2VEC_AIRLINE, data_dir, 'd2vec_airline')
    download_from_drive(BERT_IMDB, data_dir, 'bert_imdb')
    download_from_drive(BERT_AIRLINE, data_dir, 'bert_airline')
