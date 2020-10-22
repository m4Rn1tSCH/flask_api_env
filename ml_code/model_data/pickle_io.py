import pickle
import os

def store_pickle(model, file_name='model'):
    """
    Usage of a Pickle Model - Store a trained Model
    Parameters
    ----------
    model : the model to save
    file_name : name to save file as {name}.sav in current directory
    """
    model_file = f"{file_name}.sav"
    with open(model_file, mode='wb') as m_f:
        pickle.dump(model, m_f)

    print(f"Model saved in: {os.getcwd()}")
    return model_file

def open_pickle(model_file):
    """
    Usage of a Pickle Model - Loading of a Pickle File

    Model file can be opened either with:
    FILE NAME
    open_pickle(model_file="model.sav")
    INTERNAL PARAMETER
    open_pickle(model_file=model_file)
    """
    with open(model_file, mode='rb') as m_f:
        model = pickle.load(m_f)
    return model
