from pickle import dump, load

def save_model(model, path):
    model.save(path)

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return load(f)

def save_scaler(scaler, path):
    save_pickle(scaler, path)

def load_scaler(path):
    load_pickle(path)

def save_calibrator(calibrator, path):
    save_pickle(calibrator, path)

def load_calibrator(path):
    load_pickle(path)
