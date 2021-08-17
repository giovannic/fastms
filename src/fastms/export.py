from pickle import dump, load

def save_model(model, path):
    model.save(path)

def save_scaler(scaler, path):
    with open(path, 'wb') as f:
        dump(scaler, f)

def load_scaler(path):
    with open(path, 'rb') as f:
        return load(f)
