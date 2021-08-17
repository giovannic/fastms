from pickle import dump

def save_model(model, path):
    model.save(path)

def save_scaler(scaler, path):
    with open(path, 'wb') as f:
        dump(scaler, f)
