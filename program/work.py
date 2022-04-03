
import tempfile

filenames = tempfile._get_candidate_names()

def generate_prediction():

    name = next(filenames)

    return dict(name=name)




