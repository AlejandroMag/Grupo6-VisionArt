from hu_gen import generate_hu_moments_file
from testing_model import load_and_test
from training_model import train_model

generate_hu_moments_file()
model = train_model()
load_and_test(model)
