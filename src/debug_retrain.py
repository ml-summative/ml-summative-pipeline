import traceback
from preprocessing import ImagePreprocessor, DataPipeline
from model import TrafficNetModel

preprocessor = ImagePreprocessor()
data_pipeline = DataPipeline()

try:
    print('Merging retraining data...')
    data_pipeline.merge_retraining_data()

    print('Creating data generators...')
    train_gen, val_gen, test_gen = preprocessor.create_data_generators('data/train', 'data/test', batch_size=32)
    print('Generators created')

    print('Loading model...')
    model_handler = TrafficNetModel(model_path='models/traffic_net_model.h5')
    print('Model loaded')

    print('Starting retrain call...')
    history = model_handler.retrain(train_gen, val_gen, epochs=1, learning_rate=1e-4)
    print('Retrain completed')

except Exception as e:
    print('Exception during debug retrain:')
    traceback.print_exc()
