from model import DL
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if __name__== '__main__':
    config = {'data_dir' : 'C:/Users/trist/Documents/GitHub/DeepLearningProject/dataset-split',
              'results_dir' : 'C:/Users/trist/Documents/GitHub/DeepLearningProject/results/wo_augmentation',
              'epochs' : 100,
              'batch_size' : 16,
              'learning_rate' : 0.0001,
              'target_size' : (256, 256),
              'metrics' : ['acc', 'AUC', 'Precision', 'Recall']
                }

    gen =  ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True)

    session = DL(config)
    session.set_data_generator(gen)
    session.train_model()
    session.evaluate_model()
    session.save_results()