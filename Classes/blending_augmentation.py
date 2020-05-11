from model import DL
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if __name__== '__main__':
    config = {'data_dir' : 'C:/Users/trist/Documents/GitHub/DeepLearningProject/dataset-split',
              'results_dir' : 'C:/Users/trist/Documents/GitHub/DeepLearningProject/results/wo_augmentation',
              'model_dir' : 'blending_augmentation',
              'epochs' : 100,
              'batch_size' : 16,
              'learning_rate' : 0.0001,
              'target_size' : (256, 256),
              'metrics' : ['acc', 'AUC', 'Precision', 'Recall'],
              'additional_augmentations' : ['blend']
                }

    session = DL(config)
    session.train_model()
    session.evaluate_model()
    session.save_results()