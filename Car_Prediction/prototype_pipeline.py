from Car_Prediction import model_utils, splitter, label
import logging
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.models import Sequential


def run(initial_parameters_path, username, shows_only_summary,
        bounding_cpu, prepare_labels, split_data, target_variable):

    model_utils.setting_log()

    if prepare_labels:
        label.prepare()

    if split_data:
        logging.info('Starting splitting and preparing processes')
        splitter.split(target_variable=target_variable)
        logging.info('Splitting ended successfully')

    if bounding_cpu:
        model_utils.bound_cpu(n_threads=8)

    logging.info('Starting the process')
    initial_parameters = model_utils.load_parameters(initial_parameters_path)
    train_df, validation_df = model_utils.load_labels_dfs()
    train_image_generator, validation_image_generator = model_utils.get_image_generators()

    train_generator = model_utils.get_generator(train_image_generator,
                                                train_df, "../data/train/",
                                                initial_parameters,
                                                train=True)

    validation_generator = model_utils.get_generator(
                                       validation_image_generator,
                                       validation_df,
                                       "../data/validation/",
                                       initial_parameters,
                                       train=False)

    logging.info('Transforming data using ImageDataGenerator')

    model = Sequential([
        Conv2D(5, 4, activation='relu', input_shape=(initial_parameters['IMG_HEIGHT'], initial_parameters['IMG_WIDTH'], 3)),
        MaxPooling2D((8, 8)),
        Dropout(0.1, seed=initial_parameters['seed']),
        Conv2D(5, 4, padding='same', activation='relu'),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(196, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    if shows_only_summary:
        return
    history = model_utils.train_model(train_generator, validation_generator,
                                      initial_parameters, train_df, model)

    model_utils.save_model_architecture(username, model, initial_parameters)
    model_utils.save_model_performance(username, history, initial_parameters)
    logging.info('Process finished')
