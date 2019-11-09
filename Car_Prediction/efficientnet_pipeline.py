from Car_Prediction import model_utils, data, label
import logging


def run(initial_parameters_path, username, shows_only_summary,
        bounding_cpu=False, prepare_data=True):

    model_utils.setting_log()

    if prepare_data:
        logging.info('Starting splitting and preparing processes')
        #label.prepare() #todo:   label preparation si fa solo la prima volta, non dovrebbe essere nella pipeline
        data.split()
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

    base_model = model_utils.setup_base_model()
    model = model_utils.setup_final_layers(base_model, train_generator)

    if shows_only_summary:
        return model.summary()

    history = model_utils.train_model(train_generator, validation_generator,
                                      initial_parameters, train_df, model)

    model_utils.save_model_architecture(username, model, initial_parameters)
    model_utils.save_model_performance(username, history, initial_parameters)
    logging.info('Process finished')
    return model.summary()
