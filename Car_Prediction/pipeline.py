from Car_Prediction import utils, splitter, label
from Car_Prediction.models import Effnet
import logging


def run(initial_parameters_path="./config/initial_parameters.yml",
        username="rrr", shows_only_summary=False, net=Effnet,
        bounding_cpu=False, prepare_labels=False,
        split_data=True, target_variable='model', origin_data_path='data/labels/all_labels.csv',
        get_cropped_data_stanford=True):

    utils.setting_log()

    if prepare_labels:
        label.prepare()

    if split_data:
        logging.info('Starting splitting and preparing processes')
        splitter.split(target_variable=target_variable,
                       origin_data_path=origin_data_path,
                       get_cropped_data_stanford=get_cropped_data_stanford)
        logging.info('Splitting ended successfully')

    if bounding_cpu:
        utils.bound_cpu(n_threads=8)

    logging.info('Starting the process')
    initial_parameters = utils.load_parameters(initial_parameters_path)
    train_df, validation_df = utils.load_labels_dfs()
    train_image_generator, validation_image_generator = utils.get_image_generators()

    train_generator = utils.get_generator(train_image_generator,
                                                train_df, "../data/train/",
                                                initial_parameters,
                                                train=True)

    validation_generator = utils.get_generator(
                                       validation_image_generator,
                                       validation_df,
                                       "../data/validation/",
                                       initial_parameters,
                                       train=False)

    logging.info('Transforming data using ImageDataGenerator')

    model_net = net(train_generator, initial_parameters)
    model = model_net.model
    model.summary()

    if shows_only_summary:
        return

    history = utils.train_model(train_generator, validation_generator,
                                initial_parameters, train_df, model)

    utils.save_model_info(username, model, initial_parameters, history)
    logging.info('Process finished')
