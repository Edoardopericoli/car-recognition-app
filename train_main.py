import click
from Car_Prediction import pipeline
from Car_Prediction import models


@click.command()
@click.option('--initial_parameters_path',
              default=r"./config/initial_parameters.yml",
              help='config file containing initial parameters', type=str)
@click.option('--username', help='username to be used for model saving',
              type=str)
@click.option('--shows_only_summary', default=False,
              help='if True the program stops after having shown \
                    the model summary',
              type=bool)
@click.option('--net', default='effnet',
              help='the model you want to use',
              type=str)
@click.option('--bounding_cpu', default=False,
              help='if True the program will use 8 threads',
              type=bool)
@click.option('--prepare_labels', default=False,
              help='if True labels will be prepared accordingly',
              type=bool)
@click.option('--split_data', default=True,
              help='if True data will be splitted accordingly',
              type=bool)
@click.option('--target_variable', default='model',
              help='target variable of the model',
              type=str)
@click.option('--data_type', default='old',
              help='type of images',
              type=str)
@click.option('--get_cropped_data_stanford', default=False,
              help='if True takes data cropped from stanford dataset',
              type=bool)
def main(initial_parameters_path, username, shows_only_summary, net,
         bounding_cpu, prepare_labels, split_data, target_variable, data_type, get_cropped_data_stanford):

    if net == 'effnet':
        net = models.Effnet
    if net == 'prototype':
        net = models.Prototype

    pipeline.run(initial_parameters_path,
                                        username,
                                        shows_only_summary,
                                        bounding_cpu=bounding_cpu,
                                        net=net,
                                        prepare_labels=prepare_labels,
                                        split_data=split_data,
                                        target_variable=target_variable,
                                        data_type=data_type,
                                        get_cropped_data_stanford=get_cropped_data_stanford)


if __name__ == "__main__":
    main()
