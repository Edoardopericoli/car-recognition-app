import click
import os
from pathlib import Path
from CarModelClassifier import pipeline
from CarModelClassifier import models


@click.command()
@click.option('--params_file',
              default=r"initial_parameters.yml",
              help='config file containing initial parameters', type=str)
@click.option('--username', default=r"trial",
              help='username to be used for model saving',
              type=str)
@click.option('--shows_only_summary', default=False,
              help='if True the program stops after having shown \
                    the model summary',
              type=bool)
@click.option('--net', default='effnetb1',
              help='the model you want to use',
              type=str)
@click.option('--bounding_cpu', default=False,
              help='if True the program will use 8 threads',
              type=bool)
@click.option('--split_data', default=True,
              help='if True data will be splitted accordingly',
              type=bool)
@click.option('--data_type', default='old',
              help='type of images',
              type=str)
@click.option('--get_cropped_data_stanford', default=False,
              help='if True takes data cropped from stanford dataset',
              type=bool)
def main(params_file, username, shows_only_summary, net,
         bounding_cpu, split_data, data_type,
         get_cropped_data_stanford):

    if net == 'effnetb1':
        net = models.EffnetB1
    elif net == 'effnetb7':
        net = models.EffnetB7
    elif net == 'prototype':
        net = models.Prototype

    file_path = Path((os.path.dirname(os.path.abspath(__file__))))
    initial_parameters_path = file_path / 'config' / params_file

    pipeline.run(initial_parameters_path,
                 username,
                 shows_only_summary,
                 bounding_cpu=bounding_cpu,
                 net=net,
                 split_data=split_data,
                 data_type=data_type,
                 get_cropped_data_stanford=get_cropped_data_stanford)


if __name__ == "__main__":
    main()
