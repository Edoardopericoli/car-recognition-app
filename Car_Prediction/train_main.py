import click
from Car_Prediction import efficientnet_pipeline


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
@click.option('--bounding_cpu', default=False,
              help='if True the program will use 8 threads',
              type=bool)
@click.option('--prepare_data', default=True,
              help='if True data will be splitted accordingly',
              type=bool)
def main(initial_parameters_path, username, shows_only_summary,
         bounding_cpu, prepare_data=True):
    summary = efficientnet_pipeline.run(initial_parameters_path,
                                        username,
                                        shows_only_summary,
                                        bounding_cpu=bounding_cpu,
                                        prepare_data=prepare_data)
    print(summary)


if __name__ == "__main__":
    main()
