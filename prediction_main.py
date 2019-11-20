from CarModelClassifier.estimation import prediction
import click


@click.command()
@click.option('--execution_path', default=r"./", help='config model path',
              type=str)
@click.option('--images_path', default=r"./",
              help='path of images (file or directory)', type=str)
@click.option('--labels_info_path', default=r"./",
              help='path of images (file or directory)', type=str)
def main(execution_path, images_path, labels_info_path):
    out_df = prediction(execution_path, images_path, labels_info_path)
    print(out_df)


if __name__ == "__main__":
    main()
