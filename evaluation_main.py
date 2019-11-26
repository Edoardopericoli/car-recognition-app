from CarModelClassifier.estimation import evaluation
import click
import warnings

# After making sure they don't represent a problem
warnings.filterwarnings("ignore")
@click.command()
@click.option('--custom_images',
              default=False,
              help='if True use custom images to evaluate the model', 
              type=bool)
def main(custom_images):
    accuracy = evaluation(custom_images=custom_images)
    print(f'Accuracy: {accuracy:.2f}')


if __name__ == "__main__":
    main()
