from CarModelClassifier.estimation import prediction
import warnings

# After making sure they don't represent a problem
warnings.filterwarnings("ignore")
def main():
    out_df = prediction()
    print(out_df)


if __name__ == "__main__":
    main()
