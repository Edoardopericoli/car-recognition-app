from CarModelClassifier.estimation import evaluation


def main():
    accuracy = evaluation()
    print(f'Accuracy: {accuracy:.2f}')


if __name__ == "__main__":
    main()
