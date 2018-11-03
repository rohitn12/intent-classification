import argparse

from utils import *


def main(input_question):
    all_models = load_available_model()
    primary_class_prediction = all_models['primary'].predict([input_question])[0]
    secondary_class_prediction = all_models[primary_class_prediction].predict([input_question])[0]
    logging.info("Predicted class hierarchy is : " + primary_class_prediction + ":" + secondary_class_prediction)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run model on the given datapoint")
    parser.add_argument("-s", "--string", default=None, type=str, help="Enter question to classify")
    args = parser.parse_args()
    if not args.string:
        logging.info(parser.print_help())
        logging.error("No question given : Please give a question to be classified")
    else:
        main(args.string)
