from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle
from utils import *
import argparse


def main(training_filename="data/training.data"):
    data, labels = read_data(training_filename)

    # vectorization of the data and initializing the classifier
    text_processing_clf = Pipeline(
        [('vect', CountVectorizer(stop_words={'english'})), ('tfidf', TfidfTransformer()), ('clf', LinearSVC())])
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    y_train_main_class = [x.split(":")[0] for x in y_train]
    y_test_main_class = [x.split(":")[0] for x in y_test]
    primary_classifier = text_processing_clf.fit(x_train, y_train_main_class)
    primary_class_predictions_test = primary_classifier.predict(x_test)
    logging.info(
        "accuracy for predicting primary class : " + str(
            accuracy_score(y_test_main_class, primary_class_predictions_test)))
    pickle.dump(primary_classifier, open("models/primary_classifier_svm.pkl", "wb"))

    # collecting data by primary class
    class_wise_train_data, class_wise_train_labels = divide_data_by_main_class(x_train, y_train)
    class_wise_test_data, class_wise_test_labels = divide_data_by_main_class(x_test, y_test)

    # training classifier for a specific subclass
    classifiers = {}
    for each_main_class in class_wise_train_data.keys():
        classifiers[each_main_class] = Pipeline(
            [('vect', CountVectorizer(stop_words={'english'})), ('tfidf', TfidfTransformer()),
             ('clf', MultinomialNB())])

    for each_main_class in class_wise_train_data.keys():
        curr_X_train = class_wise_train_data.get(each_main_class)
        curr_y_train = class_wise_train_labels.get(each_main_class)

        curr_X_test = class_wise_test_data.get(each_main_class)
        curr_y_test = class_wise_test_labels.get(each_main_class)

        curr_classifier = classifiers.get(each_main_class)
        curr_classifier.fit(curr_X_train, curr_y_train)
        pickle.dump(curr_classifier, open("models/" + each_main_class + "_classifier.pkl", "wb"))
        curr_predictions_NB = curr_classifier.predict(curr_X_test)
        curr_accuracy = accuracy_score(curr_y_test, curr_predictions_NB)
        logging.info("subclass accuracy under " + str(each_main_class) + " : " + str(curr_accuracy))

    # Test complete testset through two classifiers
    trueCounter = 0
    for i in range(len(primary_class_predictions_test)):
        curr_true_subclass = y_test[i].split(":")[1]
        curr_main_class_prediction = primary_class_predictions_test[i]
        curr_subclass_prediction = classifiers.get(curr_main_class_prediction).predict([x_test[i]])
        if curr_subclass_prediction == curr_true_subclass:
            trueCounter += 1

    logging.info("final accuracy (over complete hierarchy) : " + str(trueCounter / len(x_test)))

    hierarchy_plot(class_wise_train_labels.keys())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run train on the hierarchy model")
    parser.add_argument("-f", "--filename", default="data/training.data", type=str, help="path to training data file")
    args = parser.parse_args()
    training_filename = args.filename

    main(training_filename)
