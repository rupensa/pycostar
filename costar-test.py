import logging
import os
import sys

from algorithms.costar import CoStar
from pprint import pprint


def execute_test(view_file_paths, n_iterations):
    # read input data
    # each file contains a matrix

    V = [[]] * len(view_file_paths)

    for i, filename in enumerate(view_file_paths):
        matrix = []
        with open(filename) as file:
            for line in file:
                line = line.strip()
                matrix.append(list(map(int, line.split(" "))))

        V[i] = matrix

    # print the loaded views
    for i, v in enumerate(V):
        print("View #{0}. Matrix size ({1},{2})".format(i, len(v), len(v[0])))

    # load data from file
    # path = os.path.join(config.__cache_folder, '20newsgroups-reduced-1-terms-tfidf.matrix')
    # V = [mutils.read_from_file(path)]

    print()
    print("Run costar algorithm for {0} iterations...".format(n_iterations))

    cs_model = CoStar(n_iterations=n_iterations, init='standard_without_merge')
    cs_model.fit(V)

    print()
    print("Rows clusters")
    pprint(cs_model.rows_)
    print("Column clusters")
    pprint(cs_model.columns_)
    print()
    print("Execution time: {0} seconds".format(cs_model.execution_time_))


if __name__ == '__main__':
    # configure the logger to printout on system out
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    data_path = "./resources/datasets/small/"
    view_files = [os.path.join(data_path, 'data.txt'), os.path.join(data_path, 'data1.txt')]

    execute_test(view_files, 1000)
