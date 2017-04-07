import logging
from random import choice
from random import randint
from time import time

import numpy as np
import scipy
from scipy.sparse import issparse
from sklearn.base import BaseEstimator
from sklearn.base import ClusterMixin
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans
from sklearn.utils import check_array


class CoStar(BaseEstimator, ClusterMixin, TransformerMixin):
    """ CoStar algorithm (Ienco, 2013).

    CoStar is an algorithm created to deal with multi-view data.
    It finds automatically the best number of row / column clusters.

    Parameters
    ------------

    n_iterations : int, optional, default: 300
        The number of iterations to be performed.

    min_n_row_clusters : int, optional, default: 2,
        The number of clusters for rows, should be at least 2

    init : {'standard_with_merge', 'standard_without_merge', 'random', 'kmeans'}, optional, default: 'standard_without_merge'
        The initialization methods:

        'standard_with_merge' : the initial clusters are discrete (one item per cluster), but identical
            rows are merged into one single cluster

        'standard_without_merge' : the initial clusters are discrete (one item per cluster)

        'random' : the initial clusters are random and are less than the number of elements

        'kmeans' : the initial cluster for both rows and columns are chosen by an execution of kmeans.
                    the number of cluster is equal to number_of_elements * scaling_factor

    scaling_factor : float, optional, default: 0,01
        The scaling factor to use to define the number of cluster desired in case of 'kmeans' initialization method.

    Attributes
    -----------

    rows_ : array, length n_rows
        Results of the clustering on rows. `rows[i]` is `c` if
        row `r` is assigned to cluster `c`. Available only after calling ``fit``.

    columns_ : array-like, shape (n_view, n_features_per_view)
        Results of the clustering on columns, for each view is like `rows`.
        
    execution_time_ : float
        The execution time. 

    References
    ----------

    * Ienco, Dino, et al., 2013. `Parameter-less co-clustering for star-structured heterogeneous data.`
        Data Mining and Knowledge Discovery 26.2: 217-254

    """

    def __init__(self, n_iterations=500, min_n_row_clusters=2, init='standard_without_merge', scaling_factor=0.1):

        self.n_iterations = n_iterations
        self.min_n_row_clusters = max(min_n_row_clusters, 2)
        self.initialization_mode = init
        self.initialization_scaling_factor = scaling_factor

        # these fields will be available after calling fit
        self.rows_ = None
        self.columns_ = None

        np.seterr(all='ignore')

    def _init_all(self, V):
        """
        Initialize all variables needed by the model.

        :param V: the dataset
        :return:
        """

        # verify that all matrices are correctly represented
        # check_array is a sklearn utility method
        self._dataset = [None] * len(V)

        for i, X in enumerate(V):
            self._dataset[i] = check_array(X, accept_sparse='csr', dtype=[np.float64, np.float32, np.int32])

        self._csc_dataset = None
        if issparse(self._dataset[0]):
            # transform also to csc
            self._csc_dataset = [d.tocsc() for d in self._dataset]

        # the number of views
        self._n_views = len(self._dataset)

        # Support values to compute delta tau for columns
        self._Ic = np.zeros(self._n_views)
        self._Cc = np.zeros(self._n_views)

        # Support values to compute delta tau for rows
        self._Ir = 0.0
        self._Cr = 0.0

        # the number of documents in the data
        self._n_documents = self._dataset[0].shape[0]

        # the number of features per each view
        self._n_features_per_view = np.array([v.shape[1] for v in self._dataset])

        # the number of row clusters
        self._n_row_clusters = 0

        # the number of column clusters for each view, initialized as 1 per feature
        self._n_col_clusters = [0] * self._n_views

        # a list of n_documents elements
        # for each document d contains the row cluster index d is associated to
        self._row_assignment = np.zeros(self._n_documents, int)

        # a list of lists, the list in position i contains one element for each feature of the view V[i]
        # the cell _col_assignment[v][f] contains the index of the column cluster the feature f is associated to
        # in view v
        self._col_assignment = [np.zeros(self._n_features_per_view[v], int) for v in range(self._n_views)]

        # T is a contingency matrix
        # T is a list of n_view elements, the element with index v is a matrix
        # with shape = (n_row_clusters, n_col_clusters[v)
        self._T = None

        # a list of total sum for each view
        self._tot_per_view = None

        # sum_elements_of_view^2 for each view
        self._square_tot_per_view = None

        # 2 / sum_elements_of_view for each view
        self._two_divided_by_tot_per_view = None

        # sum of data values per row cluster
        self._tot_t_per_rc = None

        # sum of data values per col cluster
        self._tot_t_per_cc = None

        # sum of (data values ^ 2) per row cluster
        self._tot_t_square_per_rc = None

        # sum of (data values ^ 2) per col cluster
        self._tot_t_square_per_cc = None

        if self.initialization_mode == 'random':
            self._random_initialization()
        elif self.initialization_mode == 'kmeans':
            self._kmeans_initialization()
        elif self.initialization_mode == 'standard_with_merge':
            self._discrete_initialization(merge_identical_objects=True)
        else:
            self._discrete_initialization(merge_identical_objects=False)

        self._update_I_and_C()

        self._performed_moves_on_rows = 0
        self._performed_moves_on_cols = [0] * self._n_views

        logging.debug("[INFO] Row's initialization: {0}".format(list(self._row_assignment)))
        logging.debug("[INFO] Col's initialization: {0}".format([list(c) for c in self._col_assignment]))

    def fit(self, V, y=None):
        """Compute coclusters for the provided data using costar.

        Parameters
        ----------

        V : list of array-like or sparse matrices, one for each view;
            shape of each matrix = (n_samples, n_features_in_view)

        """

        # Initialization phase
        self._init_all(V)

        start_time = time()

        # Execution phase
        actual_n_iterations = 0

        while actual_n_iterations != self.n_iterations:
            iter_start_time = time()
            logging.debug("[INFO] ##############################################\n" +
                         "\t\t### Iteration {0}".format(actual_n_iterations))

            if actual_n_iterations % 100 == 0:
                logging.info("Iteration #{0}".format(actual_n_iterations))

            # update row
            # logging.debug("[INFO] Row partitioning")
            self._row_partition_clustering()
            for view in range(self._n_views):
                # logging.debug("[INFO] View #{0} column partitioning".format(view))
                self._column_partition_clustering(view)
            actual_n_iterations += 1
            iter_end_time = time()
            logging.debug("[INFO] # row clusters: {0}; # col clusters: {1}".format(self._n_row_clusters, self._n_col_clusters))
            logging.debug("[INFO] Iteration time: {0}".format(iter_end_time - iter_start_time))

        end_time = time()

        execution_time = end_time - start_time
        logging.info('#####################################################')
        logging.info("[INFO] Execution time: {0}".format(execution_time))

        self.rows_ = np.copy(self._row_assignment).tolist()
        self.columns_ = [list(np.copy(r)) for r in self._col_assignment]
        self.execution_time_ = execution_time

        logging.info("[INFO] Number of row clusters found: {0}".format(self._n_row_clusters))
        logging.info("[INFO] Number of column clusters found per view: {0}".format(self._n_col_clusters))
        logging.info("[INFO] Number of moves performed on rows: {0} / {1}".format(self._performed_moves_on_rows, self.n_iterations))
        logging.info("[INFO] Number of moves performed on columns per view ({1} total possible moves per view): {0}"
                     .format(self._performed_moves_on_cols, self.n_iterations))

        return self

    def _kmeans_initialization(self):
        # init number of clusters
        self._n_col_clusters = [int(max(n * self.initialization_scaling_factor, 2)) for n in self._n_features_per_view]
        self._n_row_clusters = int(max(self._n_documents * self.initialization_scaling_factor, 2))

        self._col_assignment = [None] * self._n_views

        # apply kmeans for each view
        for view in range(self._n_views):
            kmeans = KMeans(n_clusters=self._n_col_clusters[view])
            self._col_assignment[view] = kmeans.fit_predict(np.transpose(self._dataset[view]))

        # apply kmeans on rows
        if self._n_views > 1:
            if issparse(self._dataset[0]):
                all_dataset = scipy.sparse.hstack(self._dataset)
            else:
                all_dataset = np.concatenate((self._dataset[0], self._dataset[1]), axis=1)
                for view in range(2, self._n_views):
                    all_dataset = np.concatenate((all_dataset, self._dataset[view]), axis=1)

        else:
            all_dataset = self._dataset[0]

        kmeans = KMeans(n_clusters=self._n_row_clusters)
        self._row_assignment = kmeans.fit_predict(all_dataset)

        # compute the contingency matrix for the new assignment
        self._T = self._compute_contingency_matrix_T()

        # update the T related matrices
        self._init_T_derived_fields()

    def _random_initialization(self):
        """
        Randomly initialize clusters on rows and columns

        :return:
        """

        # random init for rows

        # 2 <= _n_row_clusters <= (_n_documents / 2)
        self._n_row_clusters = self._random_rows_initialization()

        # random init for cols for each view
        for k in range(self._n_views):
            self._n_col_clusters[k] = self._random_columns_initialization_for_view(k)

        # update the contingency matrix and derivated fields
        self._T = self._compute_contingency_matrix_T()

        # update the T related matrices
        self._init_T_derived_fields()

    def _random_columns_initialization_for_view(self, view_index):
        """
        Randomly initialize clusters for columns of a specific view

        :param view_index: int, the index of the view to be considered
        :return: int, the number of created clusters
        """
        n_features = self._n_features_per_view[view_index]
        if n_features > 3:
            new_n_col_clusters = randint(2, int(n_features / 2))
        else:
            new_n_col_clusters = randint(2, n_features)

        # assign at least one feature per cluster
        self._col_assignment[view_index][:new_n_col_clusters] = range(new_n_col_clusters)

        # assign the other features randomly
        for i in range(new_n_col_clusters, n_features):
            self._col_assignment[view_index][i] = randint(0, new_n_col_clusters - 1)

        return new_n_col_clusters

    def _random_rows_initialization(self):
        """
        Randomly initialize clusters for rows.

        :return: the number of created clusters
        """
        if self._n_documents > 3:
            new_n_row_clusters = randint(2, int(self._n_documents / 2))
        else:
            new_n_row_clusters = randint(2, self._n_documents)

        # assign at least one item for each of n_row_clusters
        self._row_assignment[:new_n_row_clusters] = range(new_n_row_clusters)

        # assign randomly the other elements
        for i in range(new_n_row_clusters, self._n_documents):
            self._row_assignment[i] = randint(0, new_n_row_clusters - 1)

        return new_n_row_clusters

    def _discrete_initialization(self, merge_identical_objects=True):
        """
        This initialization method basically assign each row to a new cluster and each feature to a column cluster.
        Note that if the merge_identical_objects parameter is set to True, identical rows (or respectively columns)
        are assigned to the same row cluster (column cluster). Rows are identical only if they have the same values
        in each view.

        IMPORTANT NOTE: The identical row / col merge is still to be tested!!

        :param merge_identical_objects: boolean, optional, default False
                If True identical rows (columns) are assigned to the same row cluster (column cluster),
                otherwise each row (column) is assigned to a new discrete cluster. The complexity of the merged solution
                is higher than the simple discrete assignment.

        :return:
        """

        if not merge_identical_objects:
            # simple assign each row to a row cluster and each column of a view to a column cluster
            self._n_row_clusters = self._n_documents
            self._n_col_clusters = np.copy(self._n_features_per_view)

            # assign each row to a row cluster
            self._row_assignment = np.arange(self._n_documents)

            # assign each column to a cluster
            self._col_assignment = [np.arange(v.shape[1]) for v in self._dataset]

            # in this case the T contingency matrix is equal to the data matrix
            if len(self._dataset) > 0 and not issparse(self._dataset[0]):
                # if are dense matrices
                self._T = [np.copy(v) for v in self._dataset]
            else:
                # if is a sparse matrix
                # TODO handle also T as a sparse matrix instead to transform in a dense one
                self._T = [v.toarray() for v in self._dataset]

            # init the T-derived fields for further computations
            self._init_T_derived_fields()

        else:
            logging.debug("[INFO] Start the initialization process")
            # generate an hashmap of string-converted rows (considering all views) and an hashmap
            # of string-converted columns (indipendently view per view)
            row_strings = [''] * self._n_documents

            for view in range(self._n_views):
                # increment the rows string for this view
                for ri in range(self._n_documents):
                    row_strings[ri] += " " + str(self._dataset[view][ri])

                cols_hashmap = {}
                next_col_cluster_index = 0

                for fi in range(self._n_features_per_view[view]):
                    # get the column string
                    col_array = [''] * self._n_documents

                    for row in range(self._n_documents):
                        # TODO in the sparse case, the string to be compared could be composed by pairs (index, value)
                        col_array[row] = str(get_element(self._dataset[view], row, fi))
                    col_string = " ".join(col_array)

                    # assign the column clusters
                    if col_string not in cols_hashmap:
                        cols_hashmap[col_string] = next_col_cluster_index
                        self._col_assignment[view][fi] = next_col_cluster_index
                        next_col_cluster_index += 1
                    else:
                        self._col_assignment[view][fi] = cols_hashmap[col_string]

                # save the number of column clusters for this view
                self._n_col_clusters[view] = next_col_cluster_index

            # assign the row clusters
            rows_hashmap = {}
            next_row_cluster_index = 0
            for ri, row in enumerate(row_strings):
                if row not in rows_hashmap:
                    rows_hashmap[row] = next_row_cluster_index
                    self._row_assignment[ri] = next_row_cluster_index
                    next_row_cluster_index += 1
                else:
                    self._row_assignment[ri] = rows_hashmap[row]

            # save the number of row clusters
            self._n_row_clusters = next_row_cluster_index

            # logging.debug("[INFO] Completed rows assignment.")

            # compute the contingency matrix for the new assignment
            self._T = self._compute_contingency_matrix_T()

            # update the T related matrices
            self._init_T_derived_fields()


    def _compute_contingency_matrix_T(self):
        """
        Initialize the T contingency matrix for all views. This object contains for each view a matrix
        of shape = (n_row_clusters, n_col_clusters[v])

        :param update_related_fields: boolean, default True
                Updates the matrices directly dependent by T (self._nTot, self._nTot_pow, self._two_dividedby_nTot)
        :return:
        """
        logging.debug("[INFO] Compute the contingency matrix...")

        if issparse(self._dataset[0]):
            new_t = self._compute_contingency_matrix_T_sparse()
        else:
            # dense case
            new_t = [None] * self._n_views

            for vi in range(self._n_views):
                # init the new matrix for view v of shape = (n_row_clusters * n_col_clusters[v])
                new_t[vi] = np.zeros((self._n_row_clusters, self._n_col_clusters[vi]), dtype=float)

                for di in range(self._n_documents):
                    row_cluster_of_d = self._row_assignment[di]

                    for ci in range(self._n_features_per_view[vi]):
                        col_cluster_of_c = self._col_assignment[vi][ci]
                        new_t[vi][row_cluster_of_d][col_cluster_of_c] += self._dataset[vi][di][ci]

        logging.debug("[INFO] End of contingency matrix computation...")
        return new_t

    def _compute_contingency_matrix_T_sparse(self):
        new_t = [None] * self._n_views

        for vi in range(self._n_views):
            # init the new matrix for view v of shape = (n_row_clusters * n_col_clusters[v])
            new_t[vi] = np.zeros((self._n_row_clusters, self._n_col_clusters[vi]), dtype=float)

            for di in range(self._dataset[vi].shape[0]):
                row = self._dataset[vi].getrow(di)
                row_cluster_of_d = self._row_assignment[di]
                for fi in row.indices:
                    col_cluster_of_c = self._col_assignment[vi][fi]
                    new_t[vi][row_cluster_of_d][col_cluster_of_c] += self._dataset[vi][di,fi]

        return new_t

    def _init_T_derived_fields(self):
        """
        Initialize the following lists:
        * total sum of values per view,
        * pow(total) per view,
        * 2/total per view
        * totals of cols per row cluster
        * totals of cols^2 per row cluster
        * totals of rows per col cluster
        * totals of rows^2 per col cluster

        :return:
        """

        logging.debug("[INFO] Init fields derived by the contingency matrix...")

        # list of totals for each view
        self._tot_per_view = [t.sum() for t in self._T]

        # list of totals ^ 2 for each view
        self._square_tot_per_view = np.power(self._tot_per_view, 2)

        # list of 2 / total for each view
        self._two_divided_by_tot_per_view = np.apply_along_axis(lambda x: 2/x, 0, self._tot_per_view)

        # sum of data values per row cluster
        self._tot_t_per_rc = np.empty(self._n_views, np.ndarray)

        # sum of data values per col cluster
        self._tot_t_per_cc = np.empty(self._n_views, np.ndarray)

        # sum of (data values ^ 2) per row cluster
        self._tot_t_square_per_rc = np.empty(self._n_views, np.ndarray)

        # sum of (data values ^ 2) per col cluster
        self._tot_t_square_per_cc = np.empty(self._n_views, np.ndarray)

        # compute some support values
        for view in range(self._n_views):
            view_square = np.power(self._T[view], 2)
            # sum per row of T[view] (axis = 1)
            self._tot_t_per_rc[view] = np.sum(self._T[view], 1)
            self._tot_t_square_per_rc[view] = np.sum(view_square, 1)
            # sum per col of T[view] (axis = 0)
            self._tot_t_per_cc[view] = np.sum(self._T[view], 0)
            self._tot_t_square_per_cc[view] = np.sum(view_square, 0)

    def _update_I_and_C(self):
        """
        Updates the following instance values:

        - self.Ic, array, length = n_views,
        - self.Cc, array, length = n_views,
        - self.Ir, double,
        - self.Cr, double,
        :return:
        """

        ir_acc = 0.0
        cr_acc = 0.0

        for view in range(self._n_views):
            # compute I
            sum_i = np.sum(np.power(self._tot_t_per_cc[view], 2))
            self._Ic[view] = 1 - (sum_i / self._square_tot_per_view[view])

            # compute C
            sum_c = np.nansum(np.true_divide(self._tot_t_square_per_rc[view], self._tot_t_per_rc[view]))
            self._Cc[view] = 1 - (sum_c / self._tot_per_view[view])

            # accumulate on i_acc for this view
            sum_ir = np.sum(np.power(self._tot_t_per_rc[view], 2))
            ir_acc += (sum_ir / self._square_tot_per_view[view])

            # accumulate on c_acc for this view
            sum_cr = np.nansum(np.true_divide(self._tot_t_square_per_cc[view], self._tot_t_per_cc[view]))
            cr_acc += (sum_cr / self._tot_per_view[view])

        self._Ir = self._n_views - ir_acc
        self._Cr = self._n_views - cr_acc

    def _row_partition_clustering(self):
        """
        Perform partitioning of the rows.

        :return:
        """

        # init at random row cluster
        selected_cluster = randint(1, self._n_row_clusters * 100) % self._n_row_clusters

        # init at random element of selected_cluster
        selected_element = choice(np.where(self._row_assignment == selected_cluster)[0])

        # logging.debug("[INFO] Selected row = {0}".format(selected_element))
        lambdas, sum_lambdas = self._compute_lambdas_per_row(selected_element)

        # get the number of clusters considering one empty cluster (that can be created during this iteration),
        # if the number of clusters is already equal to the number of documents the creation of a new cluster
        # will not be allowed
        n_row_clusters_plus_one_empty = min(self._n_documents, self._n_row_clusters + 1)

        all_delta_tau_r = self._delta_tau_r_cumulative(self._tot_t_per_cc, sum_lambdas, lambdas, selected_cluster)

        # consider all partitions of the neighborhood of the randomly selected cluster, i.e. partitions where the
        # random_object is moved to another existent cluster or to the empty cluster
        all_delta_tau_r = all_delta_tau_r[:n_row_clusters_plus_one_empty]
        min_delta_tau_r = np.min(all_delta_tau_r)
        equal_solutions = np.where(min_delta_tau_r == all_delta_tau_r)[0]
        e_min = equal_solutions[0]

        if min_delta_tau_r != 0:
            if len(equal_solutions) > 1:
                # logging.debug('[INFO] Tau', min_delta_tau_r)
                e_min = self._find_non_dominated_row(equal_solutions, lambdas, selected_cluster)

            go_on_normally = True
            if self._n_row_clusters == self.min_n_row_clusters:  # or e_min == self._n_row_clusters:
                # if the number of row cluster is already equal to min check that the move will not delete a cluster
                go_on_normally = self._check_row_clustering_size(selected_cluster)

            if go_on_normally:
                # go with the move
                self._row_assignment[selected_element] = e_min
                self._modify_row_cluster(lambdas, selected_cluster, e_min)
                # TODO perform incrementally within the modify_* functions
                self._update_I_and_C()
        else:
            logging.debug("[INFO] Ignored move of row {2} from row cluster {0} to {1}".format(selected_cluster,
                                                                                             e_min, selected_element))

    def _compute_lambdas_per_row(self, selected_row):
        """
        Compute lambda values related to the selected row element.

        In particular:
        * lambdas, array-like, shape = (n_view, n_col_cluster[view]),
                    contains for each view and for each column cluster, the sum of data related to the selected row.
        * sum_lambdas, array, length n_view,
                    contains for each view the sum of the row data

        :param selected_row: int, the id of the selected element
        :return: a pair (lambdas, sum_lambdas),
                    see the method description for more details
        """

        # contains for each view and for each column cluster, the sum of data related to the selected row.
        # selected_row_data_per_col_cluster
        lambdas = np.empty(self._n_views, np.ndarray)
        # contains for each view the sum of the row data
        # selected_row_data_per_view
        sum_lambdas = np.zeros(self._n_views)

        for view in range(self._n_views):
            lambdas[view] = self._sum_row_data_per_col_cluster(view, selected_row)
            sum_lambdas[view] = np.sum(lambdas[view])

        return lambdas, sum_lambdas

    def _sum_row_data_per_col_cluster(self, view, row_index):
        """
        Fixed a document (row), computes for each column cluster the sum of the values of the features
        associated to that cluster.

        :param view: int, the index of the view to consider
        :param row_index: int, the index of the document to consider
        :return:
        """
        sum_m = np.zeros(self._n_col_clusters[view])
        if issparse(self._dataset[view]):
            row = self._dataset[view].getrow(row_index)
            for fi in row.indices:
                sum_m[self._col_assignment[view][fi]] += row[0, fi]
        else:
            for fi in range(self._n_features_per_view[view]):
                sum_m[self._col_assignment[view][fi]] += self._dataset[view][row_index][fi]

        return sum_m

    def _delta_tau_r_cumulative(self, tot_t_per_cc, sum_lambdas, lambdas, original_cluster):
        """
        Compute the delta tau values for row clusters. Fixed a source_cluster computes the delta tau values
        for each of the other existent cluster (plus the empty one)

        :param tot_t_per_cc: the sum of t values grouped by column cluster
        :param sum_lambdas: the sum of lambdas per view
        :param lambdas: for each view and column cluster the difference due to the move of the element
        :param original_cluster: the cluster currently containing the element
        :return: list of int, the tau value for each row cluster
        """

        computed_taus = np.zeros(self._n_row_clusters + 1)
        division_one = np.empty(self._n_views, np.ndarray)
        subtraction_one = np.empty(self._n_views, np.ndarray)
        subtraction_two = np.empty(self._n_views, np.ndarray)

        division_two = np.nan_to_num(np.true_divide(np.multiply(sum_lambdas, 2), self._square_tot_per_view))

        for destination_cluster in range((self._n_row_clusters + 1)):
            x_array = np.zeros(self._n_views)
            y_array = np.zeros(self._n_views)

            for view in range(self._n_views):
                if division_one[view] is None:
                    # compute useful values
                    division_one[view] = np.nan_to_num(np.true_divide(lambdas[view], tot_t_per_cc[view]))
                    subtraction_one[view] = np.subtract(self._T[view][original_cluster], lambdas[view])
                    subtraction_two[view] = np.subtract(lambdas[view], self._T[view][original_cluster])

                if destination_cluster == self._n_row_clusters:
                    x_array[view] = np.nansum(np.multiply(division_one[view],
                                                          subtraction_one[view]))
                    y_array[view] = np.nansum(subtraction_two[view])

                else:
                    x_array[view] = np.nansum(np.multiply(division_one[view],
                                                          np.subtract(subtraction_one[view],
                                                                      self._T[view][destination_cluster])))
                    y_array[view] = np.nansum(np.add(subtraction_two[view], self._T[view][destination_cluster]))

            x = np.nansum(np.multiply(self._two_divided_by_tot_per_view, x_array))
            y = np.nansum(np.multiply(division_two, y_array))

            delta_tau_r = 0.0

            if (self._Ir * (self._Ir - y)) != 0:
                delta_tau_r = ((self._Ir * x) + (self._Cr * y)) / (self._Ir * (self._Ir - y))

            computed_taus[destination_cluster] = delta_tau_r

        # tau of the original_cluster should be zero
        computed_taus[original_cluster] = 0.0

        # check if the source is a singleton cluster and force useless move to empty cluster to 0.0
        is_singleton = not self._check_row_clustering_size(original_cluster)
        if is_singleton:
            computed_taus[-1] = 0.0

        return computed_taus

    def _delta_tau_r(self, tot_t_per_cc, sum_lambdas, lambdas, original_cluster, destination_cluster):
        """
        Compute the delta tau value for row clusters. Min is best.

        :param tot_t_per_cc: the sum of t values grouped by column cluster
        :param sum_lambdas: the sum of lambdas per view
        :param lambdas: for each view and column cluster the difference due to the move of the element
        :param original_cluster: the cluster currently containing the element
        :param destination_cluster: the cluster into which move the element
        :return: int, the tau value
        """

        if original_cluster == destination_cluster:
            return 0.0

        # check if the source is a singleton cluster and force useless move to empty cluster to 0.0
        if destination_cluster == self._n_row_clusters:
            is_singleton = not self._check_row_clustering_size(original_cluster)
            if is_singleton:
                return 0.0

        x_array = np.zeros(self._n_views)
        y_array = np.zeros(self._n_views)

        for view in range(self._n_views):
            if destination_cluster == self._n_row_clusters:
                x_array[view] = np.nansum(np.multiply(np.true_divide(lambdas[view], tot_t_per_cc[view]),
                                                      np.subtract(self._T[view][original_cluster], lambdas[view])))
                y_array[view] = np.nansum(np.subtract(lambdas[view], self._T[view][original_cluster]))

            else:
                x_array[view] = np.nansum(np.multiply(np.true_divide(lambdas[view], tot_t_per_cc[view]),
                                                      np.subtract(np.subtract(self._T[view][original_cluster],
                                                                              self._T[view][destination_cluster]),
                                                                  lambdas[view])))
                y_array[view] = np.nansum(np.add(np.subtract(self._T[view][destination_cluster],
                                                             self._T[view][original_cluster]), lambdas[view]))

        x = np.nansum(np.multiply(self._two_divided_by_tot_per_view, x_array))
        y = np.nansum(np.multiply(np.true_divide(np.multiply(sum_lambdas, 2), self._square_tot_per_view), y_array))

        delta_tau_r = 0.0

        if (self._Ir * (self._Ir - y)) != 0:
            delta_tau_r = ((self._Ir * x) + (self._Cr * y)) / (self._Ir * (self._Ir - y))

        return delta_tau_r

    def _find_non_dominated_row(self, equal_solutions, lambdas, original_cluster):
        """
        Checks the clusters in the list equal_solution and select the best solution

        :param equal_solutions: list of row cluster indexes within choose the best
        :param lambdas: the result of _sum_row_data_per_col_cluster function foreach view
        :param original_cluster: the id of the row cluster that contains the item we want to move
        :return:
        """
        logging.debug("[INFO] Number of equal solutions evaluated for row clusters: {0}".format(len(equal_solutions)))

        # initialize variables
        best_solution = equal_solutions[0]
        tau_c_best_solution = self._compute_emulated_tau_c(lambdas, original_cluster, best_solution)

        for ci in range(len(equal_solutions)):
            evaluated_solution = equal_solutions[ci]
            tau_c_evaluated_solution = self._compute_emulated_tau_c(lambdas, original_cluster, evaluated_solution)
            best_count = 0
            evaluated_count = 0
            for view in range(self._n_views):
                if tau_c_best_solution[view] > tau_c_evaluated_solution[view]:
                    best_count += 1
                elif tau_c_best_solution[view] < tau_c_evaluated_solution[view]:
                    evaluated_count += 1

            if best_count < evaluated_count:
                # the evaluated cluster is the new best
                best_solution = evaluated_solution
                tau_c_best_solution = tau_c_evaluated_solution

        return best_solution

    def _compute_emulated_tau_c(self, lambda_t, source_cluster, destination_cluster):
        """
        Simulate the moving of the selected document from the original_cluster to the evaluated_cluster and computes
         tau_c values for each view.

        :param lambda_t: array-like with shape (n_view, n_col_clusters[view])
        :param source_cluster: int, the row cluster originally containing the document
        :param destination_cluster: int, the row cluster into which we want to move the document
        :return: the value of the compute_tau_c function after the simulated move
        """
        temp_n_row_clusters = self._n_row_clusters
        temp_t = [None] * self._n_views

        if destination_cluster == self._n_row_clusters:
            # the evaluated destination cluster is a new one
            for view in range(self._n_views):
                # clone t[view]
                temp_t[view] = np.copy(self._T[view])
                # subtract the lambdas values from source
                temp_t[view][source_cluster] -= lambda_t[view]
                # add the new row at the end
                temp_t[view] = np.vstack([temp_t[view], lambda_t[view]])

            temp_n_row_clusters += 1
        else:
            # we simulate the move of object x from the original cluster to the destination cluster
            # check that the original cluster is not empty
            is_empty = not self._check_row_clustering_size(source_cluster, 2)

            if is_empty:
                # update the number of cluster
                temp_n_row_clusters -= 1

            for view in range(self._n_views):
                # create the smaller T matrix
                temp_t[view] = np.copy(self._T[view])
                temp_t[view][source_cluster] -= lambda_t[view]
                temp_t[view][destination_cluster] += lambda_t[view]

                if is_empty:
                    # delete the source row
                    temp_t[view] = np.delete(temp_t[view], source_cluster, 0)

                # note: it's not necessary to update the row assignment for this simulation

        return self._compute_tau_c(temp_t, temp_n_row_clusters)

    def _compute_tau_c(self, temp_t=None, temp_n_row_clusters=None):
        """
        Compute a tau_c value for each view

        :param temp_t: the temporary T matrix, if None use self._T
        :param temp_n_row_clusters: the temporary value for n_row_cluster, if None use self._n_row_clusters
        :return:
        """

        if temp_t is None:
            temp_t = self._T
            temp_tot_t_per_rc = self._tot_t_per_rc
            temp_tot_t_per_cc = self._tot_t_per_cc
        else:
            # These values has to be computed if temp_t is different from self._T
            temp_tot_t_per_rc = [None] * self._n_views
            temp_tot_t_per_cc = [None] * self._n_views

            for view in range(self._n_views):
                # compute some support values
                temp_tot_t_per_rc[view] = np.sum(temp_t[view], 1)  # sum per row (axis = 1)
                temp_tot_t_per_cc[view] = np.sum(temp_t[view], 0)  # sum per col (axis = 0)

        result = np.zeros(self._n_views)

        for view in range(self._n_views):
            pow_view = pow(self._tot_per_view[view], 2)
            a1 = np.nansum(np.true_divide(np.power(temp_tot_t_per_cc[view], 2), pow_view))

            denominators = np.multiply(temp_tot_t_per_rc[view], self._tot_per_view[view])
            b1 = np.nansum(np.true_divide(np.power(temp_t[view], 2), denominators[np.newaxis].T))

            result[view] = (b1 - a1) / (1 - a1)

        return result

    def _compute_tau_r(self, view, temp_t_view=None, temp_n_col_clusters=None):
        """
        Compute tau_r value for the specified view.
        Tau_r is the equivalent of tau(X|Yv) where X is the
        partition on rows and Yv is the partition on columns on view v.

        :param view: the view to consider
        :param temp_t_view: the temporary T matrix, if None use self._T
        :param temp_n_col_clusters: the temporary value for n_col_clusters, if None use self._n_col_clusters
        :return:
        """
        if temp_t_view is None:
            temp_t_view = self._T[view]
            tot_t_per_rc = self._tot_t_per_rc[view]
            tot_t_per_cc = self._tot_t_per_cc[view]
        else:
            # compute some support values if t is different from self._T
            tot_t_per_rc = np.sum(temp_t_view, 1)  # sum per row (axis = 1)
            tot_t_per_cc = np.sum(temp_t_view, 0)  # sum per col (axis = 0)

        a = np.nansum(np.true_divide(np.power(tot_t_per_rc, 2), pow(self._tot_per_view[view], 2)))
        b = np.nansum(np.true_divide(np.power(temp_t_view, 2), np.multiply(tot_t_per_cc, self._tot_per_view[view])))

        # aggiunto controllo per a != 1
        if a != 1:
            tau_r = (b - a) / (1 - a)
        else:
            tau_r = 0.0

        return tau_r

    def _check_row_clustering_size(self, row_cluster_id, min_number_of_elements=2):
        """
        Check if the specified row cluster has at least min_number_of_elements elements.
        Returns True if the cluster contains at least the specified number of elements, False otherwise.

        :param row_cluster_id: int, the id of the cluster that contains the element at this moment
        :param min_number_of_elements: int, default 2, the min number of elements that the cluster should have
        :return: boolean, True if the cluster has at least min_number_of_elements elements, False otherwise
        """

        for rc in self._row_assignment:
            if rc == row_cluster_id:
                min_number_of_elements -= 1
            if min_number_of_elements <= 0:
                # stop when the min number is found
                return True

        return False

    def _get_documents_count_per_row_cluster(self):
        """
        Computes an array where each position refers to a row cluster and each cell contains the number of
        rows associated to that cluster.

        :return: the array with the count
        """
        counting = np.zeros(self._n_row_clusters, np.int16)

        for rc in self._row_assignment:
            counting[rc] += 1

        return counting

    def _modify_row_cluster(self, lambda_t, source_rc, destination_rc):
        """
        Updates the T contingency matrix in order to move one element x from a source cluster to a destination cluster.

        :param lambda_t: list of lists, is the difference for T values related to the item we want to move;
                            the list has one element per view, the item in position v is a list of n_col_clusters[v]
                            double values that represents sum of data for the element x grouped by col cluster.
        :param source_rc: int, the id of the original cluster
        :param destination_rc: int, the id of the destination cluster
        :return:
        """
        logging.debug("[INFO] Move element from row cluster {0} to {1}".format(source_rc, destination_rc))
        if destination_rc == self._n_row_clusters:
            logging.debug("[INFO] Create new cluster {0}".format(destination_rc))

            # the destination cluster is a new one
            for view in range(self._n_views):
                # add the row for the new row cluster
                self._T[view] = np.concatenate((self._T[view], [lambda_t[view]]), axis=0)

                # update the source row
                self._T[view][source_rc] -= lambda_t[view]

                # tot_t_per_cc doesn't change
                # tot_t_square_per_cc has to be completely updated
                # we have to remove from tot_t_per_rc[view][source_rc] the value sum(lambda_t[view])
                # we have to add a new element to tot_t_per_rc[view] equal to sum(lambda_t[view])
                lambda_tot = np.sum(lambda_t[view])
                self._tot_t_per_rc[view][source_rc] -= lambda_tot
                self._tot_t_per_rc[view] = np.concatenate((self._tot_t_per_rc[view], [lambda_tot]))
                # for squares we have to recompute the squares for the source_rc
                # and to add the new value to the new cluster
                t_square = np.power(self._T[view], 2)
                self._tot_t_square_per_rc[view][source_rc] = np.sum(t_square[source_rc])
                self._tot_t_square_per_rc[view] = np.concatenate(
                    (self._tot_t_square_per_rc[view],
                     [np.sum(t_square[self._n_row_clusters])]))

                # update sum of squares per col clusters
                self._tot_t_square_per_cc[view] = np.sum(t_square, 0)

            self._n_row_clusters += 1
        else:
            # we move the object x from the original cluster to the destination cluster
            for view in range(self._n_views):
                for cc in range(self._n_col_clusters[view]):
                    self._T[view][source_rc][cc] -= lambda_t[view][cc]
                    self._T[view][destination_rc][cc] += lambda_t[view][cc]

                # tot_t_per_cc doesn't change
                # tot_t_square_per_cc has to be completely updated
                # we have to remove from tot_t_per_rc[view][source_rc] the value sum(lambda_t[view])
                # we have to add a new element to tot_t_per_rc[view] equal to sum(lambda_t[view])
                lambda_tot = np.sum(lambda_t[view])
                t_square = np.power(self._T[view], 2)
                self._tot_t_per_rc[view][source_rc] -= lambda_tot
                self._tot_t_per_rc[view][destination_rc] += lambda_tot
                # for squares we have to recompute the squares for the source_rc
                # and to add the new value to the new cluster
                self._tot_t_square_per_rc[view][source_rc] = np.sum(t_square[source_rc])
                self._tot_t_square_per_rc[view][destination_rc] = np.sum(t_square[destination_rc])
                # completely update tot_square_per_cc
                self._tot_t_square_per_cc[view] = np.sum(t_square, 0)

            # check that the original cluster has at least one remaining element
            is_empty = not self._check_row_clustering_size(source_rc, min_number_of_elements=1)

            if is_empty:
                # compact the contingency matrix
                for view in range(self._n_views):
                    # delete the source_rc row
                    self._T[view] = np.delete(self._T[view], source_rc, 0)
                    # update the total values removing the source cluster item
                    self._tot_t_per_rc[view] = np.delete(self._tot_t_per_rc[view], source_rc)
                    self._tot_t_square_per_rc[view] = np.delete(self._tot_t_square_per_rc[view], source_rc)

                # update the row assignments to reflect the new row cluster ids
                for di in range(self._n_documents):
                    if self._row_assignment[di] > source_rc:
                        self._row_assignment[di] -= 1

                self._n_row_clusters -= 1

        self._performed_moves_on_rows += 1

    def _column_partition_clustering(self, view):
        """
        Perform partitioning of the columns.

        :return:
        """
        selected_cluster = randint(1, self._n_col_clusters[view] * 100) % self._n_col_clusters[view]
        selected_feature = choice(np.where(self._col_assignment[view] == selected_cluster)[0])

        lambdas, sum_lambdas = self._compute_lambdas_per_col(view, selected_feature)

        n_col_clusters_plus_one_empty = self._n_features_per_view[view] if \
            ((self._n_col_clusters[view] + 1) > self._n_features_per_view[view]) else (self._n_col_clusters[view] + 1)

        all_delta_tau_c = self._delta_tau_c_cumulative(view, self._tot_t_per_rc[view], sum_lambdas, lambdas,
                                                       selected_cluster)

        # consider all partitions of the neighborhood of the randomly selected cluster, i.e. partitions where the
        # random_object is moved to another existent cluster or to the empty cluster
        all_delta_tau_c = all_delta_tau_c[:n_col_clusters_plus_one_empty]
        min_delta_tau_c = np.min(all_delta_tau_c)
        equal_solutions = np.where(min_delta_tau_c == all_delta_tau_c)[0]
        e_min = equal_solutions[0]

        if len(equal_solutions) > 1 and min_delta_tau_c != 0:
            # choose the best destination cluster
            e_min = self._find_non_dominated_col(view, equal_solutions, lambdas, selected_cluster)

        go_on_normally = True

        # if the number of clusters is already at minimum value (2) we cant remove another cluster
        if self._n_col_clusters[view] == 2:
            go_on_normally = self._check_col_clustering_size(view, selected_cluster)

        if min_delta_tau_c != 0 and go_on_normally:
            self._col_assignment[view][selected_feature] = e_min
            self._modify_col_cluster(view, lambdas, selected_cluster, e_min)
            self._update_I_and_C()
        else:
            logging.debug("[INFO] Ignored move of {2} from col cluster {0} to {1}".format(selected_cluster,
                                                                                         e_min, selected_feature))

    def _compute_lambdas_per_col(self, view, selected_feature):
        """
        Compute lambda values related to the selected feature element in the specified view.

        In particular:
        * lambdas, array, length = n_row_clusters,
                    contains for each row cluster, the sum of data related to the selected column in the view.
        * sum_lambdas, int,
                    contains the sum of the column data

        :param: view, int, the index of the considered view containing the selected feature
        :param selected_feature: int, the id of the selected element
        :return: a pair (lambdas, sum_lambdas),
                    see the method description for more details
        """
        lambdas = self._sum_feature_data_per_row_cluster(view, selected_feature)
        sum_lambdas = np.sum(lambdas)

        return lambdas, sum_lambdas

    def _sum_feature_data_per_row_cluster(self, view, feature):
        """
        Fixed a feature (column of a view matrix), computes the sum of all values of each document for that feature
        group by row cluster.

        :param view: the view the feature belongs to
        :param feature: the feature to consider
        :return: array of n_row_clusters elements, each one is the sum of the feature column of documents belonging to
                    the row cluster
        """
        sum_f = np.zeros(self._n_row_clusters)

        if issparse(self._dataset[view]):
            col = self._csc_dataset[view].getcol(feature)
            for r in col.indices:
                # for each non zero row
                sum_f[self._row_assignment[r]] += col[r, 0]
        else:
            for r in range(self._n_documents):
                sum_f[self._row_assignment[r]] += self._dataset[view][r][feature]

        return sum_f

    def _delta_tau_c(self, view, tot_t_per_rc, sum_lambdas, lambdas, original_cluster, destination_cluster):

        if original_cluster == destination_cluster:
            return 0.0

        if destination_cluster == self._n_col_clusters[view]:
            is_singleton = not self._check_col_clustering_size(view, original_cluster)
            if is_singleton:
                return 0.0

        t_cc_orig = self._T[view][:,original_cluster]

        if destination_cluster == self._n_col_clusters[view]:
            x = np.nansum(np.multiply(np.true_divide(lambdas, tot_t_per_rc), np.subtract(t_cc_orig, lambdas)))
            y = np.nansum(np.subtract(lambdas, t_cc_orig))
        else:
            t_cc_dest = self._T[view][:, destination_cluster]
            x = np.nansum(np.multiply(np.true_divide(lambdas, tot_t_per_rc), np.subtract(np.subtract(t_cc_orig, lambdas),
                                                                                    t_cc_dest)))
            y = np.nansum(np.add(np.subtract(lambdas, t_cc_orig), t_cc_dest))

        x *= self._two_divided_by_tot_per_view[view]
        y *= ((2 * sum_lambdas) / self._square_tot_per_view[view])

        delta_tau_c = 0.0
        if (self._Ic[view] * (self._Ic[view] - y)) != 0:
            delta_tau_c = ((self._Ic[view] * x) + (self._Cc[view] * y)) / (self._Ic[view] * (self._Ic[view] - y))

        return delta_tau_c

    def _delta_tau_c_cumulative(self, view, tot_t_per_rc, sum_lambdas, lambdas, original_cluster):

        computed_taus = [0.0] * (self._n_col_clusters[view] + 1)
        t_cc_orig = self._T[view][:,original_cluster]
        division_one = np.true_divide(lambdas, tot_t_per_rc)
        subtraction_one = np.subtract(t_cc_orig, lambdas)
        subtraction_two = np.subtract(lambdas, t_cc_orig)

        for evaluated_cluster in range(self._n_col_clusters[view] + 1):

            if evaluated_cluster == self._n_col_clusters[view]:
                x = np.nansum(np.multiply(division_one, subtraction_one))
                y = np.nansum(subtraction_two)
            else:
                t_cc_dest = self._T[view][:,evaluated_cluster]
                x = np.nansum(np.multiply(division_one, np.subtract(subtraction_one, t_cc_dest)))
                y = np.nansum(np.add(subtraction_two, t_cc_dest))

            x *= self._two_divided_by_tot_per_view[view]
            y *= ((2 * sum_lambdas) / self._square_tot_per_view[view])

            delta_tau_c = 0.0
            if (self._Ic[view] * (self._Ic[view] - y)) != 0:
                delta_tau_c = ((self._Ic[view] * x) + (self._Cc[view] * y)) / (self._Ic[view] * (self._Ic[view] - y))

            computed_taus[evaluated_cluster] = delta_tau_c

        # set to zero the non-move element
        computed_taus[original_cluster] = 0.0

        # check if the source is a singleton and set to zero the empty-cluster move
        is_singleton = not self._check_col_clustering_size(view, original_cluster)
        if is_singleton:
            computed_taus[-1] = 0.0

        return computed_taus

    def _find_non_dominated_col(self, view, col_clusters_to_evaluate, lambdas, source_cluster):
        """
        Considers all specified column clusters c, emulates the move of the object from the source_cluster to c and
          computes the tau_r value for the obtained combination of row and col clustering. Returns the col cluster index
          that minimize the tau_r value.
          
        :param view: int, the view to consider
        :param col_clusters_to_evaluate: array, the list of col clusters to consider
        :param lambdas: the lambdas value related to the feature to move from the source cluster
        :param source_cluster: the cluster that originally contains the feature to move
        :return:
        """
        logging.debug(
            "[INFO] Number of equal solutions evaluated for col clusters: {0}".format(len(col_clusters_to_evaluate)))

        best_solution = col_clusters_to_evaluate[0]
        tau_r_best = self._compute_emulated_tau_r(view, lambdas, source_cluster, best_solution)

        for ci in range(len(col_clusters_to_evaluate)):
            evaluated_solution = col_clusters_to_evaluate[ci]
            tau_r_evaluated = self._compute_emulated_tau_r(view, lambdas, source_cluster, evaluated_solution)

            if tau_r_best < tau_r_evaluated:
                best_solution = evaluated_solution
                tau_r_best = tau_r_evaluated

        return best_solution

    def _check_col_clustering_size(self, view, col_cluster_id, min_number_of_elements=2):
        """
        Checks if the specified col cluster has at least min_number_of_elements elements.
        Return True if the cluster contains at least the specified number of elements, False otherwise.

        :param view: int, the view to consider
        :param col_cluster_id: int, the id of the column cluster to consider
        :param min_number_of_elements: int, default 2, the min number of elements that the cluster should have
        :return: boolean, True if the cluster has at least min_number_of_elements elements, False otherwise
        """

        for cc in self._col_assignment[view]:
            if cc == col_cluster_id:
                min_number_of_elements -= 1
            if min_number_of_elements <= 0:
                # stop when the min number is found
                return True

        return False

    def _get_features_count_per_col_cluster(self, view):
        """
        Counts the number of features for each column cluster of a specific view

        :param view: the view to consider
        :return: array of n_col_clusters[view] elements, each one is the count of objects.
        """
        counting = np.zeros(self._n_col_clusters[view], np.int16)
        for f in range(self._n_features_per_view[view]):
            counting[self._col_assignment[view][f]] += 1

        return counting

    def _modify_col_cluster(self, view, lambda_t, source_cc, destination_cc):
        """
        Updates the T contingency matrix in order to move one feature f from a source column cluster, to a destination
        column cluster.

        :param view: the considered view
        :param lambda_t: array, length = n_row_clusters
        :param source_cc: int, the column cluster containing the feature
        :param destination_cc: int, the column cluster into which the feature has to be moved
        :return:
        """
        logging.debug("[INFO] Move element from col cluster {0} to {1}".format(source_cc, destination_cc))
        if destination_cc == self._n_col_clusters[view]:
            # the destination col cluster is an empty one
            logging.debug("[INFO] Create new col cluster {0}".format(destination_cc))

            # update the source cluster values
            for rc in range(self._n_row_clusters):
                self._T[view][rc][source_cc] -= lambda_t[rc]

            # append the new column
            self._T[view] = np.append(self._T[view], lambda_t[np.newaxis].T, axis=1)

            # tot_t_per_rc doesn't change
            # tot_t_square_per_rc has to be completely updated for the considered view
            # tot_t_per_cc and tot_t_square_per_cc change only for the considered view
            lambda_tot = np.sum(lambda_t)
            t_square = np.power(self._T[view], 2)
            self._tot_t_per_cc[view][source_cc] -= lambda_tot
            self._tot_t_per_cc[view] = np.concatenate((self._tot_t_per_cc[view], [lambda_tot]))
            self._tot_t_square_per_cc[view][source_cc] = np.sum(t_square[:, source_cc])
            self._tot_t_square_per_cc[view] = np.concatenate((self._tot_t_square_per_cc[view],
                                                              [np.sum(t_square[:, self._n_col_clusters[view]])]))

            self._tot_t_square_per_rc[view] = np.sum(t_square, 1)

            self._n_col_clusters[view] += 1

        else:
            # the destination cluster is an existent one
            self._T[view][:, source_cc] = np.subtract(self._T[view][:, source_cc], lambda_t)
            self._T[view][:, destination_cc] = np.add(self._T[view][:, destination_cc], lambda_t)

            # update totals per column cluster
            # tot_t_per_rc doesn't change
            # tot_t_square_per_rc has to be completely updated
            # tot_t_per_cc and tot_t_square_per_cc change only for the considered view
            lambda_tot = np.sum(lambda_t)
            t_square = np.power(self._T[view], 2)
            self._tot_t_per_cc[view][source_cc] -= lambda_tot
            self._tot_t_per_cc[view][destination_cc] += lambda_tot
            self._tot_t_square_per_cc[view][source_cc] = np.sum(t_square[:, source_cc])
            self._tot_t_square_per_cc[view][destination_cc] = np.sum(t_square[:, destination_cc])
            self._tot_t_square_per_rc[view] = np.sum(t_square, 1)

            is_empty = not self._check_col_clustering_size(view, source_cc, min_number_of_elements=1)

            if is_empty:
                # compact the contingency matrix to remove the empty cluster
                self._T[view] = np.delete(self._T[view], source_cc, 1)

                # compact the totals arrays
                self._tot_t_per_cc[view] = np.delete(self._tot_t_per_cc[view], source_cc)
                self._tot_t_square_per_cc[view] = np.delete(self._tot_t_square_per_cc[view], source_cc)

                # update the column assignment to reflect the new numbering of clusters
                for f in range(self._n_features_per_view[view]):
                    if self._col_assignment[view][f] > source_cc:
                        self._col_assignment[view][f] -= 1

                # self._T[view] = new_t
                self._n_col_clusters[view] -= 1

        self._performed_moves_on_cols[view] += 1

    def _compute_emulated_tau_r(self, view, lambdas, source_cc, destination_cc):
        """
        Simulate the moving of the selected feature from the source_cluster to the destination_cluster (considering
        columns) partition computes tau_r value for the considered view.

        :param view: the considered view
        :param lambdas: array of n_row_clusters elements, each cell contains the lambda T values for element x for
                            the considered row cluster
        :param source_cc: int, the column cluster that contains the feature x
        :param destination_cc: int, the column cluster into which we want to move the feature x
        :return: the value of compute_tau_r function after the simulated move
        """
        temp_n_col_clusters = self._n_col_clusters[view]
        temp_t_v = np.copy(self._T[view])

        if destination_cc == temp_n_col_clusters:
            temp_t_v[:, source_cc] = np.subtract(temp_t_v[:, source_cc], lambdas)
            # append the new column
            temp_t_v = np.append(temp_t_v, lambdas[np.newaxis].T, axis=1)

            temp_n_col_clusters += 1
        else:
            # the destination is an already existent cluster
            # check that the original cluster will not be empty
            is_empty = not self._check_col_clustering_size(view, source_cc, 2)

            temp_t_v[:, source_cc] = np.subtract(temp_t_v[:, source_cc], lambdas)
            temp_t_v[:, destination_cc] = np.add(temp_t_v[:, destination_cc], lambdas)

            if is_empty:
                # delete the source column
                temp_n_col_clusters -= 1
                temp_t_v = np.delete(temp_t_v, source_cc, 1)

        return self._compute_tau_r(view, temp_t_v, temp_n_col_clusters)


def get_element(matrix, row, col):
    """
    Get the element in position <row,col> of the matrix.

    :param matrix: the matrix
    :param row: the row position
    :param col: the column position
    """

    if issparse(matrix):
        return matrix[row, col]
    else:
        return matrix[row][col]
