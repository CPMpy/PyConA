import os
import time
import warnings


class Metrics:
    """
    A class to track and manage various metrics related to queries and constraints.
    """

    def __init__(self):
        """
        Initialize the Metrics object with default values.
        """
        self.cl = 0

        self.total_queries = 0
        self.membership_queries_count = 0
        self.top_lvl_queries = 0
        self.generated_queries = 0
        self.findscope_queries = 0
        self.findc_queries = 0

        self.average_size_queries = 0
        self.total_size_queries = 0

        self.start_time = time.time()
        self.start_time_query = time.time()

        self.average_waiting_time = 0
        self.max_waiting_time = 0

        self.generation_time = 0
        self.average_generation_time = 0

        self.total_time = 0

        # Recommendation and Generalization queries
        self.recommendation_queries_count = 0
        self.positive_recommendation_queries = 0
        self.generalization_queries_count = 0
        self.positive_generalization_queries = 0

        # Query progress per constraint learned statistics
        self.cl_progress = []
        self.total_queries_progress = []
        self.membership_queries_progress = []
        self.recommendation_queries_progress = []
        self.generalization_queries_progress = []

        self.converged = 1

        self.metrics_dict = None
        self.metrics_short_dict = None
        self.queries_dict = None

    def increase_membership_queries_count(self, amount=1):
        """
        Increase the count of membership queries.

        :param amount: The amount to increase by, default is 1.
        """
        self.membership_queries_count += amount
        self.total_queries += amount

    def increase_top_queries(self, amount=1):
        """
        Increase the count of top-level queries.

        :param amount: The amount to increase by, default is 1.
        """
        self.top_lvl_queries += amount

    def increase_generated_queries(self, amount=1):
        """
        Increase the count of generated queries.

        :param amount: The amount to increase by, default is 1.
        """
        self.generated_queries += amount

    def increase_findscope_queries(self, amount=1):
        """
        Increase the count of find-scope queries.

        :param amount: The amount to increase by, default is 1.
        """
        self.findscope_queries += amount

    def increase_findc_queries(self, amount=1):
        """
        Increase the count of find-c queries.

        :param amount: The amount to increase by, default is 1.
        """
        self.findc_queries += amount

    def increase_generation_time(self, amount):
        """
        Increase the total generation time.

        :param amount: The amount to increase by.
        """
        self.generation_time += amount

    def increase_queries_size(self, amount):
        """
        Increase the total size of queries.

        :param amount: The amount to increase by.
        """
        self.total_size_queries += amount

    def aggreagate_max_waiting_time(self, max2):
        """
        Aggregate the maximum waiting time.

        :param max2: The new waiting time to compare with the current maximum.
        """
        if self.max_waiting_time < max2:
            self.max_waiting_time = max2

    def increase_recommendation_queries_count(self, amount=1):
        """
        Increase the count of recommendation queries.

        :param amount: The amount to increase by, default is 1.
        """
        self.recommendation_queries_count += amount
        self.total_queries += amount

    def increase_generalization_queries_count(self, amount=1):
        """
        Increase the count of generalization queries.

        :param amount: The amount to increase by, default is 1.
        """
        self.generalization_queries_count += amount
        self.total_queries += amount

    def store_queries_progress(self):
        """
        Store the progress of various query counts.
        """
        self.cl_progress.append(self.cl)
        self.total_queries_progress.append(self.total_queries)
        self.membership_queries_progress.append(self.membership_queries_count)
        self.recommendation_queries_progress.append(self.recommendation_queries_count)
        self.generalization_queries_progress.append(self.generalization_queries_count)

    def asked_query(self):
        """
        Measure the waiting time of the user from the previous query.
        """
        end_time_query = time.time()

        # To measure the maximum waiting time for a query
        waiting_time = end_time_query - self.start_time_query
        self.aggreagate_max_waiting_time(waiting_time)
        self.start_time_query = time.time()  # to measure the maximum waiting time for a query

    def finalize_statistics(self):
        """
        Finalize and calculate the statistics of the metrics.
        """
        self.total_time = time.time() - self.start_time

        self.average_size_queries = round(self.total_size_queries / self.membership_queries_count,
                                          4) if self.membership_queries_count > 0 else 0
        self.average_generation_time = round(self.generation_time / self.generated_queries,
                                             4) if self.generated_queries > 0 else 0
        self.average_waiting_time = self.total_time / self.membership_queries_count \
            if self.membership_queries_count > 0 else 0

        self.total_queries = self.membership_queries_count + self.recommendation_queries_count + \
                             self.generalization_queries_count

        self.metrics_dict = {
            "CL": self.cl,
            "tot_q": self.total_queries,
            "memb_q": self.membership_queries_count,
            "rec_q": self.recommendation_queries_count,
            "gen_q": self.generalization_queries_count,
            "top_lvl_q": self.top_lvl_queries,
            "qgen_q": self.generated_queries,
            "tfs_q": self.findscope_queries,
            "tfc_q": self.findc_queries,
            "avg_q_size": self.average_size_queries,
            "avg_gen_time": self.average_generation_time,
            "avg_t": round(self.average_waiting_time, 4),
            "max_t": round(self.max_waiting_time, 4),
            "tot_t": round(self.total_time, 4),
            "conv": self.converged
        }

        self.metrics_short_dict = {
            "CL": self.cl,
            "tot_q": self.total_queries,
            "top_lvl_q": self.top_lvl_queries,
            "tfs_q": self.findscope_queries,
            "tfc_q": self.findc_queries,
            "avg_q_size": self.average_size_queries,
            "avg_gen_time": self.average_generation_time,
            "avg_t": round(self.average_waiting_time, 4),
            "max_t": round(self.max_waiting_time, 4),
            "tot_t": round(self.total_time, 4),
            "conv": self.converged
        }

        self.store_queries_progress()

        self.queries_dict = {
            "cl": self.cl_progress,
            "tot_q": self.total_queries_progress,
            "memb_q": self.membership_queries_progress,
            "rec_q": self.recommendation_queries_progress,
            "gen_q": self.generalization_queries_progress,
        }

    def print_statistics(self):
        """
        Print the statistics of the metrics.
        """

        try:
            import pandas as pd
            results_df = pd.DataFrame([self.metrics_dict])
            print(results_df.to_string(index=False))
        except ImportError:
            print(self.metrics_dict)

    @property
    def statistics(self):
        """
        get the statistics of the metrics.
        """

        try:
            import pandas as pd
            results_df = pd.DataFrame([self.metrics_dict])
            return results_df
        except ImportError:
            print(self.metrics_dict)

    def print_short_statistics(self):
        """
        Print the short statistics.
        """

        try:
            import pandas as pd
            results_df = pd.DataFrame([self.metrics_short_dict])
            print(results_df.to_string(index=False))
        except ImportError:
            print(self.metrics_dict)

    @property
    def short_statistics(self):
        """
        Get the short statistics.
        """
        try:
            import pandas as pd
            results_df = pd.DataFrame([self.metrics_short_dict])
            return results_df
        except ImportError:
            return self.metrics_dict

    def write_to_file(self, filename):
        """
        Write the results to a CSV file.

        :param filename: The name of the file to write to.
        """
        try:
            import pandas as pd
            results_df = pd.DataFrame([self.metrics_dict])

            file_exists = os.path.isfile(filename)
            if not file_exists:
                results_df.to_csv(filename, index=False)
            else:
                results_df.to_csv(filename, mode='a', header=False, index=False)
        except ImportError:
            warnings.warn("pandas library not installed! Cannot write to file")

    def write_queries_to_file(self, filename):
        """
        Write the query progress to a CSV file.

        :param filename: The name of the file to write to.
        """
        try:
            import pandas as pd
            queries_df = pd.DataFrame(self.queries_dict)
            queries_df.to_csv(filename, index=False)
        except ImportError:
            warnings.warn("pandas library not installed! Cannot write to file")
