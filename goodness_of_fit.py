import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns


class Test:
    """
    This class will preform the following goodness-of-fit tests agains a
    Bellyband dataset:
        - Chi Squared
        - KS

    Args:
        filename (str): The name of the CSV file containing the dataset
        epc (str): The id of the tag to be tested
        column (str, optional): The column to be tested. Defaults to 'rssi'
        dists (list, optional): The distributions to be tested. This can be
            any in the scipy.stats library. Defaults to ['rayleigh'].
        plot (bool, optional): Whether or not to plot a histogram of the data
            with an expected KDE plot for each distribution.
        r (list, optional): The start and end time to be used in seconds.
    """

    def __init__(self, filename, epc, column='rssi', dists=['rayleigh'],
                 plot=False, r=None):
        self.cols = [column, 'epc96', 'relative_timestamp']
        self.dists = dists
        self.epc = epc
        self.filename = filename
        self.plot = plot
        self._range = r

        self._get_data()

    @property
    def range(self):
        """list: Contains beginning and ending timestamps in microseconds."""

        if self._range:
            return [int(x) * 1000000 for x in self._range]
        else:
            return False

    def chi_square(self, E):
        """
        Preform Chi Squared test and print results.

        Args:
            E (numpy NDArray): The expected values.
        """

        # build histograms to be passed to chisqaure
        hist_o = np.histogram(self.obs, self._bins())[0]
        hist_e = np.histogram(E, self._bins())[0]

        s = []
        for i, e in enumerate(hist_e):
            if e == 0:
                s.append(i)

        hist_o = np.delete(hist_o, s)
        hist_e = np.delete(hist_e, s)

        print(f'observed: {hist_o}\nexpected: {hist_e}')

        chisq, chisq_p = st.chisquare(hist_o, hist_e)

        print(f'---- chi_square -----\n' +
              f'Chisq: {chisq}\np: {chisq_p}')

    def ks(self, dist):
        """
        Preform KS test and print results.

        Args:
            dist (callable): A callable representing the distribution used to
                calculate the CDF.
        """

        D, D_p = st.kstest(self.obs, dist.name, args=([*dist.fit(self.obs)]))

        print(f'---- ks_test ----\n' +
              f'D: {D}\np: {D_p}')

    def run_tests(self):
        """
        Run both Chi Squared and KS tests for each distribution.
        """

        sns.set()

        for d in self.dists:

            # get the distribution from the scipy stats library
            dist = vars(st)[d]
            print(f'\n\n#### {dist.name} ####')

            E = self._expected(dist)
            self.chi_square(E)
            self.ks(dist)

            if self.plot:
                self.generate_plot(dist.name, E)

    def generate_plot(self, name, E):
        """
        Generate a histogram of the observed values and the expected KDE of a
        distribution fit to those values.

        Args:
            name (str): The name of the distribution.
            E (numpy NDArray): The expected values.
        """

        bins = self._bins()
        sns.distplot(self.obs, bins=bins, kde=False,
                     norm_hist=True, hist_kws={'label': self.cols[0]})
        sns.distplot(E, bins=bins, kde_kws={'label': 'Expected KDE'},
                     hist_kws={'label': 'Expected'})

        plt.title(name)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_column(filename, c, r=None):
        """
        Generate a scatter plot of a column's data while labelling the tag ID
        of each point. Each ID will be printed for copying.

        Args:
            filename (str): The name of the CSV file.
            c (str): The name of the column to be plotted.
            r (list, optional): Contains the start and end time in seconds.
        """

        sns.set()

        df = pd.read_csv(filename, usecols=[c, 'epc96', 'relative_timestamp'])
        df['relative_timestamp'] = df['relative_timestamp'] / 1000000

        if r:
            df = df[int(r[0]) < df['relative_timestamp']]
            df = df[int(r[1]) > df['relative_timestamp']]

        sns.relplot(x='relative_timestamp', y=c, hue='epc96', data=df)
        print('#### tags:', pd.Series(df['epc96']).unique())

        plt.show()

    def _expected(self, dist):
        E = dist.rvs(*dist.fit(self.obs), self.obs.size)
        return E

    def _bins(self):
        if np.unique(self.obs).size <= 20:
            bins = np.unique(self.obs).size + 1
            bins = np.add(np.arange(0, bins), np.min(self.obs))
        else:
            bins = np.linspace(np.min(self.obs), np.max(self.obs), 20)

        return bins

    def _get_data(self):
        self.df = pd.read_csv(self.filename, usecols=self.cols)

        # check that the epc exists in the dataframe
        try:
            assert self.epc in self.df['epc96'].values
        except AssertionError:
            raise AssertionError('Tag ID not found in data')

        if self.range:
            self.df = self.df[self.range[0] < self.df['relative_timestamp']]
            self.df = self.df[self.range[1] > self.df['relative_timestamp']]

        x = self.df[self.cols[0]]
        self.obs = x[self.df['epc96'] == self.epc].values


if __name__ == '__main__':
    desc = ('''Test the goodness of fit of an RSSI signal to a given
            distribution using chi sqaured and ks tests. The ID of the tag
            must be passed with -e or --epc. Without this argument the signal
            will be plotted with the IDs labelled.''')
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('filename', help='The CSV file containing the signal.')
    parser.add_argument('-c', '--column',
                        action='store',
                        default='rssi',
                        help='''The column to read the signal from. Defaults to
                                \'rssi\'.''')
    parser.add_argument('-d', '--distributions',
                        action='store',
                        nargs='+',
                        default=['rayleigh'],
                        help='''The distributions to compare to. Defaults to
                                \'rayleigh\'.'''
                        )
    parser.add_argument('-e', '--epc',
                        action='store',
                        help='The ID of the tag to be tested.')
    parser.add_argument('-p', '--plot',
                        action='store_true',
                        default=False,
                        help='Generate plots inline with each test.'
                        )
    parser.add_argument('-r', '--range',
                        action='store',
                        nargs=2,
                        help='The start and end time to be used in seconds.')

    args = parser.parse_args()

    if args.epc:
        test = Test(args.filename, args.epc, args.column, args.distributions,
                    args.plot, args.range)
        test.run_tests()
    else:
        Test.plot_column(args.filename, args.column, args.range)
