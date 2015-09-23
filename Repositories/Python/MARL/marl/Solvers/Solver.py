"""
The Solver class and functions, defining the solving logic.
"""
from marl.Storage import *


class Solver(object):
    """
    This is the base class for all solvers, i.e. all policy update algorithms.
    """
    def __init__(self, bla=None):
        self.temp_storage = {'Data': ['?']}

    def init_temp_storage(self, start, domain, options):
        print(
            'Every method must define init_temp_storage(self,start,domain,options)' +
            'to return a dictionary of information the method will need for later use' +
            'plus the work necessary to obtain that information.')

        self.temp_storage['Data'][-1] = start

        return self.temp_storage

    def book_keeping(self, temp_data):
        for item in self.temp_storage:
            self.temp_storage[item].pop(0)
            self.temp_storage[item].append(temp_data[item])

    def update(self, record):
        print(
            'Every method must define update(self,Record) to return' +
            'a data packet containing any information required for subsequent' +
            'iterations by the method as well as the work done to obtain that information.')

        temp_data = record.TempStorage
        self.book_keeping(temp_data)

        return self.temp_storage



def solve(start, method, domain, options):
    raise NotImplementedError


def solve_repeated_games(start, method, domain, options):
    # Check Validity of options
    options.check_options(method, domain)

    # Create Storage Object for Record Keeping
    record = [Storage(start, domain, method, options) for _ in range(domain.players)]
    # Begin Solving
    for i_rec in range(domain.players):
        # if the update is stochastic - play :-)
        while not options.term.is_terminal(record[i_rec]):
            temp_storage = method.update(i_rec, record[i_rec])  # should also report projections

            # Record update Stats
            record[i_rec].book_keeping(temp_storage)


def solve_sequential_games(start, method, domain, options):
    pass


def solve_analytical_games(start, method, domain, options):
    # Check Validity of options
    options.check_options(method, domain)

    # Create Storage Object for Record Keeping
    record = Storage(start, domain, method, options)

    # Begin Solving
    while not options.term.is_terminal(record):
        temp_storage = method.update(record)  # should also report projections

        # Record update Stats
        record.book_keeping(temp_storage)
