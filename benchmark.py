import argparse
import csv
import datetime
import os
import timeit
from enum import Enum
from typing import Dict, Any, Optional, Union

from encoder.SAT_encoder import SATEncoder
from encoder.SAT_model import NUMBER_OF_LITERAL
from encoder.problem import Problem


class EncoderType(Enum):
    THESIS_2022 = 1
    STAIRCASE = 2
    LIA = 3
    NEW_STAIRCASE = 4


class InfoAttribute(Enum):
    LB = 0
    UB = 1
    NUM_VAR = 2
    NUM_CLAUSE = 3
    NUM_CONSISTENCY_ClAUSE = 4
    NUM_PB_CLAUSE = 5
    FEASIBLE = 6
    MAKE_SPAN = 7
    TOTAL_SOLVING_TIME = 8
    OPTIMIZED = 9
    ZERO_LITS = 10
    ONE_LITS = 11
    TWO_LITS = 12
    THREE_LITS = 13
    FOUR_LITS = 14
    MORE_THAN_FOUR_LITS = 15
    MORE_THAN_TEN_LITS = 16


class BenchmarkLogger:
    """Handles logging for benchmark operations."""

    def __init__(self, log_file: str, verbose: bool = False):
        self.log_file = log_file
        self.verbose = verbose

        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(self, message: str):
        """Write a message to the log file and optionally print to console."""
        timestamp = datetime.datetime.now()
        with open(self.log_file, "a+") as f:
            f.write(f'{timestamp} {message}\n')
        if self.verbose:
            print(f'[{timestamp}] {message}')


class ResultManager:
    """Manages result output and solution files."""

    def __init__(self, output_path: str, encoder_type: EncoderType, show_solution: bool = False):
        self.output_path = output_path
        self.encoder_type = encoder_type
        self.show_solution = show_solution
        self.solution_file = None

        # Create result directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Initialize output file with headers
        self._initialize_output_file()

        # Create solution file if needed
        if show_solution:
            solution_dir = 'solution'
            os.makedirs(solution_dir, exist_ok=True)
            self.solution_file = f'{solution_dir}/{os.path.basename(output_path).replace(".csv", ".sol")}'
            with open(self.solution_file, "a+") as f:
                f.write('file_name,make_span,solution\n')

    def _initialize_output_file(self):
        """Create the output file with appropriate headers based on encoder type."""
        with open(self.output_path, "a+") as f:
            if self.encoder_type in [EncoderType.STAIRCASE, EncoderType.THESIS_2022,
                                     EncoderType.NEW_STAIRCASE]:
                f.write(
                    'file_name,'
                    'lb,'
                    'ub,'
                    'num_var,'
                    'num_clause,'
                    'num_consistency_clause,'
                    'num_PB_clause,'
                    'zero_lits,'
                    'one_lits,'
                    'two_lits,'
                    'three_lits,'
                    'four_lits,'
                    'five_to_ten,'
                    'more_than_ten,'
                    'feasible,'
                    'make_span,'
                    'total_solving_time,'
                    'optimized,'
                    'timeout\n')
            elif self.encoder_type == EncoderType.LIA:
                f.write(
                    'file_name,'
                    'lb,'
                    'ub,'
                    'feasible,'
                    'make_span,'
                    'total_solving_time,'
                    'optimized,'
                    'timeout\n')

    def save_result(self, result_info: Dict[str, Any]):
        """Save benchmark results to the output file."""
        with open(self.output_path, "a+") as f:
            if self.encoder_type in [EncoderType.STAIRCASE, EncoderType.THESIS_2022,
                                     EncoderType.NEW_STAIRCASE]:
                f.write(
                    f'{result_info["file_name"]},'
                    f'{result_info["lb"]},'
                    f'{result_info["ub"]},'
                    f'{result_info["num_var"]},'
                    f'{result_info["num_clause"]},'
                    f'{result_info["num_consistency_clause"]},'
                    f'{result_info["num_PB_clause"]},'
                    f'{result_info["zero_lits"]},'
                    f'{result_info["one_lits"]},'
                    f'{result_info["two_lits"]},'
                    f'{result_info["three_lits"]},'
                    f'{result_info["four_lits"]},'
                    f'{result_info["five_to_ten"]},'
                    f'{result_info["more_than_ten"]},'
                    f'{int(result_info["feasible"])},'
                    f'{result_info["make_span"]},'
                    f'{result_info["total_solving_time"]},'
                    f'{int(result_info["optimized"])},'
                    f'{int(result_info["timeout"])}\n')
            elif self.encoder_type == EncoderType.LIA:
                f.write(
                    f'{result_info["file_name"]},'
                    f'{result_info["lb"]},'
                    f'{result_info["ub"]},'
                    f'{int(result_info["feasible"])},'
                    f'{result_info["make_span"]},'
                    f'{result_info["total_solving_time"]},'
                    f'{int(result_info["optimized"])},'
                    f'{int(result_info["timeout"])}\n')

    def save_solution(self, file_name: str, solution: str):
        """Save solution to solution file if enabled."""
        if self.show_solution and self.solution_file:
            with open(self.solution_file, "a+") as f:
                f.write(f'{file_name}\t{solution}\n')


class BenchmarkRunner:
    """Runs the benchmark process for a dataset using specified encoder."""

    def __init__(self, data_set_name: str, encoder_type: EncoderType,
                 timeout: Optional[int], verify: bool = False, verbose: bool = False,
                 show_solution: bool = False):
        self.data_set_name = data_set_name
        self.encoder_type = encoder_type
        self.timeout = timeout
        self.verify = verify
        self.verbose = verbose
        self.show_solution = show_solution

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_name = f'result/{data_set_name}_{encoder_type.name}_{timestamp}.csv'
        log_file = f'log/{data_set_name}_{encoder_type.name}_{timestamp}.log'

        self.logger = BenchmarkLogger(log_file, verbose)
        self.result_manager = ResultManager(output_name, encoder_type, show_solution)

    def create_encoder(self, problem: Problem, upper_bound: int) -> Union[SATEncoder, 'LIAEncoder']:
        """Factory method to create the appropriate encoder."""
        if self.encoder_type == EncoderType.STAIRCASE:
            from encoder.staircase import StaircaseSATEncoder
            return StaircaseSATEncoder(problem, upper_bound, self.timeout, self.verify)
        elif self.encoder_type == EncoderType.THESIS_2022:
            from encoder.thesis_2022 import Thesis2022SATEncoder
            return Thesis2022SATEncoder(problem, upper_bound, self.timeout, self.verify)
        elif self.encoder_type == EncoderType.LIA:
            from encoder.LIA_encoder import LIAEncoder
            return LIAEncoder(problem, upper_bound, self.timeout, self.verify)
        elif self.encoder_type == EncoderType.NEW_STAIRCASE:
            from encoder.staircase_new import NewStaircaseSATEncoder
            return NewStaircaseSATEncoder(problem, upper_bound, self.timeout, self.verify)
        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")

    def create_result_info(self, file_name: str, lb: int, ub: int, encoder) -> Dict[str, Any]:
        """Create initial result info dictionary based on encoder type."""
        if isinstance(encoder, SATEncoder):
            return {
                'file_name': file_name,
                'lb': lb,
                'ub': ub,
                'num_var': encoder.sat_model.number_of_variable,
                'num_clause': encoder.sat_model.number_of_clause,
                'num_consistency_clause': encoder.sat_model.number_of_consistency_clause,
                'num_PB_clause': encoder.sat_model.number_of_PB_clause,
                'zero_lits': encoder.sat_model.number_of_literal[NUMBER_OF_LITERAL.ZERO.value],
                'one_lits': encoder.sat_model.number_of_literal[NUMBER_OF_LITERAL.ONE.value],
                'two_lits': encoder.sat_model.number_of_literal[NUMBER_OF_LITERAL.TWO.value],
                'three_lits': encoder.sat_model.number_of_literal[NUMBER_OF_LITERAL.THREE.value],
                'four_lits': encoder.sat_model.number_of_literal[NUMBER_OF_LITERAL.FOUR.value],
                'five_to_ten': encoder.sat_model.number_of_literal[
                    NUMBER_OF_LITERAL.FIVE_TO_TEN.value],
                'more_than_ten': encoder.sat_model.number_of_literal[
                    NUMBER_OF_LITERAL.MORE_THAN_TEN.value],
                'feasible': False,
                'make_span': 0,
                'total_solving_time': 0,
                'optimized': False,
                'timeout': False
            }
        else:  # LIAEncoder
            return {
                'file_name': file_name,
                'lb': lb,
                'ub': ub,
                'feasible': False,
                'make_span': 0,
                'total_solving_time': 0,
                'optimized': False,
                'timeout': False
            }

    def process_instance(self, file_name: str, lb: int, ub: int):
        """Process a single problem instance."""
        self.logger.log(f'benchmark for {file_name} using {self.encoder_type.name} started')

        # Create problem and encoder
        problem = Problem(file_name)
        encoder = self.create_encoder(problem, ub)

        # Encode the problem
        encoder.encode()

        # Create initial result info
        result_info = self.create_result_info(file_name, lb, ub, encoder)

        # Solve the problem
        self._solve_and_optimize(encoder, result_info, file_name)

        # Save results
        self.result_manager.save_result(result_info)

        return result_info

    def _solve_and_optimize(self, encoder, result_info: Dict[str, Any], file_name: str):
        """Solve the problem and optimize the makespan."""
        # Initial solve
        sat = encoder.solve()

        if sat is None:
            result_info['timeout'] = True
            self.logger.log(f'{file_name} timeout while checking makespan: '
                            f'{encoder.makespan} with total running time: {round(encoder.time_used, 5)}')
            if self.show_solution and result_info['feasible']:
                self.result_manager.save_solution(file_name, encoder.get_solution())
            return

        if sat:
            # Solution found - verify and update result info
            encoder.verify()
            result_info['feasible'] = True
            result_info['make_span'] = encoder.makespan
            result_info['total_solving_time'] = round(encoder.time_used, 5)

            self.logger.log(f'{file_name} feasible with makespan: '
                            f'{encoder.makespan} with total running time: {round(encoder.time_used, 5)}')

            # Try to optimize by decreasing makespan
            while sat and encoder.makespan > result_info['lb']:
                encoder.decrease_makespan()
                sat = encoder.solve()

                if sat is None:
                    # Timeout during optimization
                    result_info['timeout'] = True
                    self.logger.log(f'{file_name} timeout while checking makespan: '
                                    f'{encoder.makespan} with total running time: {round(encoder.time_used, 5)}')

                    if self.show_solution and result_info['feasible']:
                        self.result_manager.save_solution(file_name, encoder.get_solution())
                    break

                elif sat:
                    # Better solution found
                    encoder.verify()
                    result_info['make_span'] = encoder.makespan
                    result_info['total_solving_time'] = round(encoder.time_used, 5)
                    self.logger.log(f'{file_name} feasible with makespan: '
                                    f'{encoder.makespan} with total running time: {round(encoder.time_used, 5)}')

                else:
                    # No better solution - optimal found
                    self.logger.log(f'{file_name} unfeasible with makespan: '
                                    f'{encoder.makespan} with total running time: {round(encoder.time_used, 5)}')
                    self.logger.log(f'{file_name} optimized with makespan:'
                                    f'{result_info["make_span"]} with total running time: {round(encoder.time_used, 5)}')
                    result_info['optimized'] = True

                    if self.show_solution:
                        self.result_manager.save_solution(file_name, encoder.get_solution())
                    break
            else:
                # Reached lower bound - optimal solution
                result_info['optimized'] = True
                self.logger.log(f'{file_name} optimized with makespan:'
                                f' {result_info["make_span"]} with total running time: {round(encoder.time_used, 5)}')

                if self.show_solution:
                    self.result_manager.save_solution(file_name, encoder.get_solution())

    def run(self):
        """Run the benchmark for all instances in the dataset."""
        start = timeit.default_timer()

        # Read bounds from CSV file
        with open(f'bound/bound_{self.data_set_name}.csv', encoding='utf8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                file_name = f'data_set/{self.data_set_name}/{row[0]}'
                lb = int(row[1])
                ub = int(row[2])

                self.process_instance(file_name, lb, ub)

        total_time = round(timeit.default_timer() - start, 5)
        self.logger.log(f'Benchmark using {self.encoder_type.name} '
                        f'finished with total running time {total_time}')
        return total_time


def benchmark(data_set_name: str, encoder_type: EncoderType, timeout: int, verify: bool,
              verbose: bool = False, show_solution: bool = False):
    """
    Run the benchmark for the given dataset and encoder type.
    Args:
        data_set_name (str): The name of the dataset to benchmark.
        encoder_type (EncoderType): The type of encoder to use.
        timeout: (int) Timeout for solving (0 for no timeout).
        verify: (bool) Verify the result after solving.
        verbose: (bool) Display logs in terminal when True.
        show_solution: (bool) Save the best solution to a file when True.
    """
    runner = BenchmarkRunner(
        data_set_name=data_set_name,
        encoder_type=encoder_type,
        timeout=None if timeout == 0 else timeout,
        verify=verify,
        verbose=verbose,
        show_solution=show_solution
    )
    return runner.run()


def main():
    parser = argparse.ArgumentParser(description='Benchmarking script for SAT encoders.')
    parser.add_argument('dataset_name', type=str, help='The name of the dataset to benchmark.')
    parser.add_argument('encoder_type', type=str,
                        choices=['thesis', 'staircase', 'lia', 'new_staircase'],
                        help='The type of encoder to use: thesis for THESIS_2022, staircase for STAIRCASE.')
    parser.add_argument('timeout', type=int, help='Timeout for solving (0 for no timeout).')
    parser.add_argument('--show_solution', action='store_true',
                        help='Show the solution after solving.')
    parser.add_argument('--verify', action='store_true', help='Verify the result after solving.')
    parser.add_argument('--verbose', action='store_true',
                        help='Display logs in terminal during execution.')

    args = parser.parse_args()

    encoder_type_map = {
        'thesis': EncoderType.THESIS_2022,
        'staircase': EncoderType.STAIRCASE,
        'lia': EncoderType.LIA,
        'new_staircase': EncoderType.NEW_STAIRCASE
    }

    encoder_type = encoder_type_map[args.encoder_type]
    timeout = args.timeout

    print(
        f'Benchmark for {args.dataset_name} using {encoder_type.name} started at {datetime.datetime.now()}')
    benchmark(args.dataset_name, encoder_type, timeout, args.verify, args.verbose,
              args.show_solution)
    print(
        f'Benchmark for {args.dataset_name} using {encoder_type.name} finished at {datetime.datetime.now()}')


if __name__ == '__main__':
    main()
