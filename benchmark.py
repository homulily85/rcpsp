import argparse
import csv
import datetime
import multiprocessing
import os
import subprocess
import timeit
from enum import Enum, auto
from typing import Dict, Any, Optional

from encoder.model.sat_model import NUMBER_OF_LITERALS
from encoder.model.status import SOLVER_STATUS


class EncoderType(Enum):
    THESIS = auto()
    STAIRCASE = auto()
    LIA = auto()
    MAXSAT = auto()


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
            print(f'[{timestamp}] {message}', flush=True)


class ResultManager:
    """Manages result output and solution files."""

    def __init__(self, output_path: str,
                 encoder_type: EncoderType,
                 show_solution: bool = False):
        self.output_path = output_path
        self.encoder_type = encoder_type
        self.show_solution = show_solution
        self.solution_file = None

        # Create result directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Initialize output file with headers
        self.__initialize_output_file()

        # Create _solution file if needed
        if show_solution:
            solution_dir = 'solution'
            os.makedirs(solution_dir, exist_ok=True)
            self.solution_file = f'{solution_dir}/{os.path.basename(output_path).replace(".csv", ".sol")}'

    def __initialize_output_file(self):
        """Create the output file with appropriate headers based on encoder type."""
        with open(self.output_path, "a+") as f:
            match self.encoder_type:
                case EncoderType.STAIRCASE | EncoderType.THESIS:
                    f.write(
                        'file_name,'
                        'lower_bound,'
                        'upper_bound,'
                        'variables,'
                        'clauses,'
                        'consistency_clauses,'
                        'pb_clauses,'
                        'zero_literal,'
                        'one_literals,'
                        'two_literals,'
                        'three_literals,'
                        'four_literals,'
                        'five_to_ten_literals,'
                        'more_than_ten_literals,'
                        'feasible,'
                        'makespan,'
                        'total_solving_time,'
                        'optimized,'
                        'timeout\n')
                case EncoderType.MAXSAT:
                    f.write(
                        'file_name,'
                        'lower_bound,'
                        'upper_bound,'
                        'variables,'
                        'clauses,'
                        'soft_clauses,'
                        'hard_constraint_clauses,'
                        'consistency_clauses,'
                        'pb_clauses,'
                        'zero_literals,'
                        'one_literals,'
                        'two_literals,'
                        'three_literals,'
                        'four_literals,'
                        'five_to_ten_literals,'
                        'more_than_ten_literals,'
                        'feasible,'
                        'makespan,'
                        'total_solving_time,'
                        'optimized,'
                        'timeout\n')
                case EncoderType.LIA:
                    f.write(
                        f'file_name,'
                        f'feasible,'
                        f'makespan,'
                        f'total_solving_time,'
                        f'optimized,'
                        f'timeout\n')

    def save_result(self, result_info: Dict[str, Any]):
        """Save benchmark results to the output file."""
        with open(self.output_path, "a+") as f:
            match self.encoder_type:
                case EncoderType.STAIRCASE | EncoderType.THESIS:
                    f.write(
                        f'{result_info["file_name"]},'
                        f'{result_info["lower_bound"]},'
                        f'{result_info["upper_bound"]},'
                        f'{result_info["variables"]},'
                        f'{result_info["clauses"]},'
                        f'{result_info["consistency_clauses"]},'
                        f'{result_info["pb_clauses"]},'
                        f'{result_info["zero_literals"]},'
                        f'{result_info["one_literals"]},'
                        f'{result_info["two_literals"]},'
                        f'{result_info["three_literals"]},'
                        f'{result_info["four_literals"]},'
                        f'{result_info["five_to_ten_literals"]},'
                        f'{result_info["more_than_ten_literals"]},'
                        f'{int(result_info["feasible"])},'
                        f'{result_info["makespan"]},'
                        f'{result_info["total_solving_time"]},'
                        f'{int(result_info["optimized"])},'
                        f'{int(result_info["timeout"])}\n')
                case EncoderType.MAXSAT:
                    f.write(
                        f'{result_info["file_name"]},'
                        f'{result_info["lower_bound"]},'
                        f'{result_info["upper_bound"]},'
                        f'{result_info["variables"]},'
                        f'{result_info["clauses"]},'
                        f'{result_info["soft_clauses"]},'
                        f'{result_info["hard_clauses"]},'
                        f'{result_info["consistency_clauses"]},'
                        f'{result_info["pb_clauses"]},'
                        f'{result_info["zero_literals"]},'
                        f'{result_info["one_literals"]},'
                        f'{result_info["two_literals"]},'
                        f'{result_info["three_literals"]},'
                        f'{result_info["four_literals"]},'
                        f'{result_info["five_to_ten_literals"]},'
                        f'{result_info["more_than_ten_literals"]},'
                        f'{int(result_info["feasible"])},'
                        f'{result_info["makespan"]},'
                        f'{result_info["total_solving_time"]},'
                        f'{int(result_info["optimized"])},'
                        f'{int(result_info["timeout"])}\n')
                case EncoderType.LIA:
                    f.write(
                        f'{result_info["file_name"]},'
                        f'{int(result_info["feasible"])},'
                        f'{result_info["makespan"]},'
                        f'{result_info["total_solving_time"]},'
                        f'{int(result_info["optimized"])},'
                        f'{int(result_info["timeout"])}\n'
                    )

    def save_solution(self, file_name: str, solution: list[int]):
        """Save solution to _solution file if enabled."""
        if self.show_solution and self.solution_file:
            with open(self.solution_file, "a+") as f:
                f.write(f'{file_name}\t{solution}\n')


class BenchmarkRunner:
    """Runs the benchmark process for a dataset using specified encoder."""

    def __init__(self, data_set_name: str,
                 encoder_type: EncoderType,
                 timeout: Optional[int],
                 verify: bool = False,
                 verbose: bool = False,
                 show_solution: bool = False):
        self.data_set_name = data_set_name
        self.encoder_type = encoder_type
        self.timeout = timeout
        self.verify = verify
        self.verbose = verbose
        self.show_solution = show_solution
        self.solution = None

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_name = f'result/{data_set_name}_{encoder_type.name}_{timestamp}.csv'
        log_file = f'log/{data_set_name}_{encoder_type.name}_{timestamp}.log'

        self.logger = BenchmarkLogger(log_file, verbose)
        self.result_manager = ResultManager(output_name, encoder_type, show_solution)

    def create_encoder(self, input_file: str,
                       lower_bound: int,
                       upper_bound: int):
        """Factory method to create the appropriate encoder."""
        match self.encoder_type:
            case EncoderType.STAIRCASE:
                from encoder.sat.incremental_sat.staircase_new import ImprovedStaircaseMethod
                return ImprovedStaircaseMethod(input_file, lower_bound, upper_bound, self.timeout,
                                               self.verify)
            case EncoderType.THESIS:
                from encoder.sat.incremental_sat.thesis_2022 import ThesisMethod
                return ThesisMethod(input_file, lower_bound, upper_bound, self.timeout,
                                    self.verify)
            case EncoderType.LIA:
                from encoder.lia.lia_solver import LIASolver
                return LIASolver(input_file, self.timeout, self.verify)

            case EncoderType.MAXSAT:
                from encoder.sat.max_sat.maxsat_solver import MaxSATSolver
                return MaxSATSolver(input_file, lower_bound, upper_bound, self.timeout, self.verify)
            case _:
                raise ValueError(f"Unsupported encoder type: {self.encoder_type.name}")

    def create_result_info(self, file_name: str,
                           lower_bound: int,
                           upper_bound: int,
                           encoder) \
            -> Dict[str, str | int] | None:
        """Create initial result info dictionary based on encoder type."""
        match self.encoder_type:
            case EncoderType.STAIRCASE | EncoderType.THESIS:
                return {
                    'file_name': file_name,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'variables': encoder.sat_model.number_of_variables,
                    'clauses': encoder.sat_model.number_of_clauses,
                    'consistency_clauses': encoder.sat_model.number_of_consistency_clauses,
                    'pb_clauses': encoder.sat_model.number_of_pb_clauses,
                    'zero_literals': encoder.sat_model.number_of_literals[NUMBER_OF_LITERALS.ZERO],
                    'one_literals': encoder.sat_model.number_of_literals[NUMBER_OF_LITERALS.ONE],
                    'two_literals': encoder.sat_model.number_of_literals[NUMBER_OF_LITERALS.TWO],
                    'three_literals': encoder.sat_model.number_of_literals[
                        NUMBER_OF_LITERALS.THREE],
                    'four_literals': encoder.sat_model.number_of_literals[NUMBER_OF_LITERALS.FOUR],
                    'five_to_ten_literals': encoder.sat_model.number_of_literals[
                        NUMBER_OF_LITERALS.FIVE_TO_TEN],
                    'more_than_ten_literals': encoder.sat_model.number_of_literals[
                        NUMBER_OF_LITERALS.MORE_THAN_TEN],
                    'feasible': False,
                    'makespan': 0,
                    'total_solving_time': 0,
                    'optimized': False,
                    'timeout': False
                }
            case EncoderType.MAXSAT:  # MaxSATSolver
                return {
                    'file_name': file_name,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'variables': encoder.sat_model.number_of_variables,
                    'clauses': encoder.sat_model.number_of_clauses,
                    'soft_clauses': encoder.sat_model.number_of_soft_clauses,
                    'hard_clauses': encoder.sat_model.number_of_hard_clauses,
                    'consistency_clauses': encoder.sat_model.number_of_consistency_clauses,
                    'pb_clauses': encoder.sat_model.number_of_pb_clauses,
                    'zero_literals': encoder.sat_model.number_of_literals[NUMBER_OF_LITERALS.ZERO],
                    'one_literals': encoder.sat_model.number_of_literals[NUMBER_OF_LITERALS.ONE],
                    'two_literals': encoder.sat_model.number_of_literals[NUMBER_OF_LITERALS.TWO],
                    'three_literals': encoder.sat_model.number_of_literals[
                        NUMBER_OF_LITERALS.THREE],
                    'four_literals': encoder.sat_model.number_of_literals[NUMBER_OF_LITERALS.FOUR],
                    'five_to_ten_literals': encoder.sat_model.number_of_literals[
                        NUMBER_OF_LITERALS.FIVE_TO_TEN],
                    'more_than_ten_literals': encoder.sat_model.number_of_literals[
                        NUMBER_OF_LITERALS.MORE_THAN_TEN],
                    'feasible': False,
                    'makespan': 0,
                    'total_solving_time': 0,
                    'optimized': False,
                    'timeout': False
                }
            case EncoderType.LIA:
                return {
                    'file_name': file_name,
                    'feasible': False,
                    'makespan': 0,
                    'total_solving_time': 0,
                    'optimized': False,
                    'timeout': False
                }
            case _:
                raise ValueError(f"Unsupported encoder type: {self.encoder_type.name}")

    def process_instance(self, file_name: str, lower_bound: int, upper_bound: int):
        """Process a single problem instance."""
        self.logger.log(f'benchmark for {file_name} using {self.encoder_type.name} started')

        # Create problem and encoder
        encoder = self.create_encoder(file_name, lower_bound, upper_bound)

        # Encode the problem
        if self.encoder_type != EncoderType.LIA:
            encoder.encode()

        # Create initial result info
        result_info = self.create_result_info(file_name, lower_bound, upper_bound, encoder)

        # Solve the problem
        self._solve_and_optimize(encoder, result_info, file_name)

        # Save results
        self.result_manager.save_result(result_info)

    def _solve_and_optimize(self, encoder,
                            result_info: Dict[str, Any],
                            file_name: str):
        """Solve the problem and optimize the makespan."""
        # Initial solve
        if self.encoder_type == EncoderType.MAXSAT:
            sat = encoder.solve()
            match sat:
                case SOLVER_STATUS.UNKNOWN:
                    result_info['timeout'] = True
                    result_info['total_solving_time'] = round(encoder.time_used, 5)
                    self.logger.log(f'{file_name} timeout, '
                                    f'total running time: {round(encoder.time_used, 5)}s')
                    return
                case SOLVER_STATUS.UNSATISFIABLE:
                    result_info['feasible'] = False
                    result_info['total_solving_time'] = round(encoder.time_used, 5)
                    self.logger.log(f'{file_name} unfeasible within given bounds, '
                                    f'total running time: {round(encoder.time_used, 5)}s')
                    return
                case SOLVER_STATUS.SATISFIABLE:
                    result_info['feasible'] = True
                    result_info['optimized'] = False
                    result_info['timeout'] = True
                    result_info['makespan'] = encoder.makespan
                    result_info['total_solving_time'] = round(encoder.time_used, 5)
                    self.result_manager.save_solution(file_name, encoder.solution)
                    self.logger.log(f'{file_name} feasible with makespan: '
                                    f'{encoder.makespan}, total running time: {round(encoder.time_used, 5)}s '
                                    f'but cannot find optimal solution.')
                    return
                case SOLVER_STATUS.OPTIMUM:
                    result_info['feasible'] = True
                    result_info['optimized'] = True
                    result_info['timeout'] = False
                    result_info['makespan'] = encoder.makespan
                    result_info['total_solving_time'] = round(encoder.time_used, 5)
                    self.result_manager.save_solution(file_name, encoder.solution)
                    self.logger.log(f'{file_name} feasible with makespan: '
                                    f'{encoder.makespan}, total running time: {round(encoder.time_used, 5)}s. '
                                    f'This is the optimal solution.')
                    return

        elif self.encoder_type == EncoderType.LIA:
            sat = encoder.solve()
            match sat:
                case SOLVER_STATUS.SATISFIABLE:
                    result_info['feasible'] = True
                    result_info['optimized'] = False
                    result_info['timeout'] = True
                    result_info['makespan'] = encoder.makespan
                    result_info['total_solving_time'] = round(encoder.time_used, 5)
                    self.result_manager.save_solution(file_name, encoder.solution)
                    self.logger.log(f'{file_name} feasible with makespan: '
                                    f'{encoder.makespan}, total running time: {round(encoder.time_used, 5)}s '
                                    f'but cannot find optimal solution.')
                    return
                case SOLVER_STATUS.OPTIMUM:
                    result_info['feasible'] = True
                    result_info['optimized'] = True
                    result_info['timeout'] = False
                    result_info['makespan'] = encoder.makespan
                    result_info['total_solving_time'] = round(encoder.time_used, 5)
                    self.result_manager.save_solution(file_name, encoder.solution)
                    self.logger.log(f'{file_name} feasible with makespan: '
                                    f'{encoder.makespan}, total running time: {round(encoder.time_used, 5)}s. '
                                    f'This is the optimal solution.')
                    return
        else:
            sat = encoder.solve()

            if sat is None:
                result_info['timeout'] = True
                self.logger.log(f'{file_name} timeout while checking makespan: '
                                f'{encoder.makespan}, total running time: {round(encoder.time_used, 5)}s')
                result_info['total_solving_time'] = round(encoder.time_used, 5)
                if self.show_solution and result_info['feasible']:
                    self.result_manager.save_solution(file_name, self.solution)
                return

            if sat:
                # Solution found - verify and update result info
                encoder.verify()
                result_info['feasible'] = True
                result_info['makespan'] = encoder.makespan
                result_info['total_solving_time'] = round(encoder.time_used, 5)
                if self.show_solution:
                    self.solution = encoder.solution

                self.logger.log(f'{file_name} feasible with makespan: '
                                f'{encoder.makespan}, total running time: {round(encoder.time_used, 5)}s')

                # Try to optimize by decreasing makespan
                while sat and encoder.makespan > result_info['lower_bound']:
                    encoder.decrease_makespan()
                    sat = encoder.solve()

                    if sat is None:
                        # Timeout during optimization
                        result_info['timeout'] = True
                        self.logger.log(f'{file_name} timeout while checking makespan: '
                                        f'{encoder.makespan}, total running time: {round(encoder.time_used, 5)}s')

                        if self.show_solution and result_info['feasible']:
                            self.result_manager.save_solution(file_name, self.solution)
                        break

                    elif sat:
                        # Better solution found
                        encoder.verify()
                        result_info['makespan'] = encoder.makespan
                        result_info['total_solving_time'] = round(encoder.time_used, 5)
                        if self.show_solution:
                            self.solution = encoder.solution
                        self.logger.log(f'{file_name} feasible with makespan: '
                                        f'{encoder.makespan}, total running time: {round(encoder.time_used, 5)}s')

                    else:
                        # No better solution - optimal found
                        self.logger.log(f'{file_name} unfeasible with makespan: '
                                        f'{encoder.makespan} with total running time: {round(encoder.time_used, 5)}s')
                        self.logger.log(f'{file_name} optimized with makespan:'
                                        f'{result_info["makespan"]}, total running time: {round(encoder.time_used, 5)}s')
                        result_info['optimized'] = True

                        if self.show_solution:
                            self.result_manager.save_solution(file_name, self.solution)
                        break
                else:
                    # Reached lower bound - optimal solution
                    result_info['optimized'] = True
                    self.logger.log(f'{file_name} optimized with makespan:'
                                    f' {result_info["makespan"]}, total running time: {round(encoder.time_used, 5)}s')

                    if self.show_solution:
                        self.result_manager.save_solution(file_name, self.solution)
            else:
                # No solution found
                result_info['feasible'] = False
                result_info['makespan'] = 0
                result_info['total_solving_time'] = round(encoder.time_used, 5)
                self.logger.log(f'{file_name} unfeasible with makespan: '
                                f'{encoder.makespan}, total running time: {round(encoder.time_used, 5)}s')

    def run(self):
        """Run the benchmark for all instances in the dataset."""
        start = timeit.default_timer()

        # Read bounds from CSV file
        with open(f'bound/bound_{self.data_set_name}.csv', encoding='utf8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader)  # Skip the header line
            for row in csv_reader:
                file_name = f'data_set/{self.data_set_name}/{row[0]}'
                lower_bound = int(row[1])
                upper_bound = int(row[2])

                p = multiprocessing.Process(target=self.process_instance,
                                            args=(file_name, lower_bound, upper_bound))
                p.start()
                p.join()
                p.terminate()

        total_time = round(timeit.default_timer() - start, 5)
        self.logger.log(f'Benchmark using {self.encoder_type.name} '
                        f'finished, total running time {total_time}s')


def benchmark(data_set_name: str,
              encoder_type: EncoderType,
              timeout: int,
              verify: bool = False,
              verbose: bool = False,
              save_solution: bool = False):
    """
    Run the benchmark for the given dataset and encoder type.
    Args:
        data_set_name (str): The name of the dataset to benchmark.
        encoder_type (EncoderType): The type of encoder to use.
        timeout: (int) Timeout for solving (0 for no timeout).
        verify: (bool) Verify the result after solving.
        verbose: (bool) Display logs in terminal when True.
        save_solution: (bool) Save the best solution to a file when True.
    """
    runner = BenchmarkRunner(
        data_set_name=data_set_name,
        encoder_type=encoder_type,
        timeout=None if timeout == 0 else timeout,
        verify=verify,
        verbose=verbose,
        show_solution=save_solution
    )
    runner.run()


def main():
    parser = argparse.ArgumentParser(description='Benchmarking script for SAT encoders.')
    parser.add_argument('dataset_name', type=str, help='The name of the dataset to benchmark.')
    parser.add_argument('encoder_type', type=str,
                        choices=['thesis', 'staircase', 'lia', 'maxsat'],
                        help='The type of encoder to use.')
    parser.add_argument('timeout', type=int, help='Timeout for solving (0 for no timeout).')
    parser.add_argument('--save_solution', action='store_true',
                        help='Save the solution after solving.')
    parser.add_argument('--verify', action='store_true', help='Verify the solution after solving.')
    parser.add_argument('--verbose', action='store_true',
                        help='Display logs in terminal during execution.')

    args = parser.parse_args()

    encoder_type_map = {
        'thesis': EncoderType.THESIS,
        'staircase': EncoderType.STAIRCASE,
        'lia': EncoderType.LIA,
        'maxsat': EncoderType.MAXSAT,
    }

    encoder_type = encoder_type_map[args.encoder_type]
    timeout = args.timeout

    print(
        f'Benchmark for {args.dataset_name} using {encoder_type.name} started at {datetime.datetime.now()}')
    benchmark(args.dataset_name, encoder_type, timeout, args.verify, args.verbose,
              args.save_solution)
    print(
        f'Benchmark for {args.dataset_name} using {encoder_type.name} finished at {datetime.datetime.now()}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        process = subprocess.Popen(
            "killall python tt-open-wbo-inc-Glucose4_1_static mrcpsp2smt",
            shell=True)
        process.wait()
