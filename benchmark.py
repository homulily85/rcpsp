import argparse
import csv
import datetime
import timeit
from enum import Enum

from encoder.SAT_model import NUMBER_OF_LITERAL
from encoder.problem import Problem


class EncoderType(Enum):
    THESIS_2022 = 1
    STAIRCASE = 2


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


def benchmark(data_set_name: str, encoder_type: EncoderType, timeout: int, verify: bool):
    """
    Run the benchmark for the given dataset and encoder type.
    Args:
        data_set_name (str): The name of the dataset to benchmark.
        encoder_type (EncoderType): The type of encoder to use.
        timeout: (int) Timeout for solving (0 for no timeout).
        verify: (bool) Verify the result after solving.
    """
    start = timeit.default_timer()

    output_name = (f'result/{data_set_name}_{encoder_type.name}_'
                   f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.csv')
    log = f'log/{data_set_name}_{encoder_type.name}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'

    match encoder_type:
        case EncoderType.THESIS_2022:
            from encoder.thesis_2022 import Thesis2022Encoder
        case EncoderType.STAIRCASE:
            from encoder.staircase import StaircaseEncoder

    with open(output_name, "a+") as f:
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

    with open(f'bound/bound_{data_set_name}.csv', encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            file_name = f'data_set/{data_set_name}/{row[0]}'
            lb = int(row[1])
            ub = int(row[2])
            with open(log, "a+") as f:
                f.write(f'{datetime.datetime.now()} benchmark for {file_name}'
                        f' using {encoder_type.name} started\n')

            p = Problem(file_name)
            encoder = None
            if encoder_type == EncoderType.STAIRCASE:
                encoder = StaircaseEncoder(p, ub, timeout, verify)
            elif encoder_type == EncoderType.THESIS_2022:
                encoder = Thesis2022Encoder(p, ub, timeout, verify)

            encoder.encode()

            result_info = {
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

            sat = encoder.solve()

            if sat is None:
                result_info['timeout'] = True
                with open(log, "a+") as f:
                    f.write(
                        f'{datetime.datetime.now()} {file_name} timeout while checking makespan: '
                        f'{encoder.makespan} and time: {round(encoder.time_used, 5)}\n')

            elif sat:
                encoder.verify()
                result_info['feasible'] = True
                result_info['make_span'] = encoder.makespan
                result_info['total_solving_time'] = round(encoder.time_used, 5)
                result_info['timeout'] = False

                with open(log, "a+") as f:
                    f.write(f'{datetime.datetime.now()} {file_name} feasible with makespan: '
                            f'{encoder.makespan} and time: {round(encoder.time_used, 5)}\n')

                while sat and encoder.makespan > lb:
                    encoder.decrease_makespan()
                    sat = encoder.solve()

                    if sat is None:
                        result_info['timeout'] = True
                        with open(log, "a+") as f:
                            f.write(
                                f'{datetime.datetime.now()} {file_name} timeout while checking makespan: '
                                f'{encoder.makespan} and time: {round(encoder.time_used, 5)}\n')
                        break
                    elif sat:
                        encoder.verify()
                        result_info['make_span'] = encoder.makespan
                        result_info['total_solving_time'] = round(encoder.time_used, 5)
                        with open(log, "a+") as f:
                            f.write(
                                f'{datetime.datetime.now()} {file_name} feasible with makespan: '
                                f'{encoder.makespan} and time: {round(encoder.time_used, 5)}\n')
                    else:
                        with open(log, "a+") as f:
                            f.write(
                                f'{datetime.datetime.now()} {file_name} unfeasible with makespan: '
                                f'{encoder.makespan} and time: {round(encoder.time_used, 5)}\n')
                            f.write(
                                f'{datetime.datetime.now()} {file_name} optimized with makespan:'
                                f'{result_info['make_span']} and time: {round(encoder.time_used, 5)}\n')
                        result_info['optimized'] = True
                        break
                else:
                    result_info['optimized'] = True
                    with open(log, "a+") as f:
                        f.write(
                            f'{datetime.datetime.now()} {file_name} optimized with makespan:'
                            f' {result_info['make_span']} and time: {round(encoder.time_used, 5)}\n')

            with open(output_name, "a+") as f:
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

        with open(log, "a+") as f:
            f.write(
                f'{datetime.datetime.now()} {file_name} using {encoder_type.name} '
                f'finished with total running time {round(timeit.default_timer() - start, 5)}\n')


def main():
    parser = argparse.ArgumentParser(description='Benchmarking script for SAT encoders.')
    parser.add_argument('dataset_name', type=str, help='The name of the dataset to benchmark.')
    parser.add_argument('encoder_type', type=str, choices=['thesis', 'staircase'],
                        help='The type of encoder to use: thesis for THESIS_2022, staircase for STAIRCASE.')
    parser.add_argument('timeout', type=int, help='Timeout for solving (0 for no timeout).')
    parser.add_argument('--verify', action='store_true', help='Verify the result after solving.')

    args = parser.parse_args()

    encoder_type_map = {
        'thesis': EncoderType.THESIS_2022,
        'staircase': EncoderType.STAIRCASE,
    }

    encoder_type = encoder_type_map[args.encoder_type]
    timeout = None if args.timeout == 0 else args.timeout

    print(
        f'Benchmark for {args.dataset_name} using {encoder_type.name} started at {datetime.datetime.now()}')
    benchmark(args.dataset_name, encoder_type, timeout, args.verify)
    print(
        f'Benchmark for {args.dataset_name} using {encoder_type.name} finished at {datetime.datetime.now()}')


if __name__ == '__main__':
    main()
