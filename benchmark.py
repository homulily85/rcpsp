import datetime
import multiprocessing
from enum import Enum

from pysat.solvers import Glucose42
import csv
from encoder.problem import Problem
import timeit

TIME_LIMIT = 300  # seconds


class EncoderType(Enum):
    PAPER_2022 = 1
    PSEUDO_STAIRCASE = 2
    STAIRCASE = 3


def benchmark(name: str, encoder_type: EncoderType):
    start = timeit.default_timer()

    output_name = (f'result/{name}_{encoder_type.name}_'
                   f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.csv')
    log = f'log/{name}_{encoder_type.name}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'

    match encoder_type:
        case EncoderType.PAPER_2022:
            from encoder.paper_2022 import Encoder, PreprocessingFailed
        case EncoderType.PSEUDO_STAIRCASE:
            from encoder.pseudo_staircase import Encoder, PreprocessingFailed
        case EncoderType.STAIRCASE:
            from encoder.staircase import Encoder, PreprocessingFailed

    class InfoAttribute(Enum):
        LB = 0
        UB = 1
        NUM_VAR = 2
        NUM_CLAUSE = 3
        NUM_CONSISTENCY_ClAUSE = 4
        NUM_PB_CLAUSE = 5
        FEASIBLE = 6
        MAKE_SPAN = 7
        TOTAL_ENCODING_TIME = 8
        TOTAL_SOLVING_TIME = 9
        OPTIMIZED = 10
        ZERO_LITS = 11
        ONE_LITS = 12
        TWO_LITS = 13
        THREE_LITS = 14
        FOUR_LITS = 15
        MORE_THAN_FOUR_LITS = 16
        MORE_THAN_TEN_LITS = 17

    queue = multiprocessing.Queue()

    def solver(file_name: str, lb: int, ub: int, q=queue):
        p = Problem(file_name)
        result_info = [0, 0, 0, 0, 0, 0, False, 0, 0, 0, False, 0, 0, 0, 0, 0, 0, 0]

        q.put(result_info)

        total_encoding_time = 0
        total_solving_time = 0

        start_encoding_time = 0
        for i in range(ub, lb - 1, -1):
            try:
                start_encoding_time = timeit.default_timer()
                e = Encoder(p, i)
                e.encode()
                stop_encoding_time = timeit.default_timer()
                total_encoding_time += (stop_encoding_time - start_encoding_time)
                result_info[InfoAttribute.TOTAL_ENCODING_TIME.value] = total_encoding_time * 1000
                result_info[InfoAttribute.NUM_VAR.value] = e.sat_model.number_of_variable
                result_info[InfoAttribute.NUM_CLAUSE.value] = len(e.sat_model.clauses)
                lits = e.sat_model.number_of_literal()
                result_info[InfoAttribute.ZERO_LITS.value] = lits[0]
                result_info[InfoAttribute.ONE_LITS.value] = lits[1]
                result_info[InfoAttribute.TWO_LITS.value] = lits[2]
                result_info[InfoAttribute.THREE_LITS.value] = lits[3]
                result_info[InfoAttribute.FOUR_LITS.value] = lits[4]
                result_info[InfoAttribute.MORE_THAN_FOUR_LITS.value] = lits[5]
                result_info[InfoAttribute.MORE_THAN_TEN_LITS.value] = lits[11]
                result_info[InfoAttribute.NUM_PB_CLAUSE.value] = e.sat_model.number_of_PB_clause
                result_info[
                    InfoAttribute.NUM_CONSISTENCY_ClAUSE.value] = e.sat_model.number_of_consistency_clause
                result_info[InfoAttribute.LB.value] = lb
                result_info[InfoAttribute.UB.value] = ub
                q.get()
                q.put(result_info)
            except PreprocessingFailed:
                stop_encoding_time = timeit.default_timer()
                total_encoding_time += (stop_encoding_time - start_encoding_time)
                result_info[
                    InfoAttribute.TOTAL_ENCODING_TIME.value] = total_encoding_time * 1000
                q.get()
                q.put(result_info)
                return

            solver = Glucose42()
            for c in e.sat_model.clauses:
                solver.add_clause(c)

            start_solving_time = timeit.default_timer()
            sat = solver.solve()
            stop_solving_time = timeit.default_timer()
            total_solving_time += (stop_solving_time - start_solving_time)

            result_info[InfoAttribute.TOTAL_ENCODING_TIME.value] = total_encoding_time * 1000
            result_info[InfoAttribute.TOTAL_SOLVING_TIME.value] = total_solving_time * 1000

            if sat:
                result_info[InfoAttribute.MAKE_SPAN.value] = i
                result_info[InfoAttribute.FEASIBLE.value] = True

                if i == lb:
                    result_info[InfoAttribute.OPTIMIZED.value] = True
                    q.get()
                    q.put(result_info)
                else:
                    q.get()
                    q.put(result_info)
            else:
                result_info[InfoAttribute.OPTIMIZED.value] = True
                q.get()
                q.put(result_info)
                return

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
            'total_encoding_time,'
            'total_solving_time,'
            'optimized,'
            'timeout\n')
        f.close()

    with open(f'bound/bound_{name}.csv', encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            file_name = f'data_set/{name}/{row[0]}'
            with open(log, "a+") as f:
                f.write(f'{datetime.datetime.now()}\t {file_name} started\n')
                # print(
                # f'{datetime.datetime.now()}\t {file_name} using {encoder_type.name} started\n')
                f.close()
            p = multiprocessing.Process(target=solver,
                                        args=(file_name, int(row[1]), int(row[2])))
            p.start()
            p.join(TIME_LIMIT)
            result_info = queue.get()

            with open(output_name, "a+") as f:
                f.write(f'{file_name},'
                        f'{result_info[InfoAttribute.LB.value]},'
                        f'{result_info[InfoAttribute.UB.value]},'
                        f'{result_info[InfoAttribute.NUM_VAR.value]},'
                        f'{result_info[InfoAttribute.NUM_CLAUSE.value]},'
                        f'{result_info[InfoAttribute.NUM_CONSISTENCY_ClAUSE.value]},'
                        f'{result_info[InfoAttribute.NUM_PB_CLAUSE.value]},'
                        f'{result_info[InfoAttribute.ZERO_LITS.value]},'
                        f'{result_info[InfoAttribute.ONE_LITS.value]},'
                        f'{result_info[InfoAttribute.TWO_LITS.value]},'
                        f'{result_info[InfoAttribute.THREE_LITS.value]},'
                        f'{result_info[InfoAttribute.FOUR_LITS.value]},'
                        f'{result_info[InfoAttribute.MORE_THAN_FOUR_LITS.value]},'
                        f'{result_info[InfoAttribute.MORE_THAN_TEN_LITS.value]},'
                        f'{result_info[InfoAttribute.FEASIBLE.value]},'
                        f'{result_info[InfoAttribute.MAKE_SPAN.value]},'
                        f'{round(result_info[InfoAttribute.TOTAL_ENCODING_TIME.value], 5)},'
                        f'{round(result_info[InfoAttribute.TOTAL_SOLVING_TIME.value], 5)},'
                        f'{result_info[InfoAttribute.OPTIMIZED.value]},')
                f.close()
            if p.is_alive():
                p.kill()
                with open(output_name, "a+") as f:
                    f.write(f'True\n')
                    f.close()
            else:
                with open(output_name, "a+") as f:
                    f.write(f'False\n')
                    f.close()

        stop = timeit.default_timer()
        csv_file.close()

        with open(log, "a+") as f:
            f.write(
                f'Benchmark for {name} using {encoder_type.name} finished at '
                f'{datetime.datetime.now()} with total running time is'
                f' {round(stop - start, 5)}s')


if __name__ == '__main__':
    for data_set in ['j30.sm']:
        for type in [EncoderType.STAIRCASE]:
            print(
                f'Benchmark for {data_set} using {type.name} started at {datetime.datetime.now()}')
            benchmark(data_set, type)
            print(
                f'Benchmark for {data_set} using {type.name} finished at {datetime.datetime.now()}')
