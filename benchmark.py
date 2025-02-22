import datetime
import multiprocessing
from enum import Enum

from pysat.solvers import Glucose3
import csv
from encoder.problem import Problem
import timeit

TIME_LIMIT = 600


class EncoderType(Enum):
    ORIGINAL = 1
    NEW_NO_OPT = 2
    NEW_OPT = 3


def benchmark(name: str, encoder_type: EncoderType):
    start = timeit.default_timer()

    output_name = (f'result/{name}_{encoder_type.name}_'
                   f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.csv')
    log = f'log/{name}_{encoder_type.name}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'

    match encoder_type:
        case EncoderType.ORIGINAL:
            from encoder.original_encoder import Encoder, PreprocessingFailed
        case EncoderType.NEW_NO_OPT:
            from encoder.new_encoder_no_optimize import Encoder, PreprocessingFailed
        case EncoderType.NEW_OPT:
            from encoder.new_encoder_optimize import Encoder, PreprocessingFailed

    # file_name, num_var, num_clause, feasible, make_span, total_encoding_time, total_solving_time, optimized

    class InfoAttribute(Enum):
        NUM_VAR = 0
        NUM_CLAUSE = 1
        FEASIBLE = 2
        MAKE_SPAN = 3
        TOTAL_ENCODING_TIME = 4
        TOTAL_SOLVING_TIME = 5
        OPTIMIZED = 6

    queue = multiprocessing.Queue()

    def solver(file_name: str, lb: int, ub: int, q=queue):
        p = Problem(file_name)
        result_info = [0, 0, False, 0, 0, 0, False]

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
                q.get()
                q.put(result_info)
            except PreprocessingFailed:
                stop_encoding_time = timeit.default_timer()
                total_encoding_time += (stop_encoding_time - start_encoding_time)
                result_info[
                    InfoAttribute.TOTAL_ENCODING_TIME.value] = total_encoding_time
                return

            solver = Glucose3()
            for c in e.sat_model.clauses:
                solver.add_clause(c)

            start_solving_time = timeit.default_timer()
            sat = solver.solve()
            stop_solving_time = timeit.default_timer()
            total_solving_time += (stop_solving_time - start_solving_time)


            result_info[InfoAttribute.TOTAL_ENCODING_TIME.value] = total_encoding_time
            result_info[InfoAttribute.TOTAL_SOLVING_TIME.value] = total_solving_time

            if sat:
                result_info[InfoAttribute.NUM_VAR.value] = e.sat_model.number_of_variable
                result_info[InfoAttribute.NUM_CLAUSE.value] = len(e.sat_model.clauses)
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
            'num_var,'
            'num_clause,'
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
                f.close()
            p = multiprocessing.Process(target=solver,
                                        args=(file_name, int(row[1]), int(row[2])))
            p.start()
            p.join(TIME_LIMIT)
            result_info = queue.get()

            with open(output_name, "a+") as f:
                f.write(f'{file_name},'
                        f'{result_info[InfoAttribute.NUM_VAR.value]},'
                        f'{result_info[InfoAttribute.NUM_CLAUSE.value]},'
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
                f'Benchmark for {name} finished at {datetime.datetime.now()} with total running time is'
                f' {round(stop - start, 5)}s')


if __name__ == '__main__':
    # multiprocessing.Process(target=benchmark, args=('j60.sm',)).start()
    # multiprocessing.Process(target=benchmark, args=('j90.sm',)).start()
    benchmark('j60.sm', EncoderType.NEW_OPT)
    # benchmark('j30.sm', EncoderType.ORIGINAL)
