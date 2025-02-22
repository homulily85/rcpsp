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

    def solver(file_name: str, output_name: str, lb: int, ub: int):
        start = timeit.default_timer()
        p = Problem(file_name)

        # Used to stored previous result
        tmp = {'makespan': ub + 1, 'nvar': -1, 'nclause': -1}

        for i in range(ub, lb - 1, -1):
            try:
                e = Encoder(p, i)
                e.encode()

            except (PreprocessingFailed):
                stop = timeit.default_timer()
                with open(output_name, "a+") as f:
                    f.write(
                        f'{file_name},'
                        f'{'' if tmp['nvar'] == -1 else tmp['nvar']},'
                        f'{'' if tmp['nclause'] == -1 else tmp['nclause']},'
                        f'{0 if tmp['nvar'] == -1 else 1},'
                        f'{'' if tmp['nvar'] == -1 else tmp['makespan']},'
                        f'{round(stop - start, 5)}\n')
                    f.close()
                    return

            solver = Glucose3()
            for c in e.sat_model.clauses:
                solver.add_clause(c)

            model = solver.solve()

            if model:
                if i == lb:
                    stop = timeit.default_timer()
                    with open(output_name, "a+") as f:
                        f.write(
                            f'{file_name},'
                            f'{e.sat_model.number_of_variable},'
                            f'{len(e.sat_model.clauses)},'
                            f'1,'
                            f'{e.makespan},'
                            f'{round(stop - start, 5)}\n')
                        f.close()
                else:
                    tmp['makespan'] = i
                    tmp['nvar'] = e.sat_model.number_of_variable
                    tmp['nclause'] = len(e.sat_model.clauses)
            else:
                stop = timeit.default_timer()
                with open(output_name, "a+") as f:
                    f.write(
                        f'{file_name},'
                        f'{'' if tmp['nvar'] == -1 else tmp['nvar']},'
                        f'{'' if tmp['nclause'] == -1 else tmp['nclause']},'
                        f'{0 if tmp['nvar'] == -1 else 1},'
                        f'{'' if tmp['nvar'] == -1 else tmp['makespan']},'
                        f'{round(stop - start, 5)}\n')
                    f.close()
                    return

    with open(output_name, "a+") as f:
        f.write('file_name,num_var,num_clause,feasible,make_span,solve_time\n')
        f.close()

    with open(f'bound/bound_{name}.csv', encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            file_name = f'data_set/{name}/{row[0]}'
            with open(log, "a+") as f:
                f.write(f'{datetime.datetime.now()}\t {file_name} started\n')
                f.close()
            p = multiprocessing.Process(target=solver,
                                        args=(file_name, output_name, int(row[1]), int(row[2])))
            p.start()
            p.join(TIME_LIMIT)
            if p.is_alive():
                p.kill()
                with open(output_name, "a+") as f:
                    f.write(f'{file_name},,,0,,time out\n')
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
    benchmark('j30.sm', EncoderType.NEW_OPT)
