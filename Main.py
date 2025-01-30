import datetime
import multiprocessing
import os.path

from pysat.solvers import Glucose3
import csv
from Encoder import Encoder
from Problem import Problem
import timeit

TIME_LIMIT = 600


def solver(file_name: str, output_name: str, lb: int, ub: int):
    start = timeit.default_timer()
    p = Problem(file_name)

    tmp = {'makespan': ub + 1, 'nvar': -1, 'nclause': -1}

    for i in range(ub, lb - 1, -1):
        try:
            e = Encoder(p, i, ub)
            e.encode()
        except KeyError:
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

        t = solver.solve()

        if t:
            if i == lb:
                stop = timeit.default_timer()
                with open(output_name, "a+") as f:
                    f.write(
                        f'{file_name},'
                        f'{e.sat_model.nvariable},'
                        f'{len(e.sat_model.clauses)},'
                        f'1,'
                        f'{e.makespan},'
                        f'{round(stop - start, 5)}\n')
                    f.close()
            else:
                tmp['makespan'] = i
                tmp['nvar'] = e.sat_model.nvariable
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


def benchmark(name: str):
    start = timeit.default_timer()

    output_name = f'result_{name}.csv'
    try:
        os.remove(output_name)
    except OSError:
        pass

    with open(output_name, "a+") as f:
        f.write('file_name,num_var,num_clause,feasible,make_span,solve_time\n')
        f.close()

    with open(f'makespan_{name}.csv', encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            file_name = f'{name}/{row[0]}'
            print(f'{datetime.datetime.now()}:Running {file_name}')
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

        print(f'Total running time for {name}: {round(stop - start, 5)}s')


if __name__ == '__main__':
    benchmark('j30.sm')
