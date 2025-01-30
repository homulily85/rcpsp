from pysat.solvers import Glucose3
import csv
from Encoder import Encoder
from Problem import Problem
import timeit


def solver(file_name: str, makespan: int):
    start = timeit.default_timer()
    p = Problem(f'j30.sm/{file_name}')

    e = Encoder(p, makespan, makespan)
    e.encode()
    solver = Glucose3()
    for c in e.sat_model.clauses:
        solver.add_clause(c)

    t = solver.solve()

    stop = timeit.default_timer()

    if t:
        with open("file.txt", "a") as f:
            f.write(f'{file_name} is feasible in {round(stop - start, 2)}s\n')
    else:
        print(f'{file_name} is not feasible')
        exit(1)


if __name__ == '__main__':
    start = timeit.default_timer()

    with open('makespan.csv', encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            solver(row[0], int(row[1]))

    stop = timeit.default_timer()

    print(f'Total running time: {round(stop - start, 2)}s')
