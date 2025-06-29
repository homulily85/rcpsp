import argparse
import datetime
import logging
import multiprocessing
import os
import shutil
import time
import timeit

import pandas as pd
import psutil

from src.solver import RCPSPSolver, Problem


def process_instance(file_path: str, input_format: str, method: str, lower_bound: int = None,
                     upper_bound: int = None, time_limit: int = None,
                     queue: multiprocessing.Queue = None):
    """
    Process a single instance of the given path.
    """
    s = RCPSPSolver(Problem(file_path, input_format), method, lower_bound, upper_bound)
    s.encode()
    s.solve(time_limit, find_optimal=True)
    s.verify()
    queue.put(s.get_statistics())


def benchmark(data_set_name: str, method: str, time_limit: int = None):
    """
    Benchmark a dataset.
    """
    logging.info(
        f"Benchmarking dataset: {data_set_name}"
    )
    start = timeit.default_timer()

    bound = pd.read_csv(f'./bound/bound_{data_set_name}.csv')

    stat = pd.DataFrame(columns=['name', 'lower_bound', 'upper_bound', 'variables', 'clauses',
                                 'hard_clauses', 'soft_clauses', 'status', 'makespan',
                                 'encoding_time', 'total_solving_time', 'memory_usage'])
    try:
        for row in bound.itertuples():
            file_path = f'./data_set/{data_set_name}/{row.name}'
            lower_bound = row.lower_bound
            upper_bound = row.upper_bound

            logging.info("_" * 50)
            logging.info(f"Processing instance: {file_path}")

            try:
                queue = multiprocessing.Queue()
                p = multiprocessing.Process(target=process_instance,
                                            args=(file_path,
                                                  'psplib' if data_set_name not in ['pack',
                                                                                    'pack_d'] else 'pack',
                                                  method, lower_bound, upper_bound, time_limit,
                                                  queue))
                p.start()
                peak_memory = 0

                while p.is_alive():
                    try:
                        proc = psutil.Process(p.pid)
                        mem = proc.memory_info().rss
                        for child in proc.children(recursive=True):
                            try:
                                mem += child.memory_info().rss
                            except psutil.NoSuchProcess:
                                continue
                        peak_memory = max(peak_memory, mem)
                    except psutil.NoSuchProcess:
                        break
                    time.sleep(0.1)

                p.join()
                p.terminate()
                stats = queue.get()
                stats['name'] = row.name
                stats['memory_usage'] = round(peak_memory / (1024 ** 2), 5)
                stat = pd.concat([stat, pd.DataFrame([stats])], ignore_index=True)
                stat = stat.drop('file_path', axis=1)
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")

        export_result(data_set_name, method, stat)

        end = timeit.default_timer()

        logging.info(
            f"Finished benchmarking dataset: {data_set_name} in {end - start:.2f} seconds"
        )
    except KeyboardInterrupt:
        logging.error("Benchmarking interrupted by user.")
        export_result(data_set_name, method, stat)


def export_result(data_set_name, method, stat):
    os.makedirs('./result', exist_ok=True)
    stat.to_csv(
        f'./result/{data_set_name}_{method.upper()}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.csv',
        index=False)
    report = pd.DataFrame([{
        'UNSATISFIABLE': stat['status'].str.contains('UNSATISFIABLE').sum(),
        'SATISFIABLE': stat['status'].str.contains('SATISFIABLE').sum(),
        'OPTIMAL': stat['status'].str.contains('OPTIMAL').sum(),
        'UNKNOWN': stat['status'].str.contains('UNKNOWN').sum(),
        'average_encoding_time': stat['encoding_time'].mean(),
        "max_encoding_time": stat['encoding_time'].max(),
        "min_encoding_time": stat['encoding_time'].min(),
        "average_solving_time": stat['total_solving_time'].mean(),
        "max_solving_time": stat['total_solving_time'].max(),
        "min_solving_time": stat['total_solving_time'].min(),
        "average_memory_usage": stat['memory_usage'].mean(),
        "max_memory_usage": stat['memory_usage'].max(),
        "min_memory_usage": stat['memory_usage'].min(),
    }])
    report.T.reset_index().to_csv(
        f'./result/report_{data_set_name}_{method.upper()}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.csv',
        index=False)


def main():
    parser = argparse.ArgumentParser(description='Benchmarking script for SAT encoders.')
    parser.add_argument('dataset_name', type=str, help='The name of the dataset to benchmark.')
    parser.add_argument('method', type=str,
                        choices=['sat', 'maxsat'],
                        help='The type of solver to use.')
    parser.add_argument('--time_limit', type=int, help='Time limit for solving one instance.',
                        default=None)

    args = parser.parse_args()

    benchmark(args.dataset_name, args.method, args.time_limit)


if __name__ == "__main__":
    try:
        main()
    finally:
        if os.path.exists('./out'):
            shutil.rmtree('./out')
        if os.path.exists('./wcnf'):
            shutil.rmtree('./wcnf')
