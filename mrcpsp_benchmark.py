import argparse
import concurrent
import datetime
import logging
import multiprocessing
import os
import re
import shutil
import sys
import time
import timeit
from pathlib import Path

import pandas as pd
import psutil

from src.mrcpsp.problem import MRCPSPProblem
from src.mrcpsp.solver import MRCPSPSolver


def process_instance(file_path: str, time_limit: int = None,
                     queue: multiprocessing.Queue = None):
    """
    Process a single instance of the given path.
    """
    s = MRCPSPSolver(MRCPSPProblem.from_file(file_path))
    s.encode()
    s.solve(time_limit, find_optimal=True)
    s.verify()
    queue.put(s.get_statistics())


def worker(args):
    file_path, time_limit = args

    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=process_instance, args=(file_path, time_limit, queue))
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

    if p.exitcode != 0:
        raise RuntimeError(f"Process for {file_path} failed with exit code {p.exitcode}")

    p.terminate()
    instance_stats = queue.get()
    instance_stats['name'] = file_path.split('/')[-1]
    instance_stats['memory_usage'] = round(peak_memory / (1024 ** 2), 5)
    return instance_stats


def benchmark(data_set_name: str, time_limit: int = None, continue_from: str = None,
              num_concurrent_processes: int = 1, save_interval_seconds: int = 60):
    """
    Benchmark a dataset using concurrent.futures.ProcessPoolExecutor.
    Periodically exports partial results to avoid data loss from unexpected interruptions.
    """
    logging.info(f"Benchmarking dataset: {data_set_name}")

    start_time = timeit.default_timer()

    folder_path = Path(f"./data_set/mrcpsp/{data_set_name}/")
    files = [file.name for file in folder_path.iterdir() if file.is_file()]

    # Regex to capture x and y
    pattern = re.compile(r"j30(\d+)_(\d+)\.mm")

    def extract_numbers(filename):
        match = pattern.match(filename)
        if match:
            x, y = match.groups()
            return int(x), int(y)  # Convert to integers for proper numeric sorting
        return float('inf'), float('inf')  # Push non-matching files to the end

    files.sort(key=extract_numbers)

    # Load existing progress if continuing
    if continue_from is None:
        dataset_stats = pd.DataFrame(columns=[
            'name', 'lower_bound', 'upper_bound', 'variables', 'clauses',
            'status', 'makespan',
            'encoding_time', 'total_solving_time', 'memory_usage'
        ])
    else:
        dataset_stats = pd.read_csv(continue_from)

    tasks = []
    for file in files:
        if file in dataset_stats['name'].values:
            logging.info(f"Skipping {file}, already processed.")
            continue
        tasks.append((f'{folder_path}/{file}', time_limit))

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_concurrent_processes)
    futures = []

    last_save_time = time.time()  # track time of last save

    try:
        for task in tasks:
            futures.append(executor.submit(worker, task))

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                result.pop('file_path', None)
                dataset_stats = pd.concat([dataset_stats, pd.DataFrame([result])],
                                          ignore_index=True)

                # Periodically save partial results
                current_time = time.time()
                if current_time - last_save_time >= save_interval_seconds:
                    logging.info("Saving periodic partial results...")
                    export_result(data_set_name, dataset_stats, suffix="partial")
                    last_save_time = current_time

            except Exception as e:
                logging.error(f"Worker failed: {e}")
                for f in futures:
                    f.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                export_result(data_set_name, dataset_stats, suffix="crash")
                sys.exit(1)

    except KeyboardInterrupt:
        logging.error("Benchmarking interrupted by user. Terminating remaining tasks...")
        for f in futures:
            f.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        export_result(data_set_name, dataset_stats, suffix="interrupted")
        logging.info("Partial results exported after interruption.")
        sys.exit(1)

    # Final export at the end
    export_result(data_set_name, dataset_stats)
    end = timeit.default_timer()
    logging.info(f"Benchmarking completed in {end - start_time:.2f} seconds.")


def export_result(data_set_name, stat, suffix=None):
    """
    Export both detailed results and summary reports.
    `suffix` is appended to the filename to indicate partial or crash saves.
    """
    os.makedirs('./result', exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    suffix = f"_{suffix}" if suffix else ""

    result_path = f'./result/{data_set_name}_STAIRCASE{suffix}_{timestamp}.csv'
    report_path = f'./result/report_{data_set_name}_STAIRCASE{suffix}_{timestamp}.csv'

    stat.to_csv(result_path, index=False)

    report = pd.DataFrame([{
        'UNSATISFIABLE': stat['status'].str.fullmatch('UNSATISFIABLE').sum(),
        'SATISFIABLE': stat['status'].str.fullmatch('SATISFIABLE').sum(),
        'OPTIMAL': stat['status'].str.fullmatch('OPTIMAL').sum(),
        'UNKNOWN': stat['status'].str.fullmatch('UNKNOWN').sum(),
        'average_preprocessing_time': stat.get('preprocessing_time', pd.Series(dtype=float)).mean(),
        'max_preprocessing_time': stat.get('preprocessing_time', pd.Series(dtype=float)).max(),
        'min_preprocessing_time': stat.get('preprocessing_time', pd.Series(dtype=float)).min(),
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
    report.T.reset_index().to_csv(report_path, index=False)
    logging.info(f"Results exported to {result_path} and {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Benchmarking script for SAT encoders.')
    parser.add_argument('dataset_name', type=str, help='The name of the dataset to benchmark.')
    parser.add_argument('--time_limit', type=int, help='Time limit for solving one instance.',
                        default=None)
    parser.add_argument('--continue_from', type=str, help='Result file name to continue from.',
                        default=None)
    parser.add_argument('--cleanup', action='store_true',
                        help='Cleanup temporary files after execution.')
    parser.add_argument('--num_concurrent_processes', type=int, default=1,
                        help='Number of concurrent processes to use for benchmarking.')
    parser.add_argument('--save_interval_seconds', type=int, default=60,
                        help='Interval (in seconds) between periodic partial result exports.')

    args = parser.parse_args()
    try:
        benchmark(args.dataset_name, args.time_limit, args.continue_from,
                  args.num_concurrent_processes, args.save_interval_seconds)

    finally:
        if args.cleanup:
            if os.path.exists('./out'):
                shutil.rmtree('./out')
            if os.path.exists('./dimacs'):
                shutil.rmtree('./dimacs')
            if os.path.exists('./eprime'):
                shutil.rmtree('./eprime')


if __name__ == "__main__":
    main()
