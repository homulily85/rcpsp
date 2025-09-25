 # RCPSP Solver
 ## Description
This repository contains a solver for the Resource-Constrained Project Scheduling Problem (RCPSP) using SAT.

For methodology and details, refer to the methodology folder in the repository.
## Requirements
- Linux environment
- Python 3.12+

## Installation
First clone the repository:
```bash
git clone https://github.com/homulily85/rcpsp.git
```
Then install the required dependencies:
```bash
pip install -r requirements.txt
```
## Usage

```python
from src.rcpsp.problem import RCPSPProblem
from src.rcpsp.solver import RCPSPSolver
from src.rcpsp.solver import SOLVER_STATUS

problem = RCPSPProblem.from_file('j301_1.sm')

solver = RCPSPSolver(problem)
solver.encode()
status = solver.solve(time_limit=600)

if status == SOLVER_STATUS.OPTIMAL:
    print(f'Optimal makespan: {solver.makespan}')
    for task_id, start_time in solver.start_times.items():
        print(f'Task {task_id} starts at {start_time}')
elif status == SOLVER_STATUS.SATISFIABLE:
    print(f'Feasible makespan: {solver.makespan}')
    for task_id, start_time in solver.start_times.items():
        print(f'Task {task_id} starts at {start_time}')
elif status == SOLVER_STATUS.UNSATISFIABLE:
    print('No feasible solution exists.')
else:
    print('No optimal solution found within the time limit.')
```