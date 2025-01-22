from pysat.pb import *
cnf = PBEnc.atmost(lits=[1, 2, 32], weights=[1, 2, 3], bound=3,top_id=99)
print(cnf.clauses)