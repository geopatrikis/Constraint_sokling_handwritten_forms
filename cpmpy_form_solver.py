import numpy as np
from cpmpy import *

first_letter_first_name = intvar(0, 25, shape=1, name="first_letter_first_name")
first_letter_last_name = intvar(0, 25, shape=1, name="first_letter_last_name")
rf = intvar(0, 25, shape=(10,), name="rf")
date = intvar(0, 9, shape=(8,), name="date")
nat = intvar(0, 25, shape=(8,), name="nat")
model = Model(

    # Constraints on values (cells that are not empty)
    first_letter_first_name == rf[2],
    first_letter_last_name == rf[3],
    date[0] == rf[4],
    date[0] == nat[2],
    date[1] == rf[5],
    date[1] == nat[3],
    date[2] == rf[6],
    date[2] == nat[4],
    date[3] == rf[7],
    date[3] == nat[5],
    date[6] == rf[8],
    date[7] == rf[9],
    date[6] == nat[6],
    date[7] == nat[7],
    rf[0] == 17,
    rf[1] == 5,
    rf[4:] < 10,
    nat[2] < 10,
    date[4] * 1000 + date[5] * 100 + date[6] * 10 + date[7] > 1900,
    date[4] * 1000 + date[5] * 100 + date[6] * 10 + date[7] < 2023,
    date[0] * 10 + date[1] < 32,
    date[0] * 10 + date[1] > 0,
    date[2] * 10 + date[3] < 13,
    date[2] * 10 + date[3] > 0
    # Constraints on rows and columns
)


def solve_form(fn_let_probs, ln_let_probs, date_probs, rf_probs, nat_probs):
    for i in range(4, 10):
        for j in range(16):
            rf_probs[i].append(0.0)
    for i in range(2, 8):
        for j in range(16):
            nat_probs[i].append(0.0)

    """The objective function is to maximize the probabilites that are given as input from the neural networks"""

    obj = sum([[fn_let_probs[i] * (first_letter_first_name == i) for i in range(0, 26)],
               [ln_let_probs[i] * (first_letter_last_name == i) for i in range(0, 26)],
               sum([date_probs[j][i] * (date[j] == i) for i in range(0, 10)] for j in range(0, 8)),
               sum([rf_probs[j][i] * (rf[j] == i) for i in range(0, 26)] for j in range(0, 10)),
               sum([nat_probs[j][i] * (nat[j] == i) for i in range(0, 26)] for j in range(0, 8))
               ])
    m = Model(model.constraints, maximize=obj)

    if m.solve():
        return True, first_letter_first_name.value(), first_letter_last_name.value(), rf.value(), date.value(), nat.value()
    else:
        return False, None, None, None, None, None
