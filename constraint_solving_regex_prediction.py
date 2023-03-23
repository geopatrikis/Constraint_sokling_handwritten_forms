import numpy as np
import torch
from Lenet import LeNet, LeNetChar
import db
import simple_nn_prediction as nn_pred
import cpmpy_form_solver as solver

def constraint_nationality(first_letter, second_letter):
    possible_combinations = ["DE", "BR", "BL", "IT", "BE", "US"]
    joint_probabilities = np.zeros((26, 26))
    for combination in possible_combinations:
        joint_probabilities[ord(combination[0]) - 65][ord(combination[1]) - 65] = \
            100 * first_letter[ord(combination[0]) - 65] * (100 * second_letter[ord(combination[1]) - 65])

    first_letter_jp = np.sum(joint_probabilities, axis=1)
    second_letter_jp = np.sum(joint_probabilities, axis=0)
    total = sum(first_letter_jp)
    normalized_fl = [x / total for x in first_letter_jp]
    total = sum(second_letter_jp)
    normalized_sl = [x / total for x in second_letter_jp]
    return list(normalized_fl), list(normalized_sl)


def calculate_forms_accuracy(index, folder_path, model_char, model_digits):
    int("0010")
    padded_str = '{:04d}'.format(index)
    index -= 1
    fn_string = folder_path + "/" + padded_str + "_FN.jpg"
    ln_string = folder_path + "/" + padded_str + "_LN.jpg"
    date_string = folder_path + "/" + padded_str + "_DATE.jpg"
    rf_string = folder_path + "/" + padded_str + "_RF.jpg"
    nat_string = folder_path + "/" + padded_str + "_NAT.jpg"

    probs_date = nn_pred.get_initial_probabilities_for_date(date_string, model_digits)
    probs_fn = nn_pred.get_initial_probabilities_for_string(fn_string, model_char)
    probs_ln = nn_pred.get_initial_probabilities_for_string(ln_string, model_char)
    probs_rf = nn_pred.get_initial_probabilities_for_rf_nat(
        rf_string, db.true_labels[index][3], model_char, model_digits)
    probs_nat = nn_pred.get_initial_probabilities_for_rf_nat(
        nat_string, db.true_labels[index][4], model_char, model_digits)

    #---------------RUN SOLVER AND APPLY RESULTS--------------------#
    done, fl_fn, fl_ln, solvers_rf_10let, solvers_date, solvers_nat_8let = \
        solver.solve_form(probs_fn[0], probs_ln[0], probs_date, probs_rf[:10], probs_nat[:8])

    probs_fn[0][fl_fn] += 10000
    probs_ln[0][fl_ln] += 10000
    for i in range(0,len(probs_date)):
        probs_date[i][solvers_date[i]] += 10000

    for i in range(0,len(solvers_rf_10let)):
        probs_rf[i][solvers_rf_10let[i]] += 10000

    for i in range(2,len(solvers_nat_8let)):
        probs_nat[i][solvers_nat_8let[i]] += 10000

    probs_nat[0], probs_nat[1] = constraint_nationality(probs_nat[0], probs_nat[1])

    #---------------------------------------------------------------#

    form_correct = 0
    form_sum = 0
    # Calculate for First Name
    field_correct, field_total = nn_pred.final_prediction_string(probs_fn, db.true_labels[index][0])
    form_correct += field_correct
    form_sum += field_total

    # Calculate for Last Name
    field_correct, field_total = nn_pred.final_prediction_string(probs_ln, db.true_labels[index][1])
    form_correct += field_correct
    form_sum += field_total

    # Calculate for Date
    field_correct, field_total = nn_pred.final_prediction_date(probs_date, db.true_labels[index][2])
    form_correct += field_correct
    form_sum += field_total

    # Calculate for Reference number
    field_correct, field_total = nn_pred.final_prediction_rf_nat(probs_rf, db.true_labels[index][3])
    form_correct += field_correct
    form_sum += field_total

    # Calculate for National number
    field_correct, field_total = nn_pred.final_prediction_rf_nat(probs_nat, db.true_labels[index][4])
    form_correct += field_correct
    form_sum += field_total
    print("Form:" + padded_str + " Accuracy: " + str(form_correct / form_sum))
    return form_correct, form_sum


def calculate_overall_accuracy():
    dir_path = 'C:/Users/30698/Desktop/KULeuven/Capita Selecta/formes/fields'
    model_chars = LeNetChar()
    model_chars.load_state_dict(torch.load('saved_models/characters_model_nn'))
    model_digits = LeNet()
    model_digits.load_state_dict(torch.load('saved_models/digits_model_nn3'))
    correct = 0
    sum = 0
    for i in range(1, 45):
        form_correct, sum_of_form = calculate_forms_accuracy(i, dir_path, model_chars, model_digits)
        sum += sum_of_form
        correct += form_correct
    print("FINAL ACCURACY= " + str(correct / sum))


if __name__ == '__main__':
    calculate_overall_accuracy()