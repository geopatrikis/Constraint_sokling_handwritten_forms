import numpy as np
import torch
import extract_character as extr
import os
from Lenet import LeNet, LeNetChar
import db
import simple_nn_prediction as nn_pred


def constraint_same_char(input_prob):
    results = []
    for i in range(len(input_prob[0])):
        letter_joint = 1
        for fields_prob in input_prob:
            letter_joint = letter_joint * fields_prob[i]
        results.append(letter_joint)
    # Normalize the results so that they sum to 1
    total = sum(results)
    normalized_results = [x / total for x in results]
    return normalized_results


def constraint_nationality(first_letter, second_letter):
    possible_combinations = ["DE", "BR", "BL", "IT", "BE", "US"]
    joint_probabilities = np.zeros((26, 26))
    for combination in possible_combinations:
        joint_probabilities[ord(combination[0])-65][ord(combination[1])-65] =\
            100*first_letter[ord(combination[0])-65]*(100*second_letter[ord(combination[1])-65])

    first_letter_jp = np.sum(joint_probabilities, axis=1) #TODO check for digits as well *first_letter
    second_letter_jp = np.sum(joint_probabilities, axis=0) #*second_letter
    total = sum(first_letter_jp)
    normalized_fl = [x / total for x in first_letter_jp]
    total = sum(second_letter_jp)
    normalized_sl = [x / total for x in second_letter_jp]
    return list(normalized_fl), list(normalized_sl)


def calculate_forms_accuracy(index, folder_path, model_char, model_digits):
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

    form_correct = 0
    form_sum = 0

    # Constraints for same char in First name Last Nam and RF

    constraint_decision_fn_rf = constraint_same_char([probs_fn[0], probs_rf[2]])
    constraint_decision_ln_rf = constraint_same_char([probs_ln[0], probs_rf[3]])
    probs_fn[0] = constraint_decision_fn_rf
    probs_rf[2] = constraint_decision_fn_rf
    probs_ln[0] = constraint_decision_ln_rf
    probs_rf[3] = constraint_decision_ln_rf

    # Constraints for same char in days

    constraint_decision_date_d1 = constraint_same_char([probs_date[0], probs_rf[4], probs_nat[2]])
    probs_date[0] = constraint_decision_date_d1
    probs_rf[4] = constraint_decision_date_d1
    probs_nat[2] = constraint_decision_date_d1
    constraint_decision_date_d2 = constraint_same_char([probs_date[1], probs_rf[5], probs_nat[3]])
    probs_date[1] = constraint_decision_date_d2
    probs_rf[5] = constraint_decision_date_d2
    probs_nat[3] = constraint_decision_date_d2

    # Constraints for same char in months

    constraint_decision_date_m1 = constraint_same_char([probs_date[2], probs_rf[6], probs_nat[4]])
    probs_date[2] = constraint_decision_date_m1
    probs_rf[6] = constraint_decision_date_m1
    probs_nat[4] = constraint_decision_date_m1

    constraint_decision_date_m2 = constraint_same_char([probs_date[3], probs_rf[7], probs_nat[5]])
    probs_date[3] = constraint_decision_date_m2
    probs_rf[7] = constraint_decision_date_m2
    probs_nat[5] = constraint_decision_date_m2

    # Constraints for same char in years

    constraint_decision_date_y1 = constraint_same_char([probs_date[6], probs_rf[8], probs_nat[6]])
    probs_date[6] = constraint_decision_date_y1
    probs_rf[8] = constraint_decision_date_y1
    probs_nat[6] = constraint_decision_date_y1

    constraint_decision_date_y2 = constraint_same_char([probs_date[7], probs_rf[9], probs_nat[7]])
    probs_date[7] = constraint_decision_date_y2
    probs_rf[9] = constraint_decision_date_y2
    probs_nat[7] = constraint_decision_date_y2

    # DE BR BL IT BE US
    # Constraints on nationality
    probs_nat[0] , probs_nat[1] =constraint_nationality(probs_nat[0] , probs_nat[1])

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
