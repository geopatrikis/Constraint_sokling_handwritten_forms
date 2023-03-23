import torch
import extract_character as extr
from Lenet import LeNet, LeNetChar
import db


# extract file name without extension
def get_prediction_probabilities(model, processed_input):
    with torch.no_grad():
        logps = model(processed_input)
    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    return probab


def take_max_probability_evaluate_char(list_of_probs, actual_char):
    return chr(int(list_of_probs.index(max(list_of_probs))) + 65) == actual_char


def take_max_probability_evaluate_digit(list_of_probs, actual_char):
    return int(list_of_probs.index(max(list_of_probs))) == int(actual_char)


def get_initial_probabilities_for_string(string, model_char):
    boxes_of_field = extr.extract_characters(string)
    i = 0
    probs_list = []
    for img_char in boxes_of_field:
        # model(characters_for_recognition[0])
        processed_input = extr.preprocess(img_char)
        probab = get_prediction_probabilities(model_char, processed_input)
        probs_list.append(probab)
        i += 1
    return probs_list


def final_prediction_string(probs_list, labels_of_field):
    i = 0
    correct_fields = 0
    for char in labels_of_field:
        probab = probs_list[i]
        # print("Predicted Digit =", int(probab.index(max(probab))), "\tActual =", int(char))
        if take_max_probability_evaluate_char(probab, char):
            correct_fields += 1
        i += 1
    return correct_fields, i


def get_initial_probabilities_for_rf_nat(string, labels_of_field, model_char, model_digits):
    boxes_of_field = extr.extract_characters(string)
    i = 0
    correct_fields = 0
    probab_list = []
    for char in labels_of_field:
        # model(characters_for_recognition[0])
        if len(boxes_of_field) == i:
            continue
        processed_input = extr.preprocess(boxes_of_field[i])
        if char.isdigit():
            probab = get_prediction_probabilities(model_digits, processed_input)
        else:
            probab = get_prediction_probabilities(model_char, processed_input)
        probab_list.append(probab)
        i += 1
    return probab_list


def final_prediction_rf_nat(probs_list, labels_of_field):
    i = 0
    correct_fields = 0
    for char in labels_of_field:
        if i >= len(probs_list):
            continue
        probab = probs_list[i]
        if char.isdigit():
            if take_max_probability_evaluate_digit(probab, char):
                correct_fields += 1
        else:
            if take_max_probability_evaluate_char(probab, char):
                correct_fields += 1
            print("Predicted: "+chr(int(probab.index(max(probab))) + 65) + " Actual:"+char)
        i += 1
    return correct_fields, i


def get_initial_probabilities_for_date(date_string, model_digits):
    boxes_of_field = extr.extract_characters(date_string)
    i = 0
    probs_list = []
    for img_char in boxes_of_field:
        if i == 2 or i == 5:
            i += 1
            continue
        # model(characters_for_recognition[0])
        processed_input = extr.preprocess(img_char)
        probab = get_prediction_probabilities(model_digits, processed_input)
        probs_list.append(probab)
        i += 1
    return probs_list


def final_prediction_date(probs_list, labels_of_field):
    i = 0
    correct_fields = 0
    for char in labels_of_field:
        if char == '/':
            continue
        probab = probs_list[i]
        #print("Predicted Digit =", int(probab.index(max(probab))), "\tActual =", int(char))
        if take_max_probability_evaluate_digit(probab, char):
            correct_fields += 1
        i += 1
    return correct_fields, i


def get_accurracy_for_nat(nat_string, labels_of_field, model_char, model_digits):
    return get_initial_probabilities_for_rf_nat(nat_string, labels_of_field, model_char, model_digits)


def calculate_forms_accuracy(index, folder_path, model_char, model_digits):
    padded_str = '{:04d}'.format(index)
    index -= 1
    fn_string = folder_path + "/" + padded_str + "_FN.jpg"
    ln_string = folder_path + "/" + padded_str + "_LN.jpg"
    date_string = folder_path + "/" + padded_str + "_DATE.jpg"
    rf_string = folder_path + "/" + padded_str + "_RF.jpg"
    nat_string = folder_path + "/" + padded_str + "_NAT.jpg"

    probs_date = get_initial_probabilities_for_date(date_string, model_digits)
    probs_fn = get_initial_probabilities_for_string(fn_string, model_char)
    probs_ln = get_initial_probabilities_for_string(ln_string, model_char)
    probs_rf = get_initial_probabilities_for_rf_nat(rf_string, db.true_labels[index][3], model_char, model_digits)
    probs_nat = get_initial_probabilities_for_rf_nat(nat_string, db.true_labels[index][4], model_char, model_digits)

    form_correct = 0
    form_sum = 0


    # Calculate for First Name
    field_correct, field_total = final_prediction_string(probs_fn, db.true_labels[index][0])
    form_correct += field_correct
    form_sum += field_total

    # Calculate for Last Name
    field_correct, field_total = final_prediction_string(probs_ln, db.true_labels[index][1])
    form_correct += field_correct
    form_sum += field_total

    # Calculate for Date
    field_correct, field_total = final_prediction_date(probs_date, db.true_labels[index][2])
    form_correct += field_correct
    form_sum += field_total

    # Calculate for Reference number
    field_correct, field_total = final_prediction_rf_nat(probs_rf, db.true_labels[index][3])
    form_correct += field_correct
    form_sum += field_total

    # Calculate for National number
    field_correct, field_total = final_prediction_rf_nat(probs_nat, db.true_labels[index][4])
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
