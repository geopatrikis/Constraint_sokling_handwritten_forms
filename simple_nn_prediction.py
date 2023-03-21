import torch
import extract_character as extr
import os
from Lenet import LeNet, LeNetChar

# extract file name without extension

people = [
    ['MICHELLE', 'RODRIGUEZ', '12/02/1950', 'RFMR12025033445', 'DE1202507171'],
    ['SIERRA', 'YU', '29/03/1950', 'RFSY29035047838', 'IT2903504950'],
    ['LAURA', 'MENDOZA', '16/10/1952', 'RFLM16105293143', 'BR1610522746'],
    ['ANGELA', 'FITZGERALD', '04/07/1956', 'RFAF04075656622', 'BE0407563203'],
    ['DAVID', 'LOPEZ', '17/01/1963', 'RFDL17016329220', 'BR1701638007'],
    ['HECTOR', 'COOK', '09/02/1964', 'RFHC09026495520', 'IT0902647355'],
    ['TIMOTHY', 'GREEN', '19/08/1968', 'RFTG19086827450', 'BR1908688556'],
    ['ELIZABETH', 'RIVERA', '08/07/1970', 'RFER08077010123', 'BR0807708065'],
    ['DEREK', 'CHURCH', '13/01/1972', 'RFDC13017297192', 'BL1301728110'],
    ['JEREMIAH', 'TUCKER', '11/10/1972', 'RFJT11107251133', 'BR1110727764'],
    ['LEAH', 'FERGUSON', '05/01/1976', 'RFLF05017691772', 'US0501769434'],
    ['KENNETH', 'STONE', '19/01/1977', 'RFKS19017711799', 'DE1901777829'],
    ['EDWIN', 'ALLEN', '22/09/1977', 'RFEA22097775609', 'DE2209776709'],
    ['ANGEL', 'SCHWARTZ', '18/05/1982', 'RFAS18058263146', 'BR1805822377'],
    ['JARED', 'MURRAY', '26/01/1985', 'RFJM26018598155', 'BR2601853435'],
    ['MELISSA', 'MENDOZA', '11/06/1987', 'RFMM11068724346', 'BL1106877773'],
    ['NICOLE', 'CANTU', '19/03/1989', 'RFNC19038962968', 'DE1903891767'],
    ['JAMES', 'HARRISON', '17/02/1990', 'RFJH17029026206', 'BE1702909215'],
    ['JEFFREY', 'MOONEY', '19/04/1990', 'RFJM19049013684', 'BE1904901793'],
    ['LORI', 'NUNEZ', '28/06/1997', 'RFLN28069748554', 'BR2806974343']

]

true_labels = [
    ['NICOLE', 'CANTU', '19/03/1989', 'RFNC19038962968', 'DE1903891767'],
    ['JEREMIAH', 'TUCKER', '11/10/1972', 'RFJT11107251133', 'BR1110727764'],
    ['LORI', 'NUNEZ', '28/06/1997', 'RFLN28069748554', 'BR2806974343'],
    ['SIERRA', 'YU', '29/03/1950', 'RFSY29035047838', 'IT2903504950'],
    ['TIMOTHY', 'GREEN', '19/08/1968', 'RFTG19086827450', 'BR1908688556'],
    ['LAURA', 'MENDOZA', '16/10/1952', 'RFLM16105293143', 'BR1610522746'],
    ['JAMES', 'HARRISON', '17/02/1990', 'RFJH17029026206', 'BE1702909215'],
    ['SIERRA', 'YU', '29/03/1950', 'RFSY29035047838', 'IT2903504950'],
    ['ELIZABETH', 'RIVERA', '08/07/1970', 'RFER08077010123', 'BR0807708065'],
    ['NICOLE', 'CANTU', '19/03/1989', 'RFNC19038962968', 'DE1903891767'],
    ['KENNETH', 'STONE', '19/01/1977', 'RFKS19017711799', 'DE1901777829'],
    ['MICHELLE', 'RODRIGUEZ', '12/02/1950', 'RFMR12025033445', 'DE1202507171'],
    ['DEREK', 'CHURCH', '13/01/1972', 'RFDC13017297192', 'BL1301728110'],
    ['DAVID', 'LOPEZ', '17/01/1963', 'RFDL17016329220', 'BR1701638007'],
    ['JAMES', 'HARRISON', '17/02/1990', 'RFJH17029026206', 'BE1702909215'],
    ['JEREMIAH', 'TUCKER', '11/10/1972', 'RFJT11107251133', 'BR1110727764'],
    ['EDWIN', 'ALLEN', '22/09/1977', 'RFEA22097775609', 'DE2209776709'],
    ['ELIZABETH', 'RIVERA', '08/07/1970', 'RFER08077010123', 'BR0807708065'],
    ['MELISSA', 'MENDOZA', '11/06/1987', 'RFMM11068724346', 'BL1106877773'],
    ['JEREMIAH', 'TUCKER', '11/10/1972', 'RFJT11107251133', 'BR1110727764'],
    ['LORI', 'NUNEZ', '28/06/1997', 'RFLN28069748554', 'BR2806974343'],
    ['SIERRA', 'YU', '29/03/1950', 'RFSY29035047838', 'IT2903504950'],
    ['JAMES', 'HARRISON', '17/02/1990', 'RFJH17029026206', 'BE1702909215'],
    ['ANGEL', 'SCHWARTZ', '18/05/1982', 'RFAS18058263146', 'BR1805822377'],
    ['HECTOR', 'COOK', '09/02/1964', 'RFHC09026495520', 'IT0902647355'],
    ['LEAH', 'FERGUSON', '05/01/1976', 'RFLF05017691772', 'US0501769434'],
    ['JARED', 'MURRAY', '26/01/1985', 'RFJM26018598155', 'BR2601853435'],
    ['NICOLE', 'CANTU', '19/03/1989', 'RFNC19038962968', 'DE1903891767'],
    ['JEFFREY', 'MOONEY', '19/04/1990', 'RFJM19049013684', 'BE1904901793'],
    ['HECTOR', 'COOK', '09/02/1964', 'RFHC09026495520', 'IT0902647355'],
    ['SIERRA', 'YU', '29/03/1950', 'RFSY29035047838', 'IT2903504950'],
    ['LEAH', 'FERGUSON', '05/01/1976', 'RFLF05017691772', 'US0501769434'],
    ['TIMOTHY', 'GREEN', '19/08/1968', 'RFTG19086827450', 'BR1908688556'],
    ['LORI', 'NUNEZ', '28/06/1997', 'RFLN28069748554', 'BR2806974343'],
    ['LEAH', 'FERGUSON', '05/01/1976', 'RFLF05017691772', 'US0501769434'],
    ['JEREMIAH', 'TUCKER', '11/10/1972', 'RFJT11107251133', 'BR1110727764'],
    ['LAURA', 'MENDOZA', '16/10/1952', 'RFLM16105293143', 'BR1610522746'],
    ['ANGELA', 'FITZGERALD', '04/07/1956', 'RFAF04075656622', 'BE0407563203'],
    ['NICOLE', 'CANTU', '19/03/1989', 'RFNC19038962968', 'DE1903891767'],
    ['JEFFREY', 'MOONEY', '19/04/1990', 'RFJM19049013684', 'BE1904901793'],
    ['KENNETH', 'STONE', '19/01/1977', 'RFKS19017711799', 'DE1901777829'],
    ['JARED', 'MURRAY', '26/01/1985', 'RFJM26018598155', 'BR2601853435'],
    ['LORI', 'NUNEZ', '28/06/1997', 'RFLN28069748554', 'BR2806974343'],
    ['JEFFREY', 'MOONEY', '19/04/1990', 'RFJM19049013684', 'BE1904901793']
]


def predict_date(model, characters_for_recognition, labels):
    i = 0
    correct_fields = 0
    for char in labels:
        if char != '/':
            # model(characters_for_recognition[0])
            processed_input = extr.preprocess(characters_for_recognition[i])
            with torch.no_grad():
                logps = model(processed_input)
            # Output of the network are log-probabilities, need to take exponential for probabilities
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            print("Predicted Digit =", probab.index(max(probab)), "\tActual =", char)
            if int(probab.index(max(probab))) == int(char):
                correct_fields += 1
        i += 1
    return correct_fields


def predict_fn(model, characters_for_recognition, labels):
    i = 0
    correct_fields = 0
    for char in labels:
        if char != '/':
            # model(characters_for_recognition[0])
            processed_input = extr.preprocess(characters_for_recognition[i])
            with torch.no_grad():
                logps = model(processed_input)
            # Output of the network are log-probabilities, need to take exponential for probabilities
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            print("Predicted Char =", chr(int(probab.index(max(probab))) + 65), "\tActual =", char)
            if chr(int(probab.index(max(probab))) + 65) == char:
                correct_fields += 1
        i += 1
    return correct_fields, i


def predict_dates():
    print(true_labels[39][2])
    model_digits = LeNet()
    model_digits.load_state_dict(torch.load('saved_models/digits_model_nn3'))
    # extract_characters('C:/Users/30698/Desktop/KULeuven/Capita Selecta/formes/fields/0001_DATE.jpg')
    dir_path = 'C:/Users/30698/Desktop/KULeuven/Capita Selecta/formes/fields'
    # Define the allowed image extensions
    img_extensions = ["DATE.jpg", "DATE.jpeg", "DATE.png"]

    # Initialize an empty list to store the image paths

    # Iterate over all files and subdirectories in the directory
    for root, dirs, files in os.walk(dir_path):
        # Iterate over all files in the current directory
        correct_num = 0
        for file in files:
            # Check if the file has an image extension
            if any(file.endswith(ext) for ext in img_extensions):
                # Add the full path of the image to the list
                img_path = os.path.join(root, file)
                characters_of_date = extr.extract_characters(img_path)
                file_index = int(file[:4])
                print(file_index)
                correct_num += predict_date(model_digits, characters_of_date, true_labels[file_index - 1][2])
        print("Accuracy:", correct_num / (44 * 8))


def predict_first_names():
    print(true_labels[39][0])
    model_chars = LeNetChar()
    model_chars.load_state_dict(torch.load('saved_models/characters_model_nn'))
    # extract_characters('C:/Users/30698/Desktop/KULeuven/Capita Selecta/formes/fields/0001_DATE.jpg')
    dir_path = 'C:/Users/30698/Desktop/KULeuven/Capita Selecta/formes/fields'
    # Define the allowed image extensions
    img_extensions = ["FN.jpg", "FN.jpeg", "FN.png"]

    # Initialize an empty list to store the image paths

    # Iterate over all files and subdirectories in the directory
    for root, dirs, files in os.walk(dir_path):
        # Iterate over all files in the current directory
        correct_fn = 0
        total = 0
        for file in files:
            # Check if the file has an image extension
            if any(file.endswith(ext) for ext in img_extensions):
                # Add the full path of the image to the list
                img_path = os.path.join(root, file)
                characters_of_date = extr.extract_characters(img_path)
                file_index = int(file[:4])
                print(file_index)
                cor, total_new = predict_fn(model_chars, characters_of_date, true_labels[file_index - 1][0])
                correct_fn += cor
                total += total_new
        print("Accuracy:", correct_fn / total)


def get_accurracy_for_fn(fn_string, labels_of_field, model_char):
    boxes_of_field = extr.extract_characters(fn_string)
    i = 0
    correct_fields = 0
    for char in labels_of_field:
        # model(characters_for_recognition[0])
        processed_input = extr.preprocess(boxes_of_field[i])
        with torch.no_grad():
            logps = model_char(processed_input)
        # Output of the network are log-probabilities, need to take exponential for probabilities
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        #print("Predicted Char =", chr(int(probab.index(max(probab))) + 65), "\tActual =", char)
        if chr(int(probab.index(max(probab))) + 65) == char:
            correct_fields += 1
        i += 1
    return correct_fields, i


def get_accurracy_for_ln(ln_string, labels_of_field, model_char):
    return get_accurracy_for_fn(ln_string, labels_of_field, model_char)


def get_accurracy_for_rf(rf_string, labels_of_field, model_char, model_digits):
    boxes_of_field = extr.extract_characters(rf_string)
    i = 0
    correct_fields = 0
    for char in labels_of_field:
        # model(characters_for_recognition[0])
        if(len(boxes_of_field)==i):continue
        processed_input = extr.preprocess(boxes_of_field[i])
        if char.isdigit():
            with torch.no_grad():
                logps = model_digits(processed_input)
            # Output of the network are log-probabilities, need to take exponential for probabilities
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            #print("Predicted Digit =", int(probab.index(max(probab))), "\tActual =", int(char))
            if int(probab.index(max(probab))) == int(char):
                correct_fields += 1
        else:
            with torch.no_grad():
                logps = model_char(processed_input)
            # Output of the network are log-probabilities, need to take exponential for probabilities
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            #print("Predicted Char =", chr(int(probab.index(max(probab))) + 65), "\tActual =", char)
            if chr(int(probab.index(max(probab))) + 65) == char:
                correct_fields += 1
        i += 1
    return correct_fields, i


def get_accurracy_for_date(date_string, labels_of_field, model_digits):
    boxes_of_field = extr.extract_characters(date_string)
    i = 0
    correct_fields = 0
    for char in labels_of_field:
        if char == '/':
            continue
        # model(characters_for_recognition[0])
        processed_input = extr.preprocess(boxes_of_field[i])
        with torch.no_grad():
            logps = model_digits(processed_input)
        # Output of the network are log-probabilities, need to take exponential for probabilities
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        #print("Predicted Digit =", int(probab.index(max(probab))), "\tActual =", int(char))
        if int(probab.index(max(probab))) == int(char):
            correct_fields += 1
        i += 1
    return correct_fields, i


def get_accurracy_for_nat(nat_string, labels_of_field, model_char, model_digits):
    return get_accurracy_for_rf(nat_string, labels_of_field, model_char, model_digits)


def calculate_forms_accuracy(index, folder_path, model_char, model_digits):
    padded_str = '{:04d}'.format(index)
    index-=1
    fn_string = folder_path + "/" + padded_str + "_FN.jpg"
    ln_string = folder_path + "/" + padded_str + "_LN.jpg"
    date_string = folder_path + "/" + padded_str + "_DATE.jpg"
    rf_string = folder_path + "/" + padded_str + "_RF.jpg"
    nat_string = folder_path + "/" + padded_str + "_NAT.jpg"
    form_correct = 0
    form_sum = 0

    # Calculate for First Name
    field_correct, field_total = get_accurracy_for_fn(fn_string, true_labels[index][0], model_char)
    form_correct += field_correct
    form_sum += field_total

    # Calculate for Last Name
    field_correct, field_total = get_accurracy_for_ln(ln_string, true_labels[index][1], model_char)
    form_correct += field_correct
    form_sum += field_total

    # Calculate for Date
    field_correct, field_total = get_accurracy_for_date(date_string, true_labels[index][2], model_digits)
    form_correct += field_correct
    form_sum += field_total

    # Calculate for Reference number
    field_correct, field_total = get_accurracy_for_rf(rf_string, true_labels[index][3], model_char, model_digits)
    form_correct += field_correct
    form_sum += field_total

    # Calculate for National number
    field_correct, field_total = get_accurracy_for_nat(nat_string, true_labels[index][4], model_char, model_digits)
    form_correct += field_correct
    form_sum += field_total
    print("Form:" + padded_str + " Accuracy: " + str(form_correct/form_sum))
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
    print("FINAL ACCURACY= " + str(correct/sum))


if __name__ == '__main__':
    calculate_overall_accuracy()
