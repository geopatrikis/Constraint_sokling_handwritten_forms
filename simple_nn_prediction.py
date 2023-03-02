import torch
import extract_character as extr
import os
from Lenet import LeNet
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


true_labels=[
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


def predict_date(model,characters_for_recognition,labels):
    i=0
    correct_fields=0
    for char in labels:
        if char != '/':
            #model(characters_for_recognition[0])
            processed_input = extr.preprocess(characters_for_recognition[i])
            with torch.no_grad():
                logps = model(processed_input)
            # Output of the network are log-probabilities, need to take exponential for probabilities
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            print("Predicted Digit =", probab.index(max(probab)), "\tActual =", char)
            if int(probab.index(max(probab))) == int(char):
                correct_fields+=1
        i += 1
    return correct_fields


if __name__ == '__main__':
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
        correct=0
        for file in files:
            # Check if the file has an image extension
            if any(file.endswith(ext) for ext in img_extensions):
                # Add the full path of the image to the list
                img_path = os.path.join(root, file)
                characters_of_date = extr.extract_characters(img_path, model_digits)
                file_index = int(file[:4])
                print(file_index)
                correct+=predict_date(model_digits, characters_of_date, true_labels[file_index-1][2])
        print("Accuracy:", correct/(44*8))
