import gspread
import re
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

# use creds to create a client to interact with the Google Drive API
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
client = gspread.authorize(creds)

allsheets = client.open("RWE literature review summary - Survival analysis tools")
sheet    = allsheets.get_worksheet(1)
listOfHashes = sheet.get_all_records()

print(len(listOfHashes))
print("Number of records is : {}".format(len(listOfHashes)))

def getUnique(sheet, columnIndexNumber, separator = False, filters = None):
    """gets the unique objects from the columns of a google spreadsheet. 
    If a separator is passed (either list of string), then that separator is used 
    to separate objects within a field.

    Returns a dataframe containing number & frequency of the objects mentioned 
    Args:
        sheet (google sheet object): Google Sheets object fetched using the gspread API
        columnIndexNumber (int): Column number (STARTS FROM 1) of the target column
        separator (bool, optional): If False, then no separator. If list or string, 
                                    uses those string(s) to separate the field
        NOTE: separator only handles two objects in a list for now 
    """
    try: 

        assert isinstance(columnIndexNumber, int) is True
        column = sheet.col_values(columnIndexNumber)
        mapped = [item.lower() for item in column]
        tempDict = {}

        if separator is False:
            pass
        else:
            if isinstance(separator, str):
                sepStr = ' ' + separator + '|' + separator
            if isinstance(separator, list):
                assert len(separator) == 2
                sepStr = ' ' + separator[0] + '|' + separator[0] + '| ' + separator[1] + '|' + separator[1]
        # print(mapped)
        for field in mapped: 
            tempList = re.split(sepStr, field)
            if len(tempList) != 0: 
                for item in tempList:
                    # print(item, tempList)
                    if (item is '') or (item is ' '): 
                        continue
                    if item[0] is ' ':
                        item = item[1:]
                    if item[-1] is ' ':
                        item = item[0:len(item) - 2]

                    if filters is not None: 
                        # print(item)
                        for filterwords in list(filters.keys()): 
                            # print(filterwords, 'here')
                            item = item.replace(filterwords, filters[filterwords])
                            if item[0] is ' ':
                                item = item[1:]
                        # print(item)

                    if item not in tempDict:
                        tempDict[item] = 1
                    else: 
                        tempDict[item] += 1

        return tempDict
    except Exception as e: 
        print("get unique instances failed becasue {}".format(e))

modelFilterDict = {
    'estimate' : 'estimates',

    
}

models = getUnique(sheet, 5, separator = [',', ';'])
sorted_models = sorted(models.items(), key=lambda kv: kv[1])

testFilterDict = {
    'median ': '', 
    'real-world ': '', 
    'real world': '',
    'progression-free' : 'progression free'
}

tests = getUnique(sheet, 6, separator = [',', ';'], filters = testFilterDict)
sorted_tests = sorted(tests.items(), key = lambda kv : kv[1])

print(sorted_models[-20:], len(sorted_models))
print(sorted_tests[-20:], len(sorted_tests))


modelsDF = pd.DataFrame(sorted_models)
modelsDF.columns = ['technique', 'count']
modelsDF = modelsDF.sort_values(by = ['count'], ascending = False)

testsDF = pd.DataFrame(sorted_tests)
testsDF.columns = ['technique', 'count']
testsDF = testsDF.sort_values(by = ['count'], ascending = False)
modelFilePath = '../data/final/sortedModels.csv'
testFilePath = '../data/final/sortedTests.csv'

modelsDF.to_csv(modelFilePath)
testsDF.to_csv(testFilePath)
print(modelsDF[:10])
# newsheetModels = allsheets.add_worksheet(title="Summary of Models", rows="100", cols="2")

# print(sheet.col_values(4))