import ultraimport
test_architecture = ultraimport('../scripts/model_test.py', 'test_architecture')
test_cases = ultraimport('../scripts/model_test.py', 'test_cases')

test_architecture()
test_cases()
print("The generated model has successfully passed both the architecture check and the test cases check.")