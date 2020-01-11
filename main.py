from A1.A1 import *
from A2.A2 import *
from B1.B1 import *
from B2.B2 import *
# =====================Data preprocessing==========================================
A2trainData, A2valData, A2testData = preProcessDataA2()
B2trainData, B2valData, B2testData = preProcessDataB2()
# =====================Task A1=====================================================
model_A1 = buildA1()
acc_A1_train = trainA1(model_A1)
acc_A1_test = testA1(model_A1)
del model_A1
# =====================Task A2=====================================================
model_A2 = buildA2()
acc_A2_train = trainA2(model_A2,A2trainData, A2valData)
acc_A2_test = testA2(model_A2,A2testData)
del model_A2, A2trainData, A2valData, A2testData
# =====================Task B1=====================================================
model_B1 = buildB1()
acc_B1_train = trainB1(model_B1)
acc_B1_test = testB1(model_B1)
del model_B1
# =====================Task B2=====================================================
model_B2 = buildB2()
acc_B2_train = trainB2(model_B2,B2trainData, B2valData)
acc_B2_test = testB2(model_B2,B2testData)
del model_B2,B2trainData,B2valData,B2testData
# =====================Print Results===============================================
Format='TA1: {}, {}; TA2: {}, {}; TB1: {}, {}; TB2: {}, {};'
print(Format.format(acc_A1_train, acc_A1_test, acc_A2_train, acc_A2_test,
                                            acc_B1_train, acc_B1_test,
                                            acc_B2_train, acc_B2_test))