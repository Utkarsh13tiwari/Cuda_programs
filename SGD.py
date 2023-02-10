import numpy as np
import cudf
from cuml.solvers import SGD as cumlSGD
X = cudf.DataFrame()
X['col1'] = np.array([1,1,2,2,2,5,6,53.5], dtype=np.float32)
X['col2'] = np.array([1,2,2,3,5,6,6,52], dtype=np.float32)
y = cudf.Series(np.array([1, 1, 2, 2,5,7,1,8], dtype=np.float32))
pred_data = cudf.DataFrame()
pred_data['col1'] = np.asarray([3,3], dtype=np.float32)
pred_data['col2'] = np.asarray([4,5], dtype=np.float32)
cu_sgd = cumlSGD(learning_rate='constant', eta0=0.005, epochs=2000,
                 fit_intercept=True, batch_size=2,
                 tol=0.0, penalty='none', loss='squared_loss')
cu_sgd.fit(X, y)

cu_pred = cu_sgd.predict(pred_data).to_numpy()
print(" cuML intercept : ", cu_sgd.intercept_) 

print(" cuML coef : ", cu_sgd.coef_) 

print("cuML predictions : ", cu_pred) 

#output
'''
cuML intercept :  2.04923677444458
 cuML coef :  0    0.949136
1   -0.755925
dtype: float32
cuML predictions :  [1.8729436 1.1170187]

'''
