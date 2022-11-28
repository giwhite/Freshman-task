| lr       | optim   | dim                           | 第一轮之后acc |
| -------- | ------- | ----------------------------- | ------------- |
| 0.0001   | SGD     | same                          | 49.74         |
| 0.002    | RMSprop | same                          | 49.76         |
| 0.0002   | RMSprop | same                          | 50.06，50.50  |
| 0.00001  | RMSProp | same                          | 50.18         |
| 0.000001 | RMSprop | same                          | 50.13         |
| 0.00001  | Adam    | same                          | 50.34         |
| 0.0001   | Adam    | same                          | 49.92         |
| 0.0001   | Adam    | max_len = 240,hidden_dim = 50 | 50.17         |



