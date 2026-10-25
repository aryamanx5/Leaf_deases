| Method | Scope | n | Accuracy | Macro-F1 [95% CI] | Precision | Recall (diseased) | Recall (healthy) | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| Rule-based (IP) | overall | 239 | 0.879 | 0.845 [0.789, 0.895] | 0.931 | 0.904 | 0.803 | 0.939 |
| SVM (IP features) | overall | 239 | 0.975 | 0.966 [0.938, 0.989] | 0.973 | 0.994 | 0.918 | 0.998 |
| CNN (disease MobileNetV2) | overall | 239 | 0.996 | 0.994 [0.982, 1.000] | 0.994 | 1.000 | 0.984 | 1.000 |
| Rule-based (IP) | blackgram | 150 | 0.820 | 0.746 [0.656, 0.821] | 0.903 | 0.864 | 0.656 | 0.879 |
| SVM (IP features) | blackgram | 150 | 0.960 | 0.938 [0.882, 0.981] | 0.959 | 0.992 | 0.844 | 0.995 |
| CNN (disease MobileNetV2) | blackgram | 150 | 0.993 | 0.990 [0.966, 1.000] | 0.992 | 1.000 | 0.969 | 1.000 |
| Rule-based (IP) | turmeric | 89 | 0.978 | 0.974 [0.934, 1.000] | 0.983 | 0.983 | 0.966 | 0.997 |
| SVM (IP features) | turmeric | 89 | 1.000 | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 1.000 | 1.000 |
| CNN (disease MobileNetV2) | turmeric | 89 | 1.000 | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 1.000 | 1.000 |
