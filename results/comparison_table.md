| Method | Scope | n | Accuracy | Macro-F1 [95% CI] | Precision | Recall (diseased) | Recall (healthy) | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| Rule-based (IP) | overall | 239 | 0.883 | 0.839 [0.780, 0.892] | 0.903 | 0.944 | 0.705 | 0.924 |
| SVM (IP features) | overall | 239 | 0.950 | 0.935 [0.896, 0.968] | 0.972 | 0.961 | 0.918 | 0.981 |
| CNN (MobileNetV2) | overall | 239 | 0.983 | 0.978 [0.954, 0.995] | 0.989 | 0.989 | 0.967 | 0.999 |
| Rule-based (IP) | blackgram | 150 | 0.840 | 0.737 [0.641, 0.824] | 0.873 | 0.932 | 0.500 | 0.864 |
| SVM (IP features) | blackgram | 150 | 0.940 | 0.910 [0.842, 0.962] | 0.958 | 0.966 | 0.844 | 0.963 |
| CNN (MobileNetV2) | blackgram | 150 | 0.973 | 0.960 [0.917, 0.991] | 0.983 | 0.983 | 0.938 | 0.998 |
| Rule-based (IP) | turmeric | 89 | 0.955 | 0.949 [0.896, 0.988] | 0.967 | 0.967 | 0.931 | 0.998 |
| SVM (IP features) | turmeric | 89 | 0.966 | 0.963 [0.915, 1.000] | 1.000 | 0.950 | 1.000 | 1.000 |
| CNN (MobileNetV2) | turmeric | 89 | 1.000 | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 1.000 | 1.000 |
