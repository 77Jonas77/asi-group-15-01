# asi-ml-template

Template repo do projektu

## Zbiór danych

- Link: <https://archive.ics.uci.edu/dataset/2/adult>
- Data pobrania: 03.10.2025r 16:53

### Licencja zbioru danych

This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.

## Metryka oceny modelu

Jako nasza metryka oceny modelu wybraliśmy **AP** (Average Precision). Decyzja ta jest zmotywowana faktem, że metryka ta znakomicie radzi sobie przy niezbalansowanych zbiorach danych. Dzięki niej będziemy też mogli skupić się na ocenie jakości modelu w przewidywaniu klasy pozytywnej, czyli tego czy dana osoba zarabia >$50k.

Rozważaliśmy również metrykę AUC-ROC, aczkolwiek jak ustaliliśmy, jej wynik może być łatwo zawyżany przez przewagę w ilości wierszy z klasą negatywną (<$50k).
