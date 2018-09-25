# Usage

```
pipenv install
pipenv run python gen_vectors.py logfile work/vectors
pipenv run python predict.py work/vectors work/predictions
pipenv run python rank.py work/predictions work/rank.csv
pipenv run python plot.py work/vectors work/predictions
```
