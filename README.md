# tennis_predictions

curl -LsSf https://astral.sh/uv/install.sh | sh

uv init --python

--- install dependencies
pandas
numpy
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
h2o
from h2o.automl import H2OAutoML
matplotlib.pyplot

uv add numpy pandas scikit-learn matplotlib pyarrow
uv add h2o==3.46.0.12
uv add --dev notebook
uv run jupyter notebook

uv run python -m ipykernel install --user --name=tennis_pred

uv run jupyter nbconvert --to=script notebooks/Tennis_data_master_table.ipynb

mv notebooks/Tennis_data_master_table.py src/tennis_pred_1/
mv src/tennis_pred_1/Tennis_data_master_table.py src/tennis_pred_1/train.py

cd /workspaces/tennis_predictions/tennis_pred_1
uv run python src/tennis_pred_1/train.py

Refactoring:
- Klassen
- Pipelines
- Daten interierbar machen, z.B. Jahr/Monat im Download-Link
- Docstrings einfügen
- Error handling einbauen
- Add logger (loguru)
Testing