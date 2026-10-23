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