from utils.data.database_connection import get_engine
from sqlalchemy import text
import pandas as pd

def execute_query(query, params=None, limit=None):
    engine = get_engine()

    # Convert plain string → SQLAlchemy text
    if isinstance(query, str):
        query = text(query)

    # Merge params safely
    final_params = params.copy() if params else {}

    # Apply limit only if needed
    if limit is not None:
        final_params["limit"] = limit

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=final_params)

    return df