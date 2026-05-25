from sqlalchemy import create_engine
from config.database import DATABASE_CONFIG

_engine = None
def get_engine():
    global _engine

    if _engine is None:
        cfg = DATABASE_CONFIG

        db_url = (
            f"mysql+pymysql://{cfg['user']}:"
            f"{cfg['password']}@"
            f"{cfg['host']}:"
            f"{cfg['port']}/"
            f"{cfg['database']}"
        )

        _engine = create_engine(db_url)

    return _engine
