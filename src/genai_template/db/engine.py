"""Database engine configuration."""

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.pool import ConnectionPoolEntry

from genai_template.config import settings

engine: Engine = create_engine(
    settings.DATABASE_URL,
    echo=False,
)


if engine.dialect.name == "sqlite":

    @event.listens_for(engine, "connect")
    def enable_sqlite_foreign_keys(
        dbapi_connection: DBAPIConnection,
        connection_record: ConnectionPoolEntry,
    ) -> None:
        """Enable SQLite enforcement of declared foreign-key actions.

        Args:
            dbapi_connection:
                Newly opened DBAPI connection.
            connection_record:
                Pool record associated with the connection.
        """

        del connection_record
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
