import gzip
import io
import os
import sys
from pathlib import Path

import numpy as np
import sqlalchemy as sa
from sqlalchemy import (Column, DateTime, Integer, LargeBinary, String,
                        create_engine)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import MetaData

# Recommended naming convention used by Alembic, as various different database
# providers will autogenerate vastly different names making migrations more
# difficult. See: http://alembic.zzzcomputing.com/en/latest/naming.html
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)
Base = declarative_base(metadata=metadata)

from sqlalchemy import event
from sqlalchemy.engine import Engine

# @event.listens_for(Engine, "connect")
# def set_sqlite_pragma(connection, _):
#     cursor = connection.cursor()
#     cursor.execute("PRAGMA journal_mode=WAL")
#     cursor.close()


def get_session_pg(
    username=os.environ.get("PG_USER"),
    password=os.environ.get("PG_PASSWORD"),
    hostname=os.environ.get("PG_HOSTNAME"),
    database_name=os.environ.get("PG_DATABASE"),
    port=5432,
    init: bool = False,
):

    engine = create_engine(
        f"postgresql://{username}:{password}@{hostname}:{port}/{database_name}",
        max_overflow=0,
        pool_pre_ping=True,
        pool_recycle=60,
        pool_size=20,
        connect_args=dict(
            keepalives=1,
            keepalives_count=5,
            keepalives_idle=30,
            keepalives_interval=10,
        ),
    )

    if init:
        Base.metadata.create_all(engine)  # if database does not exist

    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    session = Session()

    return session


def get_session(filename: Path, init: bool = False):

    path = f"sqlite:///{filename}"

    connect_args = dict(timeout=30, check_same_thread=False,)

    engine = create_engine(path, connect_args=connect_args)
    event.listen(engine, "connect", set_sqlite_pragma)

    if init:
        Base.metadata.create_all(engine)  # if database does not exist

    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    session = Session()

    return session


class CompressedString(sa.types.TypeDecorator):
    """Storage datatype for large blobs of text"""

    @staticmethod
    def compress(s):
        if type(s) == str:
            s = s.encode()
        b = gzip.compress(s)
        return b

    @staticmethod
    def decompress(b):
        s = gzip.decompress(b)
        return s

    impl = LargeBinary

    def process_bind_param(self, value, dialect):
        return self.compress(value)

    def process_result_value(self, value, dialect):
        return self.decompress(value)


class NumpyArray(sa.types.TypeDecorator):
    """Storage datatype for numpy arrays"""

    impl = LargeBinary

    @staticmethod
    def compress(s):
        if type(s) == str:
            s = s.encode()
        b = gzip.compress(s)
        return b

    @staticmethod
    def decompress(b):
        s = gzip.decompress(b)
        return s

    @staticmethod
    def save_array(arr):
        s = io.StringIO()
        np.savetxt(s, arr)
        return s.getvalue()

    @staticmethod
    def load_array(txt):
        s = io.StringIO(txt)
        arr = np.loadtxt(s)
        return arr

    def process_bind_param(self, value, dialect):
        value = self.save_array(value)
        return self.compress(value)

    def process_result_value(self, value, dialect):
        value = self.decompress(value)
        value = self.load_array(value)
        return value


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)

    # Basic descriptors
    hashkey = Column(String)
    created = Column(DateTime)
    name = Column(String)

    status = Column(String)

    def __repr__(self) -> str:
        return f"<{self.name}>"
