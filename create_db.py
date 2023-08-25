from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, LargeBinary
from sqlalchemy import ForeignKey

from db_pass import db_string

Base = declarative_base()

class Dataset(Base):
    __tablename__ = 'Dataset'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True)
    description = Column(String(200))
    img_size = Column(Integer)
    img_amount = Column(Integer)

class Img(Base):
    __tablename__ = 'Img'
    id = Column(Integer, primary_key=True)
    ds_id = Column(Integer, ForeignKey('Dataset.id'))
    person_id = Column(Integer, ForeignKey('Person.id'))
    location = Column(String(500))
    w = Column(Integer, nullable=True)
    h = Column(Integer, nullable=True)
    space = Column(Integer, nullable=True)
    clip = Column(Integer, ForeignKey('Clip.id'), nullable=True)
    count = Column(Integer, nullable=True)

class Person(Base):
    __tablename__ = 'Person'
    id = Column(Integer, primary_key=True)
    hash_name = Column(String(50), unique=True)
    true_name = Column(String(50), nullable=True)

class Alg_instance(Base):
    __tablename__ = 'Alg_instance'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True)
    description = Column(String(500))
    dataset_name = Column(String(50), ForeignKey('Dataset.name'))
    model = Column(LargeBinary)

class Result(Base):
    __tablename__ = 'Result'
    id = Column(Integer, primary_key=True)
    alg_instance_id = Column(Integer, ForeignKey('Alg_instance.id'))
    learn_time = Column(Float)
    avg_recognition_time = Column(Float)
    size_of_model = Column(Integer)
    FNMR = Column(Float)
    FMR = Column(Float)
    FTA = Column(Float)
    FRR = Column(Float)
    FAR = Column(Float)
    F1 = Column(Float)
    Accuracy = Column(Float)

class Trained_on(Base):
    __tablename__ = 'Trained_on'
    id = Column(Integer, primary_key=True) 
    base_img = Column(Integer, ForeignKey('Img.id'))
    alg = Column(Integer, ForeignKey('Alg_instance.id'))

class Mismatch(Base):
    __tablename__ = 'Mismatch'
    id = Column(Integer, primary_key=True) 
    from_result = Column(Integer, ForeignKey('Result.id'))
    original_img = Column(Integer, ForeignKey('Img.id'))
    recognized_as = Column(Integer, ForeignKey('Img.id'))

class Clip(Base):
    __tablename__ = 'Clip'
    id = Column(Integer, primary_key=True)
    name = Column(String(500), unique=True)

def create_db():
    """
    Create database based on classes
    """
    try:
        engine = create_engine(db_string)
        engine.connect()
    except Exception as e:
        print('Can\'t connect to database')
        print('Error message:')
        print(e)
    else:
        Base.metadata.create_all(engine)
        print('Connected to database')
        return engine

# create_db()