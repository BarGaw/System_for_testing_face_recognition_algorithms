import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from operator import add

from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import Session

from db_pass import db_string
from create_db import Dataset, Img, Person, Clip, Alg_instance, Result
from create_db import Trained_on, Mismatch

from hashlib import md5

from time import perf_counter
from pympler import asizeof 

import pickle

class UnsupportedFormat(Exception):
    def __init__(self, format):
        print(format)

class Connection():

    def __init__(self, connection_string=db_string) -> None:

        self.engine = self.connect_to_db(connection_string)

    def connect_to_db(self, connection_string):
        try:
            engine = create_engine(connection_string)
            engine.connect()
        except Exception as e:
            print('Can\'t connect to database')
            print('Error message:')
            print(e)
        else:
            return engine

    def upload(self, csv_path: str, name: str, desc: str) -> None:
        """
        Upload new dataset
        Arguments:
            csv_path: 
            name:
            desc:
        Returns:
            None
        """
        session = Session(bind=self.engine)

        data_list = pd.read_csv(csv_path)

        ### Add dataset
        if session.query(Dataset).filter_by(name=name).first() is None:
            dataset = Dataset(name=name, description=desc, img_amount=data_list.shape[0])
            session.add(dataset)
            session.commit()
        else:
            dataset = session.query(Dataset).filter_by(name=name).first()
        
        ### Add all persons
        for name in data_list.iloc[:, 0].unique():
            if session.query(Person).filter_by(true_name=name).first() is None:
                person = Person(hash_name=md5(name.encode('ascii')).hexdigest(), true_name=name) 
                session.add(person)
        session.commit()

        ### Add images
        for row in data_list.iterrows():
            name, file, n_photos = row[1]
            person = session.scalars(select(Person).where(Person.true_name == name)).one()

            if n_photos > 1:
                clip = Clip(name=file)
                session.add(clip)
                session.commit()

                for i in range(n_photos):
                    img = Img(ds_id=dataset.id, person_id=person.id, location=file, clip=clip.id, count=i)
                    session.add(img)
                session.commit()
            else:
                img = Img(ds_id=dataset.id, person_id=person.id, location=file)
                session.add(img)
                session.commit()

    def load_img(self, id: int, session=None) -> np.ndarray:
        """
        Load img, based on id
        Arguments:
            id: 
            session:
        Returns:
            img: numpy.ndarray
        """
        if session is None:
            session = Session(bind=self.engine)

        img = session.get(Img, id)
        format = img.location.split('.')[-1]

        if format == 'npz':
            arr = np.load(img.location)
            return arr[list(arr.keys())[0]][:,:,:,img.count]
        elif format == 'jpg':
            return plt.imread(img.location, format='jpeg')
        else:
            raise UnsupportedFormat(format)
   
    def load_many_img(self, list_of_img, n_per_person=None, session=None):
        if session is None:
            session = Session(bind=self.engine)
        
        res_list = []
        for i in list_of_img:
            if type(i) == Img:
                res_list.append(self.load_img(i.id, session))
            elif n_per_person:
                row = []
                for j in i:
                    row.append(self.load_img(j.id, session))

                if len(row) < n_per_person:
                    temp = row * (int(n_per_person / len(row)) + 1)
                    row = temp[0:n_per_person]
                    for idx, k in enumerate(row):
                        row[idx] = row[idx] + np.random.normal(0, 20, k.shape)
                res_list.append(row)
            else:
                row = []
                for j in i:
                    row.append(self.load_img(j.id, session))
                res_list.append(row)

        return res_list

    def create_set_for_model(self, 
                            size: int, 
                            n_photo_per_person:int = 4,
                            dataset_name:str = None,
                            session=None):
        
        if session is None:
            session = Session(bind=self.engine)

        session = Session(bind=self.engine)
        dataset_id = session.query(Dataset.id).where(Dataset.name == dataset_name)
        persons = session.query(Person).where(Person.id == Img.person_id).where(Img.ds_id == dataset_id).distinct().order_by(func.random()).all()

        set_for_model = pd.DataFrame(columns=['person', 'main_photo', 'rest_photos'])

        if len(persons) < size:
            print(f'Max size for dataset exceeded ({len(persons)})')
            print('size set to it')
            size = len(persons)

        for i in persons[:size]:
            imgs = session.query(Img).where(Img.person_id == i.id).distinct().order_by(func.random()).all()
            if len(imgs) > n_photo_per_person + 1:
                new_row = pd.DataFrame([{'person': i, 'main_photo': imgs[0], 'rest_photos': imgs[1:n_photo_per_person + 1]}])
            elif len(imgs) == 1:
                new_row = pd.DataFrame([{'person': i, 'main_photo': imgs[0], 'rest_photos': [imgs[0]]}])
            else:
                main = imgs[0]
                new_row = pd.DataFrame([{'person': i, 'main_photo': main, 'rest_photos': imgs[1:]}])

            set_for_model = pd.concat([set_for_model, new_row], ignore_index=True)

        return set_for_model
  
    def create_unknown_set(self, 
                           size: int, 
                           known_set:pd.DataFrame, 
                           n_photo_per_person:int = 4, 
                           dataset_name:str = None,
                           session=None):
        if session is None:
            session = Session(bind=self.engine)

        session = Session(bind=self.engine)
        dataset_id = session.query(Dataset.id).where(Dataset.name == dataset_name)
        
        known_id = []
        for i in known_set.iloc[:, 0]:
            known_id.append(i.id)

        persons = session.query(Person).where((Person.id == Img.person_id) & (Person.id.notin_(known_id))).where(Img.ds_id == dataset_id).distinct().order_by(func.random()).all()
        unknown_set = pd.DataFrame(columns=['person', 'photos'])


        if len(persons) < size:
            print(f'Max size for dataset exceeded ({len(persons)})')
            print('size set to it')
            size = len(persons)
         

        for i in persons[:size]:
            imgs = session.query(Img).where(Img.person_id == i.id).distinct().order_by(func.random()).all()
            if len(imgs) >= n_photo_per_person:
                new_row = pd.DataFrame([{'person': i, 'photos': imgs[0:n_photo_per_person]}])
            elif len(imgs) == 1:
                new_row = pd.DataFrame([{'person': i, 'photos': [imgs[0]]}])
            else:     
                new_row = pd.DataFrame([{'person': i, 'photos': imgs[0:]}])

            unknown_set = pd.concat([unknown_set, new_row], ignore_index=True)

        return unknown_set

    def prepare(self, dataset_name:str, known_person:int=10, n_known:int=4, unknown_person:int=10, n_unknown:int=4):
        session = Session(bind=self.engine)

        self.known_set = self.create_set_for_model(known_person, n_known, dataset_name, session)
        self.unknown_set = self.create_unknown_set(unknown_person, self.known_set, n_unknown, dataset_name, session)

        self.ks_primary = self.load_many_img(self.known_set.main_photo, session)
        self.ks_secondary = self.load_many_img(self.known_set.rest_photos, n_known, session)

        self.us = self.load_many_img(self.unknown_set.photos, n_unknown, session)
        self.curent_ds = dataset_name

    def train_model(self, model):
        before = perf_counter()
        model.given_learn(self.ks_primary)
        after = perf_counter()

        model.learning_time = after - before
        model.size_of_model = asizeof.asizeof(model)

    def add_model(self, model, description=""):
        session = Session(bind=self.engine)
        
        byte_model = pickle.dumps(model)
        mname = model.name
        algo = Alg_instance(name=mname, description=description, dataset_name=self.curent_ds, model=byte_model) 
        session.add(algo)
        session.commit()

        for i in self.known_set.main_photo:
            t = Trained_on(base_img=i.id, alg=algo.id)
            session.add(t)

        session.commit()

    def load_model(self, model_id=None, model_name=None):
        session = Session(bind=self.engine)

        if model_id:
            byte_model = session.get(Alg_instance, model_id)
        elif model_name:
            byte_model = session.query(Alg_instance).filter_by(name=model_name).first()

        return pickle.loads(byte_model.model)

    def update_model(self, model, model_id):
        pass

    def verify_model(self, model_id=None, model_name=None, visual=False):
        session = Session(bind=self.engine)
        temp_res = Result()
        session.add(temp_res)
        session.commit()

        if model_id is None:
            model_id = session.query(Alg_instance).filter_by(name=model_name).first().id
        model = self.load_model(model_id, model_name)

        ### On known persons
        FTA = 0
        result_on_known = []
        times = 0
        for idx, i in enumerate(self.ks_secondary):
            row = np.zeros(len(self.ks_primary))
            for idy, j in enumerate(i):
                before = perf_counter()
                recognize_person = model.recognize(j)
                after = perf_counter()

                times += (after - before)

                if recognize_person is None:
                    FTA += 1

                    true_index = idy % len(self.known_set.rest_photos.iloc[idx])
                    session.add(Mismatch(from_result=temp_res.id, 
                                         original_img=self.known_set.rest_photos.iloc[idx][true_index].id))
                else:
                    row = list(map(add, row, recognize_person))
                    if idx != np.argmax(row):
                           
                        true_index = idy % len(self.known_set.rest_photos.iloc[idx])

                        session.add(Mismatch(from_result=temp_res.id, 
                                            original_img=self.known_set.rest_photos.iloc[idx][true_index].id,
                                            recognized_as=self.known_set.main_photo.iloc[np.argmax(row)].id))
            result_on_known.append(row)

        if visual:
            plt.figure()
            plt.imshow(result_on_known)
            plt.show()

        ### On unknown persons
        result_on_impostor = []

        for idx, i in enumerate(self.us):
            row = np.zeros(len(self.ks_primary))
            for idy, j in enumerate(i):
                before = perf_counter()
                recognize_person = model.recognize(j)
                after = perf_counter()

                times += (after - before)

                if recognize_person is None:
                    FTA += 1
                else:
                    row = list(map(add, row, recognize_person))
                         
                    true_index = idy % len(self.unknown_set.photos.iloc[idx])

                    session.add(Mismatch(from_result=temp_res.id, 
                                        original_img=self.unknown_set.photos.iloc[idx][true_index].id,
                                        recognized_as=self.known_set.main_photo.iloc[np.argmax(row)].id))
            result_on_impostor.append(row)

        if visual:
            plt.figure()
            plt.imshow(result_on_impostor)
            plt.show()

        ### COUNT FNMR ###
        total_genuine = len(self.ks_secondary[0]) * len(self.ks_primary)
        FNMR = total_genuine
        for i in range(len(result_on_known)):
            FNMR -= result_on_known[i][i]
        FNMR = FNMR / total_genuine

        ### COUNT FMR ###
        total_impostor = len(self.us[0]) * len(self.us)
        FMR = np.sum(result_on_impostor)
        FMR = FMR / total_impostor

        total = total_genuine + total_impostor

        frr = self.FRR(FNMR, (FTA / total))
        far = self.FAR(FMR, (FTA / total))

        mean_time = times/total

        temp_res.alg_instance_id=model_id
        temp_res.learn_time=model.learning_time
        temp_res.avg_recognition_time=mean_time
        temp_res.size_of_model=model.size_of_model
        temp_res.FNMR=FNMR
        temp_res.FMR=FMR
        temp_res.FTA=FTA
        temp_res.FRR=frr
        temp_res.FAR=far

        if visual: 
            print("Name of tested model:    ", model_name)
            print("Training time:           ", model.learning_time)
            print("Average recognition time:", mean_time)
            print("Size of model:           ", model.size_of_model)
            print("")
            print("FNMR: ", FNMR)
            print("FMR:  ", FMR)
            print("FTA:  ", FTA)
            print("FRR:  ", frr)
            print("FAR:  ", far)
                        

        session.commit()

    def FAR(self, FMR, FTA=0.0):
        return FMR * (1 - FTA)
    
    def FRR(self, FNMR, FTA=0.0):
        return FTA + FNMR * (1 - FTA)

