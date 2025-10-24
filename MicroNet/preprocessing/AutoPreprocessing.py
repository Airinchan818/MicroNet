import pandas as pd 
import numpy as np 
import traceback

class StandardScaller :
    """
    do scalling data by Standar Deviation values at data.\n 
    how to use : \n 
    import numpy as np \n 
    from LittleLearn.preprocessing import StandardScaller \n 

    a = np.random.rand(10)\n 
    scaller = StandardScaller()

    scaller.fit(a)\n 

    a_scale = scaller.scalling(a) \n 
    invert_scale_a = scaller.inverse_scalling(a_scale)
    """
    def __init__ (self,epsilon=1e-6) :
        self.epsilon = epsilon
        self.std = None 
        self.mean = None 
    
    def fit(self,x) :
        self.mean = np.mean(x,axis=0,dtype=np.float32) 
        variance = np.mean((x - self.mean)**2,axis=0,dtype=np.float32)
        self.std = np.sqrt(variance)
    
    def scaling(self,x) :
        try : 
            if self.mean is None or self.std is None :
                raise RuntimeError("you must fit scaller first")
            return (x - self.mean) / (self.std + self.epsilon)
        except Exception as e :
            e.add_note("do StandardScaller().fit() before do scalling")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 

    
    def inverse_scaling(self,x) :
        try :
            if self.std is None or self.mean is None :
                raise RuntimeError("you must fit scaller first")
            return x * self.std + self.mean
        except Exception as e :
            e.add_note("do StandardScaller().fit() before do scalling")
            traceback.print_exception(type(e),e,e.__traceback__)
            raise 

    def fit_scaling (self,x) :
        self.fit(x) 
        return (x - self.mean) / (self.std + self.epsilon)

class LabelEncoder :
    """
        Label Encoder
        -------------
        its for extract string object class data to numeric representation

        how to use:

            enc=LabelEncoder()

        author: Candra Alpin Gunawan
    """

    def __init__ (self) :
        self.class_idx = dict()
        self.idx_class = dict()
        self.__counter = 0
    
    def fit(self,data : list) :
        """
            for fit or train LabelEncoder, call it with list paramters. 
            example:\n
                data = list() <= its your data in list  
                enc = LabelEncoder()
                enc.fit(data)
        """
        if not isinstance(data,list) :
            raise RuntimeError("Data must be list")
        for ob in data :
            if ob not in self.class_idx:
                self.class_idx[ob] = self.__counter
                self.idx_class[self.__counter] = ob
                self.__counter +=1
    
    def encod (self,x : list) :
        """
            encod for extract data to numeric representation at trained Labelencoder
            example:\n 
                enc = LabelEncoder()
                enc.encod(x)
        """
        if not isinstance(x,list) :
            raise RuntimeError("x must be list")
        result = list()
        for data in x :
            result.append(self.class_idx[data])
        return np.array(result)
    
    def decod (self,x) :
        """
            decod is for decod at numeric to class representation, parameters must be numpy array
            example:\n 
                enc = LabelEncoder()
                data = np.array(data)
                enc.decod(data)
        """
        result = list()
        for i in range(len(x)) :
            result.append(self.idx_class[x[i]])
        return result 
    
    def fit_encod (self,data:list) :
        """
            you can fit and encod a data too. example:
                enc = LabelEncoder()
                enc.fit_encod(data)
        """
        self.fit(data)
        return self.encod(data)

class AutoPreprocessing:
    def __init__(self,data : str,model_target : str = None ):
        if data.endswith(".csv"):
            self.df = pd.read_csv(data)
        elif data.endswith(".xlsx") :  
            self.df = pd.read_excel(data)
        if model_target is None :
            self.target_model = model_target
        else :    
            self.target_model = self.df[model_target].to_list()
            self.df = self.df.drop([model_target],axis=1)

        self.df.columns = self.df.columns.str.lower()
        data_drop = ['dates','day','month','years','year','date',
                     'days','mouths',    "id", "user_id", "transaction_id", "no_invoice", "uuid", "no_rekening",
                    "no", "index", "row_number", "sequence", "rowid",
                    "created_at", "updated_at", "timestamp", "date_created", "time", "log_time",
                    "description", "deskripsi", "comments", "notes", "remarks",
                    "nama", "name", "username", "email", "alamat", "nomor_telepon", "phone", "no_hp",
                    "kode_produk", "kode_transaksi", "product_code", "transaction_code",
                    "url", "serial_number", "sn", "path"]
        for columns in self.df.columns : 
            if columns in data_drop :
                self.df = self.df.drop([columns],axis=1)
        self.encoder = LabelEncoder()
        self.scaller= StandardScaller()
    
    def features_checking (self) :
        for features in self.df.columns:
            target = self.df[features].to_list()
            if isinstance(target[0],str) :
                self.df[features] = self.encoder.fit_encod(target)
    
    def features_scaling (self) :
        for features in self.df.columns :
            values = self.df[features].values 
            if np.max(values) > 1.0 :
                self.df[features] = self.scaller.fit_scaling(self.df[features].values)
    
    def process_data (self) :
        self.features_checking()
        self.features_scaling()
        if self.target_model is None  :
            return self.df
        else :
            return self.df,self.target_model