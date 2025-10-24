import sys 
from PyQt5.QtWidgets import (QApplication,QWidget,QPushButton,QLabel,QVBoxLayout,
                             QLineEdit,QFileDialog,QErrorMessage, QTableWidget, QTableWidgetItem,
                             QMessageBox,QRadioButton,QButtonGroup) 

import NeuralNetwork as nn 
from optimizers import AdamW
from preprocessing import AutoPreprocessing,LabelEncoder
import joblib 
import matplotlib.pyplot as plt 
import numpy as np 
from typing import Literal
import pandas as pd 

class NeuralNetwork :
    def __init__ (self,n_target,variant : Literal['regression','classification'],
                  complexity_level : Literal[1,2,3] = 1,epochs : int = 100 ,Name = None ) :
        self.loss = None 
        self.Name = Name 
        self.accuracy = None 
        self.model = None 
        self.epochs = epochs
        self.model_variant = variant
        self.level = complexity_level
        self.n_target = n_target
    
    def fit(self,x_train,y_train) :
        x = x_train.copy()
        y = y_train.copy()
        if self.model_variant == 'regression' :
            loss_fn = nn.MeanAbsoluteError()
            optimizer = AdamW()
            if y.ndim < 2 :
                y = y.reshape(-1,1)
            if self.model is None :
                if self.level == 1 :
                    self.model = nn.Sequential([
                        nn.Linear(64),
                        nn.ReLU(),
                        nn.Linear(128),
                        nn.ReLU(),
                        nn.Linear(256),
                        nn.ReLU(),
                        nn.Linear(self.n_target)
                    ])
                elif self.level == 2 :
                    self.model = nn.Sequential([
                        nn.Linear(128),
                        nn.ReLU(),
                        nn.Linear(256),
                        nn.ReLU(),
                        nn.Linear(256),
                        nn.ReLU(),
                        nn.Linear(512),
                        nn.ReLU(),
                        nn.Linear(1024),
                        nn.ReLU(),
                        nn.Linear(self.n_target)
                    ])
                elif self.level == 3 :
                    self.model = nn.Sequential([
                        nn.Linear(256),
                        nn.ReLU(),
                        nn.Linear(256),
                        nn.ReLU(),
                        nn.Linear(512),
                        nn.ReLU(),
                        nn.Linear(512),
                        nn.ReLU(),
                        nn.Linear(1024),
                        nn.ReLU(),
                        nn.Linear(1024),
                        nn.ReLU(),
                        nn.Linear(self.n_target)
                    ])
                else :
                    raise RuntimeError("Level just support 1,2,3 level")
            
            self.loss = list()
            
            for epoch in range(self.epochs) :
                y_pred = self.model(x)
                loss = loss_fn(y,y_pred)
                grad = loss_fn.backward()
                self.model.backward(grad)
                optimizer.apply_weight(self.model.get_weight())
                optimizer.apply_grad(self.model.get_gradient())
                n_weight = optimizer.step()
                self.model.update_weight(n_weight)
                self.loss.append(loss)
                print(f"[epoch : {epoch + 1}/ {self.epochs } || loss : {loss:.6f}]")
                if loss <= 0.1 :
                    break
                
        elif self.model_variant == 'classification' :
            
            if np.max(y)> 1 :
                loss_fn = nn.SparsecategoricalCrossentropy()
                last_activation = nn.Softmax(axis=-1,keepdim=True)
            else :
                loss_fn = nn.BinaryCrossEntropy()
                last_activation = nn.Sigmoid()
                y = y.reshape(-1,1)
                self.n_target = 1 
            optimizer = AdamW(clipnorm=5.0)
            
            if self.model is None :
                if self.level ==  1 :
                    self.model = nn.Sequential([
                        nn.Linear(32),
                        nn.ReLU(),
                        nn.Linear(64),
                        nn.ReLU(),
                        nn.Linear(128),
                        nn.ReLU(),
                        nn.Linear(self.n_target),
                        last_activation
                    ])
                
                elif self.level == 2 :
                    self.model = nn.Sequential([
                        nn.Linear(64),
                        nn.ReLU(),
                        nn.Linear(128),
                        nn.ReLU(),
                        nn.Linear(128),
                        nn.ReLU(),
                        nn.Linear(256),
                        nn.ReLU(),
                        nn.Linear(self.n_target),
                        last_activation
                    ])
                
                elif self.level == 3 :
                    self.model = nn.Sequential([
                        nn.Linear(128),
                        nn.ReLU(),
                        nn.Linear(256),
                        nn.ReLU(),
                        nn.Linear(256),
                        nn.ReLU(),
                        nn.Linear(512),
                        nn.ReLU(),
                        nn.Linear(512),
                        nn.ReLU(),
                        nn.Linear(self.n_target),
                        last_activation
                    ])
            self.loss = list()
            self.accuracy = list()
            
            for epoch in range(self.epochs) :
                y_pred = self.model(x_train)
                loss = loss_fn(y,y_pred)
                if isinstance(last_activation,nn.Softmax) :
                    y_ = np.argmax(y_pred,axis=-1)
                    accuracy = np.mean(np.where(y_ == y,1,0))
                else :
                    y_ = np.where(y_pred > 0.5,1,0)
                    accuracy = np.mean(np.where(y_==y,1,0))
                grad = loss_fn.backward()
                self.model.backward(grad)
                optimizer.apply_weight(self.model.get_weight())
                optimizer.apply_grad(self.model.get_gradient())
                n_weight = optimizer.step()
                self.model.update_weight(n_weight)
                self.loss.append(loss)
                self.accuracy.append(accuracy)
                print(f"[epoch : {epoch + 1} / {self.epochs} || loss : {loss:.6f} || accuracy : {accuracy:.6f}]")
                if loss <= 0.05 or accuracy >= 0.95 :
                    break 
                
    
    def predict(self,x) :
        return self.model(x)

    def plot_eval (self) :
        if self.Name == None :
            plt.title("Model Evaluation")
        else :
            plt.title(f"{self.Name} Evaluation")
        
        plt.xlabel("timestep per epoch")
        plt.ylabel("values")

        if self.model_variant == 'regression' :
            plt.plot(self.loss,label='loss',color='red')
            plt.grid(True)
            plt.legend()
            plt.show()
        else :
            plt.plot(self.loss,label='loss',color='red')
            plt.plot(self.accuracy,label='accuracy',color='green')
            plt.grid(True)
            plt.legend()
            plt.show()



class Model_Trainers (QWidget) : 
    def __init__ (self ) :
        super().__init__()
        self.df = None 
        self.setWindowTitle("Model Trainer")
        self.setGeometry(0,0,1380,720) 
        layout = QVBoxLayout()
        button_file_uploader = QPushButton("file target training Upload")
        self.table = QTableWidget()
        self.file_searcher = QFileDialog()
        self.label_target = QLabel("Non target")
        button_submit_target = QPushButton("Submit target")
        button_submit_target.clicked.connect(self.submit_target)
        button_file_uploader.clicked.connect(self.file_finder)
        button_model_eval = QPushButton("Look model loss/ac")
        button_model_eval.clicked.connect(self.plot_model_trained)
        button_train = QPushButton("Train Model") 
        button_train.clicked.connect(self.train_Model)
        self.checker_classification = QRadioButton("Classification")
        self.checker_regression = QRadioButton("Regression")
        self.levelgroup = QButtonGroup()
        self.levelgroup.setExclusive(True)
        self.level_1_checker = QRadioButton("complex lv1")
        self.level2_checker = QRadioButton("complex lv2")
        self.level_3_checker = QRadioButton("complex lv3")
        self.levelgroup.addButton(self.level_1_checker)
        self.levelgroup.addButton(self.level2_checker)
        self.levelgroup.addButton(self.level_3_checker)
        self.target_writter = QLineEdit()

        self.target_writter.setPlaceholderText("Type columns name target Training model ")
        layout.addWidget(button_file_uploader)
        layout.addWidget(self.target_writter)
        layout.addWidget(button_submit_target)
        layout.addWidget(self.checker_classification)
        layout.addWidget(self.checker_regression)
        layout.addWidget(self.level_1_checker)
        layout.addWidget(self.level2_checker)
        layout.addWidget(self.level_3_checker)
        layout.addWidget(button_train)
        layout.addWidget(button_model_eval)
        layout.addWidget(self.label_target)
        layout.addWidget(self.table)
        
        self.setLayout(layout)
        self.model = None 
        self.file_target_path = None 
        self.target_name = None 
        self.encoder = None 
        self.model_mode = None 
        self.level = 1 
        self.autoprocessing = None 


    def submit_target(self) :
        if self.df is None :
            QErrorMessage(self).showMessage(f"Error the target file is {self.df}")
        else: 
            target_name = self.target_writter.text()
            self.label_target.setText(target_name)
            if target_name not in self.df.columns :
                QErrorMessage(self).showMessage(f"Error  : {target_name} not in collumns")
            else :
                self.target_name = target_name
        

    def file_finder (self) :
        file_target_path = self.file_searcher.getOpenFileName(self,"Search Data")
        file_target_path = file_target_path[0]
        self.file_target_path = file_target_path
        if isinstance(file_target_path,str):
            if file_target_path.endswith(".csv") :
                self.df = pd.read_csv(file_target_path)
                self.display_data(self.df)
            elif file_target_path.endswith(".xlsx") :
                self.df = pd.read_excel(file_target_path)
                self.display_data(self.df)
            else :
                QErrorMessage(self).showMessage(f"Error : {file_target_path} not supported \n , this tools just support for csv or excel format file")


    def auto_processing (self) :
        path_auto,_ = self.file_searcher.getOpenFileName(self,"searching autoprocessing file")
        if path_auto :
            if isinstance(path_auto,str) :
                if path_auto.endswith(".pkl") :
                    self.autoprocessing = joblib.load(path_auto)
                else :
                    QErrorMessage().showMessage(f"Error File {path_auto} not supported")     
            
    def display_data(self, df):
        self.table.setRowCount(df.shape[0])
        self.table.setColumnCount(df.shape[1])
        self.table.setHorizontalHeaderLabels(df.columns)


        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                self.table.setItem(row, col, QTableWidgetItem(str(df.iat[row, col])))

    def find_model(self) :
        model_path = self.file_searcher.getOpenFileName(self,"Search Model" )
        model_path = model_path[0]
        if isinstance(model_path,str) :
            if model_path.endswith(".pkl") :
                self.model = joblib.load(model_path)
            else :
                QErrorMessage(self).showMessage(f"Error {model_path} is not supported, \n model just support with pkl format model")
    
    def find_encoder (self) :
        encoder_path,_ = self.file_searcher.getOpenFileName(self,"search encoder ")
        if isinstance(encoder_path,str) :
            if encoder_path.endswith(".pkl") :
                self.encoder = joblib.load(encoder_path)
    def train_Model (self) :
        if self.file_target_path is None :
            QErrorMessage(self).showMessage("Error : File target not found") 
        if self.model is None :
            reply = QMessageBox.question(self,
                                "make new model ?",
                                f"you don't load model for training, \n System will auto make model \n do you want load model ?",
                                QMessageBox.Yes | QMessageBox.No,QMessageBox.No)
            if reply ==  QMessageBox.No :
                if self.checker_classification.isChecked():
                    self.model_mode = 'classification'  
                elif self.checker_regression.isChecked() :
                    self.model_mode = 'regression'  

                if self.level_1_checker.isChecked():
                    self.level = 1
                elif self.level2_checker.isChecked() :
                    self.level = 2 
                elif self.level_3_checker.isChecked():
                    self.level = 3 
                if self.autoprocessing is None :
                    reply_atp = QMessageBox.question(self,
                                "preprocessing ?",
                                f"make new Preprocessing ?",
                                QMessageBox.Yes | QMessageBox.No,QMessageBox.No)
                    if reply_atp == QMessageBox.Yes:
                        auto_processing = AutoPreprocessing(data=self.file_target_path,model_target=self.target_name)
                        path_auto_procces,_ = self.file_searcher.getSaveFileName(self,"save autopreprocessing","autoprocessing (*.pkl)")
                        if path_auto_procces:
                            try :
                                joblib.dump(auto_processing,path_auto_procces)
                            except :
                                QErrorMessage().showMessage("Error save Failed")
                                path_auto_procces,_ = self.file_searcher.getSaveFileName(self,"save autopreprocessing","autoprocessing (*.pkl)")
                                joblib.dump(auto_processing,path_auto_procces)
                            self.autoprocessing = auto_processing
                        else :
                            QErrorMessage(self).showMessage("error : file does't created")
                    elif reply_atp == QMessageBox.No :
                        self.auto_processing()
                if self.autoprocessing is None :
                    QErrorMessage(self).showMessage("Error : preprocessing tools not found")
                x_train,y_train = self.autoprocessing.process_data()
               
                if isinstance(y_train[0],str) :
                    self.encoder = LabelEncoder()
                    y_train = self.encoder.fit_encod(y_train)
                    reply_enc = QMessageBox.question(self,
                                "New Encoder ?",
                                f"make new Encoder ?",
                                QMessageBox.Yes | QMessageBox.No,QMessageBox.No)
                    if reply_enc == QMessageBox.Yes:
                        path_encoder,_=self.file_searcher.getSaveFileName(self,"save encoder","encoder (*.pkl)")
                        if path_encoder:
                            try:
                                joblib.dump(self.encoder,path_encoder)
                            except :
                                QErrorMessage(self).showMessage("error saving failed file must be file.pkl ")
                                path_encoder,_=self.file_searcher.getSaveFileName(self,"save encoder","encoder (*.pkl)")
                                joblib.dump(path_encoder)
                        else :
                            QErrorMessage(self).showMessage("error : file does't created")
                    elif reply_enc== QMessageBox.No :
                        self.find_encoder() 
                        try:
                            y_train = self.encoder.encod(y_train)
                        except:
                            QErrorMessage(self).showMessage("Warning : this encoder nothing training for this data \nbut will retraining to this data ")
                            y_train = self.encoder.fit_encod(y_train)
                    if self.encoder is None :
                        QErrorMessage(self).showMessage("Error : encoder not found ")
                    
                y_train = np.array(y_train )
                self.model = NeuralNetwork(n_target=np.max(y_train),variant=self.model_mode,complexity_level=self.level,Name=f"Neural Network {self.model_mode}",epochs=1000)
                self.model.fit(x_train.values,y_train)  
                QMessageBox.information(self,"Training Status ", "Model Training has successfully completed!")              
                path_model,_ = self.file_searcher.getSaveFileName(self,"save model","model(*.pkl)") 
                if path_model:
                    try :
                        joblib.dump(self.model,path_model)
                    except :
                        QErrorMessage(self).showMessage("error saving is failed file must be file.pkl ")
                        path_model,_ = self.file_searcher.getSaveFileName(self,"save Model","Model (*.pkl)")
                        joblib.dump(self.model,path_model)
                else :
                    QErrorMessage(self).showMessage("error : file does't created")

            elif QMessageBox.Yes :

                self.find_model()

                self.auto_processing()
                x_train,y_train = auto_processing.process_data()
                if isinstance(y_train[0],str) :
                    self.find_encoder()
                if isinstance(self.encoder,LabelEncoder) :
                    try:
                        y_train = self.encoder.encod(y_train)
                    except :
                        QErrorMessage(self).showMessage("Error : Encoder not support for this model")
                if isinstance(y_train,list) :
                    y_train = np.array(y_train)
                else :
                    QErrorMessage(self).showMessage("Error : maybe this file not supported")    
                if isinstance(self.model,NeuralNetwork):
                    try:
                        self.model.fit(x_train.values,y_train)
                        QMessageBox.information(self,"Training Status ", "tTraining has successfully completed!")
                    except :
                        QErrorMessage(self).showMessage("Error : Model not supported for this data")    
                else:
                    QErrorMessage(self).showMessage("Error : Maybe this file not supported")    



    def plot_model_trained (self) :
        if self.model is None :
            QErrorMessage(self).showMessage("Error this model has None")
        else :
            self.model.plot_eval()


class Model_Runners (QWidget):
    def __init__(self,Model : NeuralNetwork = None,encoder : LabelEncoder = None,auto_process : AutoPreprocessing = None  ) :
        super().__init__()
        self.model = Model 
        self.encoder = encoder 

        self.df = None 
        self.autoprocessing = auto_process   

        self.setWindowTitle("ModelInference")
        self.setGeometry(0,0,1380,720)
        layout = QVBoxLayout()
        self.table = QTableWidget()
        self.target_path_data = None 
        button_file_target = QPushButton("search target File")
        button_file_target.clicked.connect(self.finder_data)
        button_predict_data = QPushButton("Predict")
        button_predict_data.clicked.connect(self.predict)
        button_choice_component = QPushButton("choice (model,encoder,autoproces)")
        button_choice_component.clicked.connect(self.model_component)
        button_plot_model =QPushButton("Plot Model eval")
        button_plot_model.clicked.connect(self.plot_model_eval)
        layout.addWidget(button_file_target)
        layout.addWidget(button_choice_component)
        layout.addWidget(button_predict_data)
        layout.addWidget(button_plot_model)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def finder_data(self) :
        data_path,_ = QFileDialog().getOpenFileName(self,"data target searching")
        if data_path :
            self.target_path_data
            if isinstance(data_path,str) :
                if data_path.endswith(".csv") :
                    df = pd.read_csv(data_path)
                    self.df = df 
                    self.display_data(df)
                elif data_path.endswith(".xlsx"):
                    df = pd.read_excel(data_path)
                    self.df = df 
                    self.display_data(df)
                else :
                    QErrorMessage(self).showMessage("Error file not support")    

    def display_data(self, df):
        self.table.setRowCount(df.shape[0])
        self.table.setColumnCount(df.shape[1])
        self.table.setHorizontalHeaderLabels(df.columns)
        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                self.table.setItem(row, col, QTableWidgetItem(str(df.iat[row, col])))
    
    def model_component (self) :
        model_path,_ = QFileDialog().getOpenFileName(self,"search model")
        if isinstance(model_path,str) :
            if model_path.endswith(".pkl") :
                self.model = joblib.load(model_path) 
            else :
                QErrorMessage(self).showMessage(f"Error File {model_path} not supported") 

            if not isinstance(self.model,NeuralNetwork) :
                QErrorMessage(self).showMessage("Error file model not supported")

        if isinstance(self.model,NeuralNetwork) :
            self.last_activation = self.model.model.component[-1]
        else :
            QErrorMessage(self).showMessage("error Model not Supported")

        encoder_path,_ = QFileDialog().getOpenFileName(self,"search encoder")
        if isinstance(encoder_path,str) :
            if encoder_path.endswith(".pkl") :
                self.encoder = joblib.load(encoder_path) 
            else :
                QErrorMessage(self).showMessage(f"Error File {encoder_path} not supported") 

            if not isinstance(self.encoder,LabelEncoder) :
                QErrorMessage(self).showMessage("Error file model not supported")
        auto_procces,_ = QFileDialog().getOpenFileName(self,"search autopreprocessing")
        if isinstance(auto_procces,str) :
            if model_path.endswith(".pkl") :
                self.autoprocessing= joblib.load(auto_procces) 
            else :
                QErrorMessage(self).showMessage(f"Error File {auto_procces} not supported") 

            if not isinstance(self.autoprocessing,AutoPreprocessing) :
                QErrorMessage(self).showMessage("Error file model not supported")

    def predict(self) :
        if self.model is None :
            model_path,_ = QFileDialog().getOpenFileName(self,"Model Searching")
            if model_path :
                try:
                    self.model = joblib.load(model_path)
                except :
                   QErrorMessage(self).showMessage(f"Error file model {model_path} is not support")
                   model_path,_ = QFileDialog().getOpenFileName(self,"Model Searching")
                   self.model = joblib.load(model_path)
        if not isinstance(self.model,NeuralNetwork) :
            QErrorMessage(self).showMessage("Error Model not supported") 
            model_path,_ = QFileDialog().getOpenFileName(self,"Model Searching")
            self.model = joblib.load(model_path)

        if isinstance(self.model,NeuralNetwork) :
            self.last_activation = self.model.model.component[-1]
        else :
            QErrorMessage(self).showMessage("error Model not Supported")

        if self.autoprocessing is None :
            path_auto,_ = QFileDialog().getSaveFileName(self,"searching autopreprocessing")
            try : 
                self.autoprocessing = joblib.load(path_auto)
            except :
                QErrorMessage(self).showMessage("Error file not supported")
                path_auto,_ = QFileDialog().getSaveFileName(self,"searching autopreprocessing") 
                self.autoprocessing = joblib.load(path_auto)

        x_train,y_train = self.autoprocessing.process_data() 
        is_encoding_data = False 
        if isinstance(y_train[0],str):
            is_encoding_data = True 
            if self.encoder is None :
                encoder_path,_ = QFileDialog().getOpenFileName(self,"encoder Searching")
                if encoder_path :
                    try :
                        self.encoder = joblib.load(encoder_path)
                    except :
                        QErrorMessage(self).showMessage(f"Error file encoder {encoder_path} is not support")
                        encoder_path,_ = QFileDialog().getOpenFileName(self,"encoder Searching")
                        self.model = joblib.load(encoder_path)     
                if not isinstance(self.encoder,LabelEncoder) :
                    QErrorMessage(self).showMessage("Error Encoder not supported")      
                    encoder_path,_ = QFileDialog().getOpenFileName(self,"encoder Searching")
                    self.encoder= joblib.load(encoder_path)
            y_train = self.encoder.encod(y_train)
        
        result = self.model.predict(x_train)
        if isinstance(self.last_activation,nn.Sigmoid) :
            result = np.where(result > 0.5,1,0)
        elif isinstance(self.last_activation,nn.Softmax) :
            result = np.argmax(result,axis=-1)

        if is_encoding_data is True :
            result = self.encoder.decod(result)

        self.df['predicted'] = result
        self.display_data(self.df)    

    def plot_model_eval(self) :
        self.model.plot_eval()    
    


class Main (QWidget) :
    def __init__(self) :
        super().__init__()
        self.setWindowTitle("Micronet(Micro neural network)")
        self.setGeometry(650,150,200,200)
        self.trainer = Model_Trainers()
        self.model_runner = Model_Runners(Model=self.trainer.model,
                                          encoder=self.trainer.encoder,
                                          auto_process=self.trainer.autoprocessing)
        button_runers = QPushButton("Prediction")
        button_trainers = QPushButton("Training")
        button_runers.clicked.connect(self.model_run)
        button_trainers.clicked.connect(self.model_trainer)

        layout = QVBoxLayout()
        layout.addWidget(button_runers)
        layout.addWidget(button_trainers)
        self.setLayout(layout)
    
    def model_run(self) :
        self.model_runner.show()
    
    def model_trainer (self) :
        self.trainer.show()
        self.model_runner.model = self.trainer.model 
        self.model_runner.autoprocessing = self.trainer.autoprocessing
        self.model_runner.encoder = self.trainer.encoder

if __name__ == "__main__" :
    app_exiter = QApplication(sys.argv)
    maingui = Main()
    maingui.show()
    sys.exit(app_exiter.exec_())