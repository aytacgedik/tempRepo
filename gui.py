import tkinter as tk                
from tkinter import font as tkfont
from tkcalendar import Calendar, DateEntry
import matplotlib.pyplot as plt
import datetime as dt
import ai
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from multiprocessing import Process
import multiprocessing as mp
import math
from sklearn.metrics import mean_squared_error
class PredictionApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")
        self.title("Stock Trend Prediction")
        self.minsize(800,400)
        
        self.canvasFrame = tk.Frame(self,height=300,width=600,bg="white")
        self.canvasFrame.grid(column = 0, row = 0)
        self.frame = tk.Frame(self)
        self.frame.grid(column = 0, row = 1)
        button = tk.Button(self,text="Build Model",command=self.BuildStockPrediction)
        button.grid(column=0,row=2)
        buttonPlot = tk.Button(self,text="Plot Data",command=self.PlotData)
        buttonPlot.grid(column=0,row=3)
        buttonSave = tk.Button(self,text="Save as image",command=self.SaveImage)
        buttonSave.grid(column=0,row=4)
        
        
        self.dataSource = tk.StringVar()
        self.AddLabelAndTextBox(0,0,"Data source : ",'yahoo',self.dataSource)
        self.tickerSymbol = tk.StringVar()
        self.AddLabelAndTextBox(0,1,"Company Ticker Symbol: ",'AAPL',self.tickerSymbol)
        self.column = tk.StringVar()
        self.AddLabelAndTextBox(0,2,"Data column : ",'Close',self.column)
        
        self.nPredictDays=tk.IntVar()
        self.AddLabelAndTextBox(2,0,"Number of days to look back to predict a day: ",60,self.nPredictDays)

        self.units=tk.IntVar()
        self.AddLabelAndTextBox(2,1,"Units: ",50,self.units)
        self.dropout=tk.DoubleVar()
        self.AddLabelAndTextBox(4,1,"Drop out: ",0.2,self.dropout)

        self.nEpochs=tk.IntVar()
        self.AddLabelAndTextBox(2,2,"Epochs: ",25,self.nEpochs)
        self.batch_size=tk.IntVar()
        self.AddLabelAndTextBox(4,2,"Batch size: ",32,self.batch_size)

        self.optimizer = tk.StringVar()
        self.AddLabelAndTextBox(2,3,"Optimizer : ",'adam',self.optimizer)
        self.loss = tk.StringVar()
        self.AddLabelAndTextBox(4,3,"Loss method : ",'mean_squared_error',self.loss,_width=20)

        self.trainStartDate = DateEntry(self.frame,width=12, background='darkblue',foreground='white', borderwidth=2)
        self.AddLabelAndDateBox(2,4,"Train start date: ",dt.datetime(2012,1,1),self.trainStartDate)

        self.trainEndDate = DateEntry(self.frame,width=12, background='darkblue',foreground='white', borderwidth=2)
        
        self.AddLabelAndDateBox(4,4,"Train end date: ",dt.datetime(2020,1,1),self.trainEndDate)

        self.testStartDate = DateEntry(self.frame,width=12, background='darkblue',foreground='white', borderwidth=2)
        self.AddLabelAndDateBox(2,5,"Test start date: ",dt.datetime(2020,1,1),self.testStartDate)

        self.testEndDate = DateEntry(self.frame,width=12, background='darkblue',foreground='white', borderwidth=2)
        self.AddLabelAndDateBox(4,5,"Test end date: ",dt.datetime.now(),self.testEndDate)
        
    def SaveImage(self):
        a = tk.filedialog.asksaveasfilename(filetypes=(("PNG Image", "*.png"),("All Files", "*.*")), 
            defaultextension='.png', title="Window-2")
        if a:
            plt.savefig(a)

    def BuildStockPrediction(self):
        self.stockPrediction = ai.StockPrediction(self.tickerSymbol.get(),self.nPredictDays.get(), self.trainStartDate.get_date(),self.trainEndDate.get_date(),self.dataSource.get())
        self.stockPrediction.PrepareData()
        modelInputs={ 'units': [self.units.get(), self.units.get(), self.units.get(), 1],'return_sequences': [True, True, False],'dropout': [self.dropout.get(), self.dropout.get(), self.dropout.get()]}
        self.stockPrediction.BuildModel(self.optimizer.get(),self.loss.get(),self.nEpochs.get(),self.batch_size.get(),modelInputs)    
        
    def PlotData(self):
        testdata = self.stockPrediction.TestAccurancyAndPlot(self.testStartDate.get_date(),self.testEndDate.get_date())
        lstmTrainScore = math.sqrt(mean_squared_error(testdata['trained_prices'][:,0], testdata['predicted_trained_prices'][:,0]))
        lstmTestScore = math.sqrt(mean_squared_error(testdata['actual_prices'], testdata['predicted_prices'][0:len(testdata['predicted_prices'])-1,0]))
        naiveScore = math.sqrt(mean_squared_error(testdata['actual_prices'], testdata['naiveMethod']))
        # arimaScore = math.sqrt(mean_squared_error(testdata['actual_prices'], testdata['arimaMethod']))

        print(f'Lstm Train Score: {lstmTrainScore} RMSE')
        print(f'Lstm Test Score: {lstmTestScore} RMSE')
        print(f'Naive Score: {naiveScore} RMSE')
        # print(f'ARIMA Score: {arimaScore} RMSE')
        fig = plt.figure()

        print(type(testdata['trained_prices'] ))
        plt.axvline(x =len(testdata['trained_prices'])+1, ymin =0, label='Start of test data',color ="blue")
        plt.plot(list(testdata['trained_prices']) + list(testdata['actual_prices']), color="red", label=f"Actual {self.tickerSymbol.get()} Price")
        x_range = range(len(testdata['trained_prices'])+1,len(testdata['trained_prices'])+len(testdata['predicted_prices'])+1)
        x_range2 = range(len(testdata['trained_prices'])+1,len(testdata['trained_prices'])+len(testdata['predicted_prices']))
        plt.plot(x_range,testdata['predicted_prices'], color="green", label=f"Lstm Predicted {self.tickerSymbol.get()} Price")
        plt.plot(x_range2,testdata['naiveMethod'], color="orange", label=f"Naive Predicted {self.tickerSymbol.get()} Price")
        # plt.plot(x_range2,testdata['arimaMethod'], color="brown", label=f"ARIMA Predicted {self.tickerSymbol.get()} Price")

        plt.title(f"{self.tickerSymbol.get()} Share prices")
        plt.xlabel('Time(Days)')
        plt.ylabel(f'{self.tickerSymbol.get()} Share Price')
        plt.legend()
        self._clear()
        canvas = FigureCanvasTkAgg(fig, master=self.canvasFrame)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        plt.show()
        

    def AddLabelAndTextBox(self,_column,_row,labelText,tickerSymbolText,prop,_width=15,_validate=None):
        label = tk.Label(self.frame, text =labelText )
        label.grid(column = _column, row = _row)
        prop.set(tickerSymbolText)
        tickerSymbolEntered = tk.Entry(self.frame, width = _width, textvariable = prop)
        tickerSymbolEntered.grid(column = _column+1, row = _row)
    
    def AddLabelAndDateBox(self,_column,_row,labelText,date,cal):
        label = tk.Label(self.frame, text =labelText )
        label.grid(column = _column, row = _row)
        cal.set_date(date)
        cal.grid(column = _column+1, row = _row)
    

    def _clear(self):
        for widget in self.canvasFrame.winfo_children():
            widget.destroy()


if __name__ == "__main__":
    app = PredictionApp()
    app.mainloop()