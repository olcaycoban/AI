from tkinter import *
from tkinter.filedialog import askopenfilename
import csv
import os
from tkinter import ttk
import time

gui = Tk()

gui.title('House Prediction GUI')

gui.geometry('600x600+400+50')

progress_bar = ttk.Progressbar(orient = 'horizontal', length=600, mode='determinate')
progress_bar.grid(row=150, columnspan=3, pady =10)

def data():
    global filename
    filename = askopenfilename(initialdir='C:\\',title = "Select file")
    e1.delete(0, END)
    e1.insert(0, filename)

    import pandas as pd
    global file1

    file1 = pd.read_csv(filename)

    global col
    col = list(file1.head(0))

    for i in range(len(col)):
        box1.insert(i+1, col[i])

def X_values():
    values = [box1.get(idx) for idx in box1.curselection()]

    for i in range(len(list(values))):
        box2.insert(i+1, values[i])
        box1.selection_clear(i+1, END)
    X_values.x1=[]
    for j in range(len(values)):
        X_values.x1.append(j)

    global x_size
    x_size = len(X_values.x1)
    print(x_size)
    print(X_values.x1)

def y_values():
    values= [box1.get(idx) for idx in box1.curselection()]
    for i in range(len(list(values))):
        box3.insert(i+1, values[i])
        box1.selection_clear(i + 1, END)
    y_values.y1=[]
    for j in range(len(values)):
        y_values.y1.append(j)

    print(y_values.y1)

l1=Label(gui, text='Select Data File')
l1.grid(row=0, column=0)
e1 = Entry(gui,text='')
e1.grid(row=0, column=1)

Button(gui,text='open',command=data).grid(row=0, column=2)

box1 = Listbox(gui,selectmode='multiple')
box1.grid(row=10, column=0)
Button(gui, text='Clear All').grid(row=12,column=0)

box2 = Listbox(gui)
box2.grid(row=10, column=1)
Button(gui, text='Select X',command=X_values).grid(row=12,column=1)

box3 = Listbox(gui)
box3.grid(row=10, column=2)
Button(gui, text='Select y').grid(row=12,column=2)

Button(gui, text='Solution').grid(row=20, column=1)

gui.mainloop()