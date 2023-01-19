from tkinter import*
import csv
from tkinter import messagebox
import joblib
import numpy as np
from PIL import ImageTk, Image  
import pathlib

global result 
def show_entry_fields():
    global result
    p1=float(e1.get())
    p2=float(e2.get())
    p3=float(e3.get())
    p4=float(e4.get())
    p5=float(e5.get())
    p6=float(e6.get())
    p7=float(e7.get())
    p8=float(e8.get())
    
    model = joblib.load('model_joblib_diabetes')
    result=model.predict([[p1,p2,p3,p4,p5,p6,p7,p8]])
    
    if result == 0:
        Label(win, text="Non-Diabetic",fg='green',font=("Helvetica", 40),width=20).place(x=500, y=600)
    else:
        Label(win, text="Diabetic",fg='red',font=("Helvetica", 40),width=20).place(x=500, y=600)
        
        
win = Tk()
win.geometry('1400x700')
win.config(bg='#D9D8D7')
win['bg'] ='black'
win.title('Diabetes Prediction Using Machine Learning')
load=Image.open("img1.png")
render = ImageTk.PhotoImage(load)
img = Label(win,image=render)
img.image = render
img.place(x=1100,y=230)

main_lst = []

lbl=Label(win, text="Diabetes Prediction Using Machine Learning", fg='white',bg='teal',width=90, font=("Helvetica", 25))
lbl.place(x=0, y=0)

Label(win, text="Pregnancies", fg='white',bg='BlueViolet',  font=("Helvetica", 16),width=38).place(x=300, y=100)
Label(win, text="Glucose", fg='white',bg='BlueViolet', font=("Helvetica", 16),width=38).place(x=300, y=150)
Label(win, text="Enter Value of Blood Pressure",fg='white',bg='BlueViolet',font=("Helvetica",16),width=38).place(x=300,y=200)
Label(win, text="Enter Value of Skin Thickness", fg='white',bg='BlueViolet', font=("Helvetica", 16),width=38).place(x=300, y=250)
Label(win, text="Enter Value of Insulin", fg='white',bg='BlueViolet', font=("Helvetica", 16),width=38).place(x=300, y=300)
Label(win, text="Enter Value of BMI", fg='white',bg='BlueViolet', font=("Helvetica", 16),width=38).place(x=300, y=350)
Label(win, text="Enter Value of Diabetes Pedigree Function", fg='white',bg='BlueViolet', font=("Helvetica", 16),width=38).place(x=300, y=400)
Label(win, text="Enter Value of Age", fg='white', bg='BlueViolet',font=("Helvetica", 16),width=38).place(x=300, y=450)

e1 = Entry(win,bd = 3,width=30)
e2 = Entry(win,bd = 3,width=30)
e3 = Entry(win,bd = 3,width=30)
e4 = Entry(win,bd = 3,width=30)
e5 = Entry(win,bd = 3,width=30)
e6 = Entry(win,bd = 3,width=30)
e7 = Entry(win,bd = 3,width=30)
e8 = Entry(win,bd = 3,width=30)

e1.place(x=800, y=100)
e2.place(x=800, y=150)
e3.place(x=800, y=200)
e4.place(x=800, y=250)
e5.place(x=800, y=300)
e6.place(x=800, y=350)
e7.place(x=800, y=400)
e8.place(x=800, y=450)

Button(win, text="Predict", command=show_entry_fields,padx=20,pady=10, fg='blue',font=26).place(x=400, y=500)

def Add():
    global result
    lst  = [e1.get(),e2.get(),e3.get(),e4.get(),e5.get(),e6.get(),e7.get(),e8.get(),result[0]]
    main_lst.append(lst)
    messagebox.showinfo("Information","The data has been added successfully")

def Save():
    with open("data_entry.csv","a") as file:
        Writer = csv.writer(file)
        Writer.writerows(main_lst)
        messagebox.showinfo("Information","Saved succesfully")

save=Button(win,text="Save",padx=20,pady=10,command=Save,font=26).place(x=700,y=500)
add=Button(win,text="Add",padx=20,pady=10,command=Add,font=26).place(x=600,y=500)
Exit=Button(win,text="Exit",padx=20,pady=10,command=win.quit,font=26).place(x=800,y=500)


   

win.mainloop()