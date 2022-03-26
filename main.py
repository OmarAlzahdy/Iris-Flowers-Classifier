import random
from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import metrics
import seaborn as sns

def signom(x):
    if x>0 :
        return 1
    elif x==0:
        return 0
    else:
        return -1

data = pd.read_csv("IrisData.csv")
X = data.drop("Class", axis=1)
Y = data["Class"]
Y2=Y
le = preprocessing.LabelEncoder()
Y = le.fit_transform(Y)
fig_plot, ax = plt.subplots(3, 2)
ax[0, 0].scatter(X["X1"], X["X2"], c=Y)
ax[0, 0].set_title("X1 & X2")
ax[0, 0].set_xlabel("X1")
ax[0, 0].set_ylabel("X2")


ax[0, 1].scatter(X["X1"], X["X3"], c=Y)
ax[0, 1].set_title("X1 & X3")
ax[0, 1].set_xlabel("X1")
ax[0, 1].set_ylabel("X3")

ax[1, 0].scatter(X["X1"], X["X4"], c=Y)
ax[1, 0].set_title("X1 & X4")
ax[1, 0].set_xlabel("X1")
ax[1, 0].set_ylabel("X4")

ax[1, 1].scatter(X["X2"], X["X3"], c=Y)
ax[1, 1].set_title("X2 & X3")
ax[1, 1].set_xlabel("X2")
ax[1, 1].set_ylabel("X3")

ax[2, 0].scatter(X["X2"], X["X4"], c=Y)
ax[2, 0].set_title("X2 & X4")
ax[2, 0].set_xlabel("X2")
ax[2, 0].set_ylabel("X4")

ax[2, 1].scatter(X["X3"], X["X4"], c=Y)
ax[2, 1].set_title("X3 & X4")
ax[2, 1].set_xlabel("X3")
ax[2, 1].set_ylabel("X4")


fig_plot.tight_layout()

plt.show()

def evaluate_model(Y_test,y_mat):
    cnf_matrix = metrics.confusion_matrix(Y_test, y_mat)
    cnf_matrix
    class_names = [0, 1]  # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
    print("Accuracy:", metrics.accuracy_score(Y_test, y_mat))
    print("Precision:", metrics.precision_score(Y_test, y_mat,zero_division=0))
    print("Recall:", metrics.recall_score(Y_test, y_mat))


def Run_Model():
    x=pd.DataFrame()
    y=pd.DataFrame()

    if(i1.get()):
        x=pd.concat([x,X.iloc[0:50,[f1.get(),f2.get()]]],ignore_index=True)
        y = pd.concat([y, Y2[0:50]], ignore_index=True)

    if(i2.get()):
        x=pd.concat([x,X.iloc[50:100,[f1.get(),f2.get()]]],ignore_index=True)
        y = pd.concat([y, Y2[50:100]], ignore_index=True)

    if(i3.get()):
        x=pd.concat([x,X.iloc[100:150,[f1.get(),f2.get()]]],ignore_index=True)
        y = pd.concat([y, Y2[100:150]], ignore_index=True)

    y=pd.DataFrame(le.fit_transform(y.values.ravel()))

    epoches=int(epoches_box.get())



    X_train1, X_test1, y_train1, y_test1 = train_test_split(x[0:50], y[0:50], test_size=0.4,random_state=1)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(x[50:100], y[50:100], test_size=0.4,random_state=1)
    X_train=pd.concat([X_train1,X_train2], ignore_index=True)
    X_test=pd.concat([X_test1,X_test2], ignore_index=True)
    X_train=X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    Y_train=pd.concat([y_train1,y_train2], ignore_index=True)
    Y_test=pd.concat([y_test1,y_test2], ignore_index=True)
    Y_train= Y_train.reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)

    data_temp=X_train.copy()
    data_temp['Class']=Y_train[0].values.tolist()
    data_temp = data_temp.sample(frac=1,random_state=5).reset_index(drop=True)

    X_train = data_temp.drop("Class", axis=1)
    Y_train = data_temp["Class"]




    X_train=np.array(X_train)
    X_test=np.array(X_test)

    Y_train=np.array(Y_train)
    Y_test=np.array(Y_test)
    Y_train=Y_train*2-1
    Y_test=Y_test*2-1

    if(bias_var.get()):
        np.random.seed(1034)
        theta = np.random.random(3)
        X_train=np.insert(X_train,0,1,1)
        X_test = np.insert(X_test, 0, 1, 1)
    else:
        np.random.seed(1034)
        theta = np.random.random(2)

    y_mat=[]
    errors_list=[]
    min_error=float('inf')
    for i in range(epoches):
        temp_error = 0
        for j in range(60):
            y_pred=signom(np.dot(theta.T,X_train[j]))
            if y_pred!=Y_train[j]:
                loss=Y_train[j]-y_pred
                theta=theta+float(eta_box.get())*loss*X_train[j]
                #print shapes
                temp_error +=loss**2

        errors_list.append(temp_error)

    total_error=0.0
    for i in range(40):
        y_pred = signom(np.dot(theta.T, X_test[i]))
        y_mat.append(y_pred)
        loss = Y_test[i] - y_pred
        total_error+=loss**2
    print(total_error)

    plt.plot(range(len(errors_list)),errors_list)

    evaluate_model(Y_test,y_mat)











frame = Tk()
frame.title("Marvel NN")
frame.geometry('500x500')

f1 = IntVar()

features1 = {"X1": "0",
             "X2": "1",
             "X3": "2",
             "X4": "3"
             }

j=0
for (text, value) in features1.items():
    i = Radiobutton(frame, text=text, variable=f1, value=value)
    i.place(x=50, y=50 + j * 30)
    j+=1

f2 = IntVar()

features2 = {"X1": "0",
             "X2": "1",
             "X3": "2",
             "X4": "3"
             }
j=0
for (text, value) in features2.items():
    i = Radiobutton(frame, text=text, variable=f2, value=value)
    i.place(x=420, y=50 + j * 30)
    j += 1


def click_me():
    string_to_show = ""
    eta_input = eta_box.get()
    Epoches_input = epoches_box.get()
    try:
        float(eta_input)
        float(Epoches_input)

    except ValueError:
        string_to_show = "Incorrect input for eta or number of epoches. \n"



    sum1 = i1.get() + i2.get() + i3.get()

    if sum1 != 2:
        string_to_show = string_to_show + "you must pick 2 types. \n"

    if f1.get() == f2.get():
        string_to_show = string_to_show + "you must pick 2 different features."

    if string_to_show == "":
        Run_Model()
    else:
        messagebox.showinfo("alert", string_to_show)

bias_var = IntVar()
biasbutton = Checkbutton(frame, text="Bias", variable=bias_var)
biasbutton.place(x=235,y=290)


i1 = IntVar()
i2 = IntVar()
i3 = IntVar()

type1 = Checkbutton(frame, text="Setosa", variable=i1)
type1.place(x=235,y=60)
#type1.pack()

type2 = Checkbutton(frame, text="Versicolor", variable=i2)
type2.place(x=235,y=100)
#type2.pack()

type3 = Checkbutton(frame, text="Virginica", variable=i3)
type3.place(x=235,y=140)
#type3.pack()

labelText = StringVar()
labelText.set("Enter Learning Rate :")
labelDir = Label(frame, textvariable=labelText)
labelDir.place(x=50,y=200)

eta_box = Entry(frame)
eta_box.focus_set()
eta_box.place(x=195,y=200)




labelText2 = StringVar()
labelText2.set("Enter epoches' number :")
labelDir2 = Label(frame, textvariable=labelText2)
labelDir2.place(x=50,y=250)

epoches_box = Entry(frame)
epoches_box.focus_set()
epoches_box.place(x=215,y=250)



Run_button = Button(frame, text="RUN", command=click_me,width=20)
Run_button.place(x=190,y=390)

frame.mainloop()
