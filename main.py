import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn import linear_model
from prettytable import PrettyTable
import time
#------------------------------------------------------------------------------
def trajectory():  
    birddata = pd.read_csv("bird_tracking.csv")
    print(" Latitude and Longitude of the bird Eric")
    bird_names = pd.unique(birddata.bird_name)  
    ix = birddata.bird_name == "Eric" 
    x,y = birddata.longitude[ix], birddata.latitude[ix] 
    plt.figure(figsize = (7,7)) 
    plt.plot(x,y,"b.") 
    print(" Latitude and Longitude of the bird Nico ") 
    bird_names = pd.unique(birddata.bird_name)  
    ix = birddata.bird_name == "Nico" 
    x,y = birddata.longitude[ix], birddata.latitude[ix] 
    plt.figure(figsize = (7,7)) 
    plt.plot(x,y,"b.") 
    print(" Latitude and Longitude of the bird Sanne")
    bird_names = pd.unique(birddata.bird_name)  
    ix = birddata.bird_name == "Sanne" 
    x,y = birddata.longitude[ix], birddata.latitude[ix] 
    plt.figure(figsize = (7,7)) 
    plt.plot(x,y,"b.") 
    plt.figure(figsize = (7,7)) 
    for bird_name in bird_names: 
        ix = birddata.bird_name == bird_name   
        x,y = birddata.longitude[ix], birddata.latitude[ix] 
        plt.plot(x,y,".", label=bird_name) 
    plt.xlabel("Longitude") 
    plt.ylabel("Latitude") 
    plt.legend(loc="lower right") 
    plt.show() 

def speed_frequency():
    birddata = pd.read_csv("bird_tracking.csv")
    print("  Comparision of frequency and speed for Eric")
    bird_names = pd.unique(birddata.bird_name)  
    ix = birddata.bird_name == "Eric" 
    speed = birddata.speed_2d[ix]  
    plt.figure(figsize = (8,4)) 
    ind = np.isnan(speed) 
    plt.hist(speed[~ind], bins = np.linspace(0,30,20), normed=True) 
    plt.xlabel(" 2D speed (m/s) ") 
    plt.ylabel(" Frequency ") 
    plt.show() 
    print("  Comparision of frequency and speed for Nico")
    birddata = pd.read_csv("bird_tracking.csv") 
    bird_names = pd.unique(birddata.bird_name)  
    ix = birddata.bird_name == "Nico" 
    speed = birddata.speed_2d[ix]  
    plt.figure(figsize = (8,4)) 
    ind = np.isnan(speed) 
    plt.hist(speed[~ind], bins = np.linspace(0,30,20), normed=True) 
    plt.xlabel(" 2D speed (m/s) ") 
    plt.ylabel(" Frequency ") 
    plt.show() 
    print("  Comparision of frequency and speed for Sanne")
    birddata = pd.read_csv("bird_tracking.csv") 
    bird_names = pd.unique(birddata.bird_name)  
    ix = birddata.bird_name == "Sanne" 
    speed = birddata.speed_2d[ix]  
    plt.figure(figsize = (8,4)) 
    ind = np.isnan(speed)   
    plt.hist(speed[~ind], bins = np.linspace(0,30,20), normed=True) 
    plt.xlabel(" 2D speed (m/s) ") 
    plt.ylabel(" Frequency ") 
    plt.show()

def elapsed_time():
    print(" Elapsed time and date for Eric")
    birddata = pd.read_csv("bird_tracking.csv") 
    bird_names = pd.unique(birddata.bird_name)  
    timestamps = [] 
    for k in range(len(birddata)): 
        timestamps.append(datetime.datetime.strptime(birddata.date_time.iloc[k][:-3], "%Y-%m-%d %H:%M:%S")) 
    birddata["timestamp"] = pd.Series(timestamps, index = birddata.index) 
    times = birddata.timestamp[birddata.bird_name == "Eric"] 
    elapsed_time = [time-times[0] for time in times]    
    plt.plot(np.array(elapsed_time)/datetime.timedelta(days=1)) 
    plt.xlabel(" Observation ") 
    plt.ylabel(" Elapsed time (days) ") 
    plt.show() 
    timestamps = []
    print(" Elapsed time and date for Nico") 
    for k in range(len(birddata)): 
        timestamps.append(datetime.datetime.strptime(birddata.date_time.iloc[k][:-3], "%Y-%m-%d %H:%M:%S")) 
    birddata["timestamp"] = pd.Series(timestamps, index = birddata.index) 
    times = birddata.timestamp[birddata.bird_name == "Nico"] 
    elapsed_time = [time-times[19795] for time in times] 
    plt.plot(np.array(elapsed_time)/datetime.timedelta(days=1)) 
    plt.xlabel(" Observation ")     
    plt.ylabel(" Elapsed time (days) ") 
    plt.show() 
    print(" Elapsed time and date for Sanne")
    timestamps = [] 
    for k in range(len(birddata)): 
        timestamps.append(datetime.datetime.strptime(birddata.date_time.iloc[k][:-3], "%Y-%m-%d %H:%M:%S")) 
    birddata["timestamp"] = pd.Series(timestamps, index = birddata.index) 
    times = birddata.timestamp[birddata.bird_name == "Sanne"] 
    elapsed_time = [time-times[40916] for time in times] 
    plt.plot(np.array(elapsed_time)/datetime.timedelta(days=1)) 
    plt.xlabel(" Observation ") 
    plt.ylabel(" Elapsed time (days) ") 
    plt.show()
    
def mean_speed():
    birddata = pd.read_csv("bird_tracking.csv") 
    bird_names = pd.unique(birddata.bird_name)  
    timestamps = [] 
    for k in range(len(birddata)): 
        timestamps.append(datetime.datetime.strptime(birddata.date_time.iloc[k][:-3], "%Y-%m-%d %H:%M:%S")) 
    birddata["timestamp"] = pd.Series(timestamps, index = birddata.index) 
    data =  birddata[birddata.bird_name == "Eric"] 
    times = data.timestamp 
    elapsed_time = [time-times[0] for time in times] 
    elapsed_days = np.array(elapsed_time)/datetime.timedelta(days=1) 
    next_day = 1
    inds = [] 
    daily_mean_speed = [] 
    for (i,t) in enumerate(elapsed_days): 
        if t < next_day: 
            inds.append(i) 
        else: 
            daily_mean_speed.append(np.mean(data.speed_2d[inds])) 
            next_day += 1
            inds = [] 
    plt.figure(figsize = (8,6)) 
    plt.plot(daily_mean_speed, "rs-") 
    plt.xlabel(" Day ") 
    plt.ylabel(" Mean Speed (m/s) "); 
    plt.show()
#------------------------------------------------------------------------------
x=PrettyTable()
class Node:
    
    def __init__(self,speed,mon,lat,long,alt):
        self.left = None
        self.right = None
        self.alt = alt
        self.mon = mon     
        self.lat = lat
        self.long = long
        self.speed = speed
    def insert(self,speed,mon,lat,long,alt):
        if self.mon:
            if mon < self.mon:
                if self.left is None:
                    self.left = Node(speed,mon,lat,long,alt)
                else:
                    self.left.insert(speed,mon,lat,long,alt)
            elif mon > self.speed:
                if self.right is None:
                    self.right = Node(speed,mon,lat,long,alt)
                else:
                    self.right.insert(speed,mon,lat,long,alt)
        else:
            self.mon = mon
    def PrintTree(self):
        x.field_names = ['Speed','Month','Latitude','Longitude','Altitude']
        if self.left:
            self.left.PrintTree()
        x.add_row([self.speed,self.mon,self.lat,self.long,self.alt]),
        if self.right:
            self.right.PrintTree()
        return x

i=0
alt=[]
mon=[]
lat=[]
long=[]
speed=[]

def eric_df():
    eric=pd.read_csv("consolidated.csv")
    n=len(eric)
    speed=eric['avg_eric']
    mon=eric['month']
    lat=eric['eric_lat']
    long=eric['eric_long']
    alt=eric['eric_alti']
    root=Node(speed[1],mon[1],lat[1],long[1],alt[1])
    #print(len(a))
    for i in range(0,n-1):
        root.insert(speed[i],mon[i],lat[i],long[i],alt[i])
    def showdata():
        print(root.PrintTree())
    showdata()

def nico_df():
    nico=pd.read_csv("cons_nico.csv")
    n=len(nico)
    speed=nico['avg_nico']
    mon=nico['month']
    lat=nico['nico_lat']
    long=nico['nico_long']
    alt=nico['nico_alti']
    root=Node(speed[1],mon[1],lat[1],long[1],alt[1])
    #print(len(a))
    for i in range(0,n-1):
        root.insert(speed[i],mon[i],lat[i],long[i],alt[i])
    def showdata():
        print(root.PrintTree())
    showdata()

def sanne_df():
    sanne=pd.read_csv("cons_sanne.csv")
    n=len(sanne)
    speed=sanne['avg_sanne']
    mon=sanne['month']
    lat=sanne['sanne_lat']
    long=sanne['sanne_long']
    alt=sanne['sanne_alti'] 
    root=Node(speed[1],mon[1],lat[1],long[1],alt[1])
    #print(len(a))
    for i in range(0,n-1):
        root.insert(speed[i],mon[i],lat[i],long[i],alt[i])
    def showdata():
        print(root.PrintTree())
    showdata()
#------------------------------------------------------------------------------    
def bubbleSort_eric(arr,arr1):
    a=time.time()
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1] :
                arr[j], arr[j+1] = arr[j+1], arr[j]
                arr1[j], arr1[j+1] = arr1[j+1], arr1[j]
    print ("Sorted array is:")
    for i in range(len(arr)):
        print ("%f " %arr[i]," \t %d" %arr1[i])
    b=time.time()
    c=b-a
    print("Execution time of bubble sort is: %f" %c)
data=pd.read_csv("consolidated.csv")
arre =data['avg_eric']
monthe = data['month']

def compe():
    a=time.time()
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1] :
                arr[j], arr[j+1] = arr[j+1], arr[j]
                arr1[j], arr1[j+1] = arr1[j+1], arr1[j]
    print ("Sorted array is:")
    for i in range(len(arr)):
        print ("%f " %arr[i]," \t %d" %arr1[i])
    b=time.time()
    c=b-a
    a1=time.time()
    for i in range(1, len(arr)): 
        key = arr[i]
        key1 = arr1[i]
        j = i-1
        while j >=0 and key < arr[j] : 
                arr[j+1] = arr[j]
                arr1[j+1]=arr1[j]
                j -= 1
        arr[j+1] = key
        arr1[j+1]= key1
    print ("Sorted array is:") 
    for i in range(len(arr)): 
        print ("%f" %arr[i],"\t %d" %arr1[i])
    b1=time.time()
    c1=b1-a1
    
def insertionSort_eric(arr,arr1):
    a=time.time()
    for i in range(1, len(arr)): 
        key = arr[i]
        key1 = arr1[i]
        j = i-1
        while j >=0 and key < arr[j] : 
                arr[j+1] = arr[j]
                arr1[j+1]=arr1[j]
                j -= 1
        arr[j+1] = key
        arr1[j+1]= key1
    print ("Sorted array is:") 
    for i in range(len(arr)): 
        print ("%f" %arr[i],"\t %d" %arr1[i])
    b=time.time()
    c=b-a
    print("Execution time of insertion sort is: %f" %c)
data=pd.read_csv("consolidated.csv")
arr = []
arre=data['avg_eric']
arr1e=data['month']

def bubbleSort_nico(arr,arr1):
    a=time.time()
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1] :
                arr[j], arr[j+1] = arr[j+1], arr[j]
                arr1[j], arr1[j+1] = arr1[j+1], arr1[j]
    print ("Sorted array is:")
    for i in range(len(arr)):
        print ("%f " %arr[i]," \t %d" %arr1[i])
    b=time.time()
    c=b-a
    print("Execution time of bubble sort is: %f" %c)
data=pd.read_csv("cons_nico.csv")
arrn =data['avg_nico']
monthn = data['month']

def insertionSort_nico(arr,arr1):
    a=time.time() 
    for i in range(1, len(arr)): 
        key = arr[i]
        key1 = arr1[i]
        j = i-1
        while j >=0 and key < arr[j] : 
                arr[j+1] = arr[j]
                arr1[j+1]=arr1[j]
                j -= 1
        arr[j+1] = key
        arr1[j+1]= key1
    print ("Sorted array is:") 
    for i in range(len(arr)): 
        print ("%f" %arr[i]," \t %d" %arr1[i])
    b=time.time()
    c=b-a
    print("Execution time of insertion sort is: %f" %c)
data=pd.read_csv("cons_nico.csv")
arr = []
arrn=data['avg_nico']
arr1n=data['month']

def bubbleSort_sanne(arr,arr1):
    a=time.time()
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1] :
                arr[j], arr[j+1] = arr[j+1], arr[j]
                arr1[j], arr1[j+1] = arr1[j+1], arr1[j]
    print ("Sorted array is:")
    for i in range(len(arr)):
        print ("%f " %arr[i]," \t %d" %arr1[i])
    b=time.time()
    c=b-a
    print("Execution time of bubble sort is: %f" %c)
datasanne=pd.read_csv("cons_sanne.csv")
arrs =datasanne['avg_sanne']
months = datasanne['month']

def insertionSort_sanne(arr,arr1):
    a=time.time()
    for i in range(1, len(arr)): 
        key = arr[i]
        key1 = arr1[i]
        j = i-1
        while j >=0 and key < arr[j] : 
                arr[j+1] = arr[j]
                arr1[j+1]=arr1[j]
                j -= 1
        arr[j+1] = key
        arr1[j+1]= key1
    print ("Sorted array is:") 
    for i in range(len(arr)): 
        print ("%f" %arr[i]," \t %d" %arr1[i])
    b=time.time()
    c=b-a
    print("Execution time of insertion sort is: %f" %c)
data=pd.read_csv("cons_sanne.csv")
arr = []
arrs=data['avg_sanne']
arr1s=data['month']

def compe():
    s1=
#------------------------------------------------------------------------------
def avg_speed_eric():
    x=PrettyTable()
    data=pd.read_csv("consolidated.csv")
    month=[]
    avgspeed=[]
    avglat=[]
    avglong=[]
    avgalti=[]
    month=data['month']
    avgspeed=data['avg_eric']
    avglat=data['eric_lat']
    avglong=data['eric_long']
    avgalti=data['eric_alti']
    n=len(data)+1
    #multiple regression
    X = data[['month','eric_lat','eric_long','eric_alti']] 
    Y = data['avg_eric']
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)

    nmonth= 10
    nlat= 55
    nlong= 2.5
    nalti=95
    print ('Average Speed: \n', regr.predict([[nmonth,nlat,nlong,nalti]]))
    plt.plot(data['month'], data['avg_eric'], color='red')
    plt.plot(data['month'], data['eric_lat'], color='green')
    plt.plot(data['month'], data['eric_long'], color='blue')
    plt.plot(data['month'], data['eric_alti'], color='grey')
    plt.title('month Vs avg_speed,avg_lat,avg_long and avg_alti', fontsize=14)
    plt.xlabel('month', fontsize=14)    
    plt.grid(True)
    plt.legend()
    plt.show()

def avg_speed_nico():
    x=PrettyTable()
    data=pd.read_csv("cons_nico.csv")
    month=[]
    avgspeed=[]
    avglat=[]
    avglong=[]
    avgalti=[]
    month=data['month']
    avgspeed=data['avg_nico']
    avglat=data['nico_lat']
    avglong=data['nico_long']
    avgalti=data['nico_alti']
    n=len(data)+1
    #multiple regression
    X = data[['month','nico_lat','nico_long','nico_alti']] 
    Y = data['avg_nico']
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)

    nmonth= 10
    nlat= 55
    nlong= 2.5
    nalti=95
    print ('Average Speed: \n', regr.predict([[nmonth,nlat,nlong,nalti]]))
    plt.plot(data['month'], data['avg_nico'], color='red')
    plt.plot(data['month'], data['nico_lat'], color='green')
    plt.plot(data['month'], data['nico_long'], color='blue')
    plt.plot(data['month'], data['nico_alti'], color='grey')
    plt.title('month Vs avg_speed,avg_lat,avg_long and avg_alti', fontsize=14)
    plt.xlabel('month', fontsize=14)    
    plt.grid(True)
    plt.legend()
    plt.show()

def avg_speed_sanne():
    x=PrettyTable()
    data=pd.read_csv("cons_sanne.csv")
    month=[]
    avgspeed=[]
    avglat=[]
    avglong=[]
    avgalti=[]
    month=data['month']
    avgspeed=data['avg_sanne']
    avglat=data['sanne_lat']
    avglong=data['sanne_long']
    avgalti=data['sanne_alti']
    n=len(data)+1
    #multiple regression
    X = data[['month','sanne_lat','sanne_long','sanne_alti']] 
    Y = data['avg_sanne']
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)

    nmonth= 10
    nlat= 55
    nlong= 2.5
    nalti=95
    print ('Average Speed: \n', regr.predict([[nmonth,nlat,nlong,nalti]]))

    plt.plot(data['month'], data['avg_sanne'], color='red')
    plt.plot(data['month'], data['sanne_lat'], color='green')
    plt.plot(data['month'], data['sanne_long'], color='blue')
    plt.plot(data['month'], data['sanne_alti'], color='grey')
    plt.title('month Vs avg_speed,avg_lat,avg_long and avg_alti', fontsize=14)
    plt.xlabel('month', fontsize=14)    
    plt.grid(True)
    plt.legend()
    plt.show()
#------------------------------------------------------------------------------
def searche(arr, x):
    a=time.time()
    data=pd.read_csv("consolidated.csv")
    month=data['month']
    avge = data['avg_eric']
    late=data['eric_lat']
    longe=data['eric_long']
    altie=data['eric_alti']
    for i in range(len(arr)):
        if arr[i] == x: 
            print("the index is: %d" %i)
            print("Month :%d" %month[i])
            print("Average Speed :%f" %avge[i])
            print("Latitude :%f" %late[i])
            print("Longitude :%f" %longe[i])
            print("Altitude :%f" %altie[i])
    b=time.time()
    c=b-a
    print("Execution time of linear search sort is: %f" %c)

def binarySearche(arr, l, r, x):
    a=time.time()
    data=pd.read_csv("consolidated.csv")
    month=data['month']
    avge = data['avg_eric']
    late=data['eric_lat']
    longe=data['eric_long']
    altie=data['eric_alti']
    if r >= l:
        mid = l + (r - l)/2
        mid=int(mid)
        if arr[mid] == x:
            print("Element is present at index %d" %mid)
            print("Month :%d" %month[mid])
            print("Average Speed :%f" %avge[mid])
            print("Latitude :%f" %late[mid])
            print("Longitude :%f" %longe[mid])
            print("Altitude :%f" %altie[mid])
        elif arr[mid] > x: 
            return binarySearch(arr, l, mid-1, x) 
        else: 
            return binarySearch(arr, mid+1, r, x)  
    else: 
        print("Not found")
    b=time.time()
    c=b-a
    print("Execution time of binary search sort is: %f" %c)

def searchn(arr, x):
    a=time.time()
    data=pd.read_csv("cons_nico.csv")
    month=data['month']
    avgn = data['avg_nico']
    latn=data['nico_lat']
    longn=data['nico_long']
    altin=data['nico_alti']
    for i in range(len(arr)):
        if arr[i] == x: 
            print("the index is: %d" %i)
            print("Month :%d" %month[i])
            print("Average Speed :%f" %avgn[i])
            print("Latitude :%f" %latn[i])
            print("Longitude :%f" %longn[i])
            print("Altitude :%f" %altin[i])
    b=time.time()
    c=b-a
    print("Execution time of linear search sort is: %f" %c)

def binarySearchn(arr, l, r, x):
    a=time.time()
    data=pd.read_csv("cons_nico.csv")
    month=data['month']
    avgn = data['avg_nico']
    latn=data['nico_lat']
    longn=data['nico_long']
    altin=data['nico_alti']
    if r >= l:
        mid = l + (r - l)/2
        mid=int(mid)
        if arr[mid] == x:
            print("Element is present at index %d" %mid)
            print("Month :%d" %month[mid])
            print("Average Speed :%f" %avgn[mid])
            print("Latitude :%f" %latn[mid])
            print("Longitude :%f" %longn[mid])
            print("Altitude :%f" %altin[mid])
        elif arr[mid] > x: 
            return binarySearchn(arr, l, mid-1, x) 
        else: 
            return binarySearchn(arr, mid+1, r, x)  
    else: 
        print("Not found")
    b=time.time()
    c=b-a
    print("Execution time of binary search sort is: %f" %c)
            
def searchs(arr, x):
    a=time.time()
    data=pd.read_csv("cons_sanne.csv")
    month=data['month']
    avgs = data['avg_sanne']
    lats=data['sanne_lat']
    longs=data['sanne_long']
    altis=data['sanne_alti']
    for i in range(len(arr)):
        if arr[i] == x: 
            print("the index is: %d" %i)
            print("Month :%d" %month[i])
            print("Average Speed :%f" %avgs[i])
            print("Latitude :%f" %lats[i])
            print("Longitude :%f" %longs[i])
            print("Altitude :%f" %altis[i])
    b=time.time()
    c=b-a
    print("Execution time of linear search sort is: %f" %c)

def binarySearchs(arr, l, r, x):
    a=time.time()
    data=pd.read_csv("cons_sanne.csv")
    month=data['month']
    avgs = data['avg_sanne']
    lats=data['sanne_lat']
    longs=data['sanne_long']
    altis=data['sanne_alti']
    if r >= l:
        mid = l + (r - l)/2
        mid=int(mid)
        if arr[mid] == x:
            print("Element is present at index %d" %mid)
            print("Month :%d" %month[mid])
            print("Average Speed :%f" %avgs[mid])
            print("Latitude :%f" %lats[mid])
            print("Longitude :%f" %longs[mid])
            print("Altitude :%f" %altis[mid])
        elif arr[mid] > x: 
            return binarySearchs(arr, l, mid-1, x) 
        else: 
            return binarySearchs(arr, mid+1, r, x)  
    else: 
        print("Not found")
    b=time.time()
    c=b-a
    print("Execution time of binary search sort is: %f" %c)
#------------------------------------------------------------------------------
def menu():
    print (30 * '-')
    print ("   M A I N - M E N U")
    print (30 * '-')
    choice = input("""
            1. Graphs
            2. Data View Using Trees
            3. Searching
            4. Sorting
            5. Multiple Regression analysis
            6. Exit
              Enter your choice: """)
    while True:
     if choice =="1":
        print ("1.Trajectory Motion")
        print ("2.Speed")
        print ("3.Elapsed Time")
        print ("4.Mean Speed Analysis")
        choice = input("Enter your Choice:")
        choice = int(choice)
        if choice == 1:
            print("Trajectory Motion...")
            trajectory()
            menu()
        elif choice == 2:
            print ("Speed vs Frequency graph ..")
            speed_frequency()
            menu()
        elif choice == 3:
            print ("Elapsed Time...")
            elapsed_time()
            menu()
        elif choice == 4:
            print ("Mean Speed Analysis...")
            mean_speed()
            menu()
        else:
            print("Invalid number... Try Again...")
            menu()
     elif choice == "2":
        print ("Data Printing")
        print("1. ERIC")
        print("2. NICO")
        print("3. SANNE")
        choice = input("Enter your Choice:")
        choice = int(choice)
        if choice ==1:
            eric_df()
            menu()
        elif choice ==2:
            nico_df()
            menu()
        elif choice ==3:
            sanne_df()
            menu()
        else:
            print("Invalid number try again...")
            menu()
     elif choice == "3":
        print ("Searching... a paticular data")
        print("1. ERIC")
        print("2. NICO")
        print("3. SANNE")
        choice = input("Enter your Choice:")
        choice = int(choice)
        if choice == 1:
            print("1. Linear Search")
            print("2. Binary Search")
            choice = input("Enter your Choice:")
            choice = int(choice)
            if choice == 1:
                print("Linear Search...")
                data=pd.read_csv('consolidated.csv')
                month=[]
                month=data['month']
                x1=int(input("Enter the value of month to search(number) :"))
                searche(month,x1)
                menu()
            elif choice == 2:
                print("Binary search.....")
                data=pd.read_csv('consolidated.csv')
                month=[]
                month=data['month']
                x1=int(input("Enter the value of month to search(number) :"))
                binarySearche(month, 0, len(month)-1, x1)
                menu()
            else:
                print("Invalid number... Try Again...")
                menu()
        elif choice == 2:
            print("1. Linear Search")
            print("2. Binary Search")
            choice = input("Enter your Choice:")
            choice = int(choice)
            if choice == 1:
                print("Linear Search...")
                data=pd.read_csv('cons_nico.csv')
                month=[]
                month=data['month']
                x1=int(input("Enter the value of month to search(number) :"))
                searchn(month,x1)
                menu()
            elif choice == 2:
                print("Binary search...")
                data=pd.read_csv('cons_nico.csv')
                month=[]
                month=data['month']
                x1=int(input("Enter the value of month to search(number) :"))
                binarySearchn(month, 0, len(month)-1, x1)
                menu()
            else:
                print("Invalid number... Try Again...")
                menu()
        elif choice == 3:
            print("1. Linear Search")
            print("2. Binary Search")
            choice = input("Enter your Choice:")
            choice = int(choice)
            if choice == 1:
                print("Linear Search...")
                data=pd.read_csv('cons_sanne.csv')
                month=[]
                month=data['month']
                x1=int(input("Enter the value of month to search(number) :"))
                searchs(month,x1)
                menu()
            elif choice ==2:
                print("Binary search...")
                data=pd.read_csv('cons_sanne.csv')
                month=[]
                month=data['month']
                x1=int(input("Enter the value of month to search(number) :"))
                binarySearchs(month, 0, len(month)-1, x1)
                menu()
            else:
                print("Invalid number... Try Again...")
        else:
            print("Invalid number... Try Again...")
     elif choice =="4" :
        print ("Select the bird to SORT the average Speed...")
        print("1. ERIC")
        print("2. NICO")
        print("3. SANNE")
        choice = input("Enter your Choice:")
        choice = int(choice)
        if choice == 1:
            print ("Sorting...")
            print("1. Bubble Sort")
            print("2. Insertion Sort")
            choice = input("Enter your Choice:")
            choice = int(choice)
            if choice == 1:
                print("Bubble Sort...")
                bubbleSort_eric(arre,monthe)
                menu()
            elif choice == 2:
                print("Insertion Sort...")
                insertionSort_eric(arre,arr1e)
                menu()
            else:
                print("Invalid number... Try Again...")
                menu()
        elif choice == 2:
            print ("Sorting...")
            print("1. Bubble Sort")
            print("2. Insertion Sort")
            choice = input("Enter your Choice:")
            choice = int(choice)
            if choice == 1:
                print("Bubble Sort... ERIC")
                bubbleSort_nico(arrn,monthn)
                menu()
            elif choice == 2:
                print("Insertion Sort... ERIC")
                insertionSort_nico(arrn,arr1n)
                menu()
            else:
                print("Invalid number... Try Again...")
                menu()
        elif choice == 3:
            print ("Sorting...")
            print("1. Bubble Sort")
            print("2. Insertion Sort")
            choice = input("Enter your Choice:")
            choice = int(choice)
            if choice == 1:
                print("Bubble Sort...")
                bubbleSort_sanne(arrs,months)
                menu()
            elif choice == 2:
                print("Insertion Sort...")
                insertionSort_sanne(arrs,arr1s)
                menu()
            else:
                print("Invalid number... Try Again...")
                menu()
        else:
                print("Invalid number... Try Again...")
                menu()
     elif choice == "5":
        print ("Multiple Regression Analysis TO Predict Speed ")
        print("1. ERIC")
        print("2. NICO")
        print("3. SANNE")
        choice = input("Enter your Choice:")
        choice = int(choice)
        if choice == 1:
            print("Regression on ERIC")
            avg_speed_eric()
            menu()
        elif choice == 2:
            print("Regression on NICO")
            avg_speed_nico()
            menu()
        elif choice == 3:
            print ("Regression on SANNE")
            avg_speed_sanne()
            menu()
        else:
            print("Invalid number... Try Again...")
            menu()
     elif choice=="6":
        print("--------------------Thanku you--------------")
        exit(1)
     else:
        print("---------Try again Enter the apt value--------")
menu()
