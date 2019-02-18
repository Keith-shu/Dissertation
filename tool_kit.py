#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 15:12:10 2018

@author: Wang Yingtao
"""
#%%
"""
compute time span (about 3 years, 2000-2003)
"""

with open("dataset/ratings.dat","r") as f:
    temp=f.readline().split("::")[3]
    min=int(temp)
    max=int(temp)
    count=0
    for i in f:
        time=i.split("::")[3]
        time=int(time)
        if time<min:
            min=time
        if time>max:
            max=time

    print (min,max)

#%%
    
"""
compute interacting counts and time span for each movie

"""

with open("dataset/ratings.dat","r") as f:
    freq={}
    for i in f:
        data=i.split("::")
        movie_id=int(data[1])
        time=int(data[3])
        if movie_id in freq:
            freq[movie_id][0]+=1
            if freq[movie_id][1]>time:
                freq[movie_id][1]=time
            if freq[movie_id][2]<time:
                freq[movie_id][2]=time
        else:
            freq[movie_id]=[1,time,time]


_freq={}
for j in freq:
    _freq[j]=[freq[j][0],freq[j][2]-freq[j][1]+1]
    

with open("dataset/movie_feature","w") as f:
    for i in _freq:
        f.write(str(i)+','+str(_freq[i][0])+','+str(_freq[i][1])+'\n')

#%%
"""
import modules
"""
from sklearn.cluster import KMeans,MiniBatchKMeans,Birch,AffinityPropagation,MeanShift
from sklearn.metrics.pairwise import pairwise_distances_argmin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


#%%
"""
compute interacting counts and time span for each user
"""
freq_u={}
with open("dataset/ratings.dat","r") as f:
    for i in f:
        data=i.split("::")
        user_id=int(data[0])
        time=int(data[3])
        if user_id in freq_u:
            freq_u[user_id][0]+=1
            if freq_u[user_id][1]>time:
                freq_u[user_id][1]=time
            if freq_u[user_id][2]<time:
                freq_u[user_id][2]=time
        else:
            freq_u[user_id]=[1,time,time]


_freq_u={}
for j in freq_u:
    _freq_u[j]=[freq_u[j][0],freq_u[j][2]-freq_u[j][1]+1]
    

with open("dataset/user_feature","w") as f:
    for i in _freq_u:
        f.write(str(i)+','+str(_freq_u[i][0])+','+str(_freq_u[i][1])+'\n')
    


#%%
"""user features ,gender,age,occupation(career) """


user_features={}
with open("dataset/users.dat","r") as f:
    for i in f:
        data=i.split("::")
        user_id=data[0]
        gender=data[1]
        age=data[2]
        occupation=data[3]
        user_features[user_id]=[gender,age,occupation]
        
#%%
    """user gender """
#weekly_male={}
#weekly_female={}    #why not using hash table?
weekly_male=[0 for i in range(150)]
weekly_female=[0 for i in range(150)]
with open("dataset/ratings.dat","r") as f:
   for i in f:
        data=i.split("::")
        time=data[3]
        user_id=data[0]
        week=(int(time)-min)//(7*24*3600)
        gender=user_features[user_id][0]
        if gender=='M':
            weekly_male[week]+=1
        else:
            weekly_female[week]+=1
    
weekly_male=np.array(weekly_male) 
weekly_female=np.array(weekly_female)
x=np.arange(150)
plt.rcParams['figure.dpi']=100

plt.plot(x,weekly_male,fmt[0],label='male',linewidth=0.5,markersize=0.6,
             markerfacecolor='w',markeredgewidth=0.8)
plt.plot(x,weekly_female,fmt[1],label='female',linewidth=0.5,markersize=0.6,
             markerfacecolor='w',markeredgewidth=0.8) 

#plt.title("movie interaction counts by gender")
plt.xlabel("time span(week)")
plt.ylabel("interaction counts")

plt.legend()


#plt.savefig("user_features/gender",format="jpg",quality=90,dpi=200)
#%%
plt.rcParams['legend.frameon']=False

#%%
"""gender average"""

weekly_male=[0 for i in range(150)]
weekly_female=[0 for i in range(150)]

weekly_male_count=[[] for i in range(150)]
weekly_female_count=[[] for i in range(150)]
with open("dataset/ratings.dat","r") as f:
   for i in f:
        data=i.split("::")
        time=data[3]
        user_id=data[0]
        week=(int(time)-min)//(7*24*3600)
        gender=user_features[user_id][0]
        if gender=='M':
            weekly_male[week]+=1
            weekly_male_count[week].append(user_id)
        else:
            weekly_female[week]+=1
            weekly_female_count[week].append(user_id)

weekly_male_avg=[0 for i in range(150)]
weekly_female_avg=[0 for i in range(150)]
for i in range(150):
    if len(set(weekly_male_count[i]))!=0:
        weekly_male_avg[i]=weekly_male[i]/len(set(weekly_male_count[i]))
    if len(set(weekly_female_count[i]))!=0:
        weekly_female_avg[i]=weekly_female[i]/len(set(weekly_female_count[i]))




weekly_male_avg=np.array(weekly_male_avg) 
weekly_female_avg=np.array(weekly_female_avg)
x=np.arange(50)


plt.rcParams['figure.dpi']=300

plt.plot(x,weekly_male_avg[:50],fmt[0],label='male',linewidth=0.5,markersize=0.6,
             markerfacecolor='w',markeredgewidth=0.8)
plt.plot(x,weekly_female_avg[:50],fmt[1],label='female',linewidth=0.5,markersize=0.6,
             markerfacecolor='w',markeredgewidth=0.8) 

#plt.title("movie interaction counts by gender")
plt.xlabel("time span(week)")
plt.ylabel("interaction average")

plt.legend()

#%%   
"""user age

	*  1:  "Under 18"
	* 18:  "18-24"
	* 25:  "25-34"
	* 35:  "35-44"
	* 45:  "45-49"
	* 50:  "50-55"
	* 56:  "56+"
"""
ages_dict={ 1:  "Under 18",
	 18:  "18-24",
	 25:  "25-34",
	 35:  "35-44",
	 45:  "45-49",
	 50:  "50-55",
	 56:  "56+"}



weekly_age={}
ages=[1,18,25,35,45,50,56]
for j in ages:
    weekly_age[j]=[0 for i in range(150)]

with open("dataset/ratings.dat","r") as f:
    for i in f:
        data=i.split("::")
        user_id=data[0]
        time=data[3]
        week=(int(time)-min)//(7*24*3600)
        age=int(user_features[user_id][1])
        weekly_age[age][week]+=1

    
for i in weekly_age:
    weekly_age[i]=np.array(weekly_age[i])

n=len(ages_dict)

x=np.arange(50)
j=0
for i in weekly_age:
   plt.plot(x,weekly_age[i][:50],fmt[j],label=ages_dict[i],linewidth=0.5,markersize=0.6,
             markerfacecolor='w',markeredgewidth=0.8)
   j+=1
plt.rcParams['figure.dpi']=300





#plt.title("movie interaction counts by age")
plt.xlabel("time span(week)")
plt.ylabel("interaction counts")
plt.legend()
#plt.savefig("user_features/age",format="jpg",quality=90,dpi=200)

#%%

weekly_age={}
ages=[1,18,25,35,45,50,56]
for j in ages:
    weekly_age[j]=[0 for i in range(150)]

weekly_age_count={}
for i in ages:
    weekly_age_count[i]=[[] for j in range(150)]

with open("dataset/ratings.dat","r") as f:
    for i in f:
        data=i.split("::")
        user_id=data[0]
        time=data[3]
        week=(int(time)-min)//(7*24*3600)
        age=int(user_features[user_id][1])
        weekly_age_count[age][week].append(user_id)
        weekly_age[age][week]+=1

weekly_age_avg={}
for i in ages:
    weekly_age_avg[i]=[0 for j in range(150)]
    
for i in ages:
    for j in range(150):
        if(len(set(weekly_age_count[i][j]))!=0):
            weekly_age_avg[i][j]=weekly_age[i][j]/len(set(weekly_age_count[i][j]))

for i in weekly_age_avg:
    weekly_age_avg[i]=np.array(weekly_age_avg[i])

n=len(ages_dict)

x=np.arange(150)
j=0
for i in weekly_age:
   plt.plot(x,weekly_age_avg[i],fmt[j],label=ages_dict[i],linewidth=0.5,markersize=0.6,
             markerfacecolor='w',markeredgewidth=0.8)
   j+=1
plt.rcParams['figure.dpi']=300





#plt.title("movie interaction counts by age")
plt.xlabel("time span(week)")
plt.ylabel("interaction average")
plt.legend()


#%%%
    """user occupation
    
    *  0:  "other" or not specified
	*  1:  "academic/educator"
	*  2:  "artist"
	*  3:  "clerical/admin"
	*  4:  "college/grad student"
	*  5:  "customer service"
	*  6:  "doctor/health care"
	*  7:  "executive/managerial"
	*  8:  "farmer"
	*  9:  "homemaker"
	* 10:  "K-12 student"
	* 11:  "lawyer"
	* 12:  "programmer"
	* 13:  "retired"
	* 14:  "sales/marketing"
	* 15:  "scientist"
	* 16:  "self-employed"
	* 17:  "technician/engineer"
	* 18:  "tradesman/craftsman"
	* 19:  "unemployed"
	* 20:  "writer"
    
    """

weekly_occupation={}
occupations=range(21)
for j in occupations:
   weekly_occupation[j]=[0 for i in range(150)]

with open("dataset/ratings.dat","r") as f:
   for i in f:
        data=i.split("::")
        user_id=data[0]
        time=data[3]
        week=(int(time)-min)//(7*24*3600)
        occupation=int(user_features[user_id][2])
        weekly_occupation[occupation][week]+=1

    
for i in weekly_occupation:
   weekly_occupation[i]=np.array(weekly_occupation[i])

occupations_dict= {0:"other",
	 1:  "academic/educator",
	 2:  "artist",
	 3:  "clerical/admin",
	 4:  "college/grad student",
	 5:  "customer service",
	 6:  "doctor/health care",
	 7:  "executive/managerial",
	 8:  "farmer",
	 9:  "homemaker",
	10:  "K-12 student",
	11:  "lawyer",
	12:  "programmer",
	13:  "retired",
	14:  "sales/marketing",
	15:  "scientist",
	16:  "self-employed",
	17:  "technician/engineer",
	18:  "tradesman/craftsman",
	19:  "unemployed",
	20:  "writer"}

x=np.arange(50)
n=len(occupations_dict)
#%%
#markers,linestyle
linestyle=['-','--','-.',':']
markers=['1','2','D','*','s','>']
fmt=[]
for i in markers:
    for j in linestyle:
      fmt.append(i+j)

#%%
y=np.arange(150)
for i in weekly_occupation:
    plt.plot(y,weekly_occupation[i],fmt[i],label=occupations_dict[i],linewidth=0.3,markersize=0.6,
             markerfacecolor='w',markeredgewidth=0.8)
    
    
    
plt.rcParams['figure.dpi']=300
plt.rcParams['legend.frameon']=False

#plt.title("movie interaction counts by occupation")
plt.xlabel("time span(week)")
plt.ylabel("interaction counts")
plt.legend(fontsize="xx-small")
#plt.savefig("user_features/occupation",format="jpg",quality=90,dpi=200)
#%%
weekly_occupation={}
occupations=range(21)
for j in occupations:
   weekly_occupation[j]=[0 for i in range(150)]

weekly_occupation_count={}
for i in occupations:
    weekly_occupation_count[i]=[[] for j in range(150)]
    

with open("dataset/ratings.dat","r") as f:
   for i in f:
        data=i.split("::")
        user_id=data[0]
        time=data[3]
        week=(int(time)-min)//(7*24*3600)
        occupation=int(user_features[user_id][2])
        weekly_occupation_count[occupation][week].append(user_id)
        weekly_occupation[occupation][week]+=1

weekly_occupation_avg={}
for i in occupations:
    weekly_occupation_avg[i]=[0 for j in range(150)]

for i in weekly_occupation_avg:
    for j in range(150):
        if(len(set(weekly_occupation_count[i][j]))!=0):
            weekly_occupation_avg[i][j]=weekly_occupation[i][j]/len(set(weekly_occupation_count[i][j]))
    
for i in weekly_occupation_avg:
   weekly_occupation_avg[i]=np.array(weekly_occupation_avg[i])

occupations_dict= {0:"other",
	 1:  "academic/educator",
	 2:  "artist",
	 3:  "clerical/admin",
	 4:  "college/grad student",
	 5:  "customer service",
	 6:  "doctor/health care",
	 7:  "executive/managerial",
	 8:  "farmer",
	 9:  "homemaker",
	10:  "K-12 student",
	11:  "lawyer",
	12:  "programmer",
	13:  "retired",
	14:  "sales/marketing",
	15:  "scientist",
	16:  "self-employed",
	17:  "technician/engineer",
	18:  "tradesman/craftsman",
	19:  "unemployed",
	20:  "writer"}

x=np.arange(50)
n=len(occupations_dict)
y=np.arange(50)
for i in weekly_occupation:
    plt.plot(y,weekly_occupation_avg[i][:50],fmt[i],label=occupations_dict[i],linewidth=0.3,markersize=0.6,
             markerfacecolor='w',markeredgewidth=0.8)
    
    
    
plt.rcParams['figure.dpi']=300
plt.rcParams['legend.frameon']=False

#plt.title("movie interaction counts by occupation")
plt.xlabel("time span(week)")
plt.ylabel("interaction average")
plt.legend(fontsize="xx-small")

    
#%%

""" interaction counts by genres

   * Action
	* Adventure
	* Animation
	* Children's
	* Comedy
	* Crime
	* Documentary
	* Drama
	* Fantasy
	* Film-Noir
	* Horror
	* Musical
	* Mystery
	* Romance
	* Sci-Fi
	* Thriller
	* War
	* Western
    
"""
genre_list=["Action",
	"Adventure",
	"Animation",
	"Children's",
	"Comedy",
	"Crime",
	"Documentary",
	"Drama",
	"Fantasy",
	"Film-Noir",
	"Horror",
	"Musical",
	"Mystery",
	"Romance",
	"Sci-Fi",
	"Thriller",
	"War",
	"Western"]

genre_dict={}
for i in genre_list:
    genre_dict[i]=[0 for j in range(150)]
movie_genre=[[] for i in range(4000)]
with open("dataset/movies.dat","r") as f:
    for i in f:
        if not i:
            break
        i.decode("utf8","ignore")
        data=i.split("::")
        movie_id=int(data[0])
        genres=data[2].split("|")
        for j in genres:
            movie_genre[movie_id].append(j)
        

with open("dataset/ratings.dat","r") as f:
    for i in f:
        data=i.split("::")
        movie_id=int(data[1])
        time=data[3]
        week=(int(time)-min)/(7*24*3600)
        genres=movie_genre[movie_id]
        for j in genres:
            genre_dict[j.strip()][week]+=1
            

for i in genre_dict:
   genre_dict[i]=np.array(genre_dict[i])        

#%%
x=np.arange(150)
j=0
for i in genre_dict:
    plt.plot(x,genre_dict[i],fmt[j],label=i,linewidth=0.3,markersize=0.6,
             markerfacecolor='w',markeredgewidth=0.8)    
    j+=1
#plt.title("movie interaction counts by genre(50 weeks)")
plt.xlabel("time span(week)")
plt.ylabel("interaction counts")
plt.legend(fontsize="xx-small")
#plt.savefig("movie_features/genre",format="jpg",quality=90,dpi=200)


#%%
"""
   connection between interaction frequency and interest drift
"""

freq_max=0
freq_min=0
for i in freq_u:
    counts_tmp=freq_u[i][0]
    week_tmp=(freq_u[i][2]-freq_u[i][1])/(7*24*3600)
    freq_tmp=counts_tmp/(week_tmp+1)
    if freq_max<freq_tmp:
        freq_max=freq_tmp
    if freq_min>freq_tmp:
        freq_min=freq_tmp
        
print freq_max,freq_min

#step=freq_max/4
intervals=[i for i in range(0,freq_max,freq_max/4)]
#%%
print freq_max,intervals
#%%
#classfying users

user_freq_type=[0 for i in range(len(freq_u)+1)]

for i in freq_u:
    counts_tmp=freq_u[i][0]
    week_tmp=(freq_u[i][2]-freq_u[i][1])/(7*24*3600)
    freq_tmp=counts_tmp/(week_tmp+1)
    if freq_tmp<intervals[1]:
        user_freq_type[i]=0
    elif intervals[1]<=freq_tmp<intervals[2]:
        user_freq_type[i]=1
    elif intervals[2]<=freq_tmp<intervals[3]:
        user_freq_type[i]=2
    else:
        user_freq_type[i]=3
        
#print user_freq_type      
#%%
        
weekly_freq_type=[[] for i in range(4)]        
        
for i in range(len(weekly_freq_type)):
    weekly_freq_type[i]=[{} for j in range(150)]
        
for i in weekly_freq_type:
    for j in i:
        for k in genre_list:
            j[k]=0
        
with open("dataset/ratings.dat","r") as f:
    for i in f:
        data=i.split("::")
        user_id=int(data[0])
        movie_id=int(data[1])
        freq_type=user_freq_type[user_id]
        time=data[3]
        week=(int(time)-min)/(7*24*3600)
        _genre=movie_genre[movie_id][0].strip()
        weekly_freq_type[freq_type][week][_genre]+=1
        

#%%
    
#print weekly_freq_type[3][10]
freq_u
user_freq_type
genre_count=[]
for i in range(4):
    genre_count.append([0,0])
genres=18
average_u=[0 for i in range(len(freq_u)+1)]
for i in freq_u:
    average_u[i]=freq_u[i][0]/18

genres_u_all=[{} for i in range(len(freq_u)+1)]
for i in genres_u_all:
    for j in genre_list:
        i[j]=0
with open("dataset/ratings.dat","r") as f:
    for i in f:
        data=i.split("::")
        user_id=int(data[0])
        movie_id=int(data[1])
        _genre=movie_genre[movie_id][0].strip()
        genres_u_all[user_id][_genre]+=1
    

#%%
variance_u=[0 for i in range(len(freq_u)+1)]
for i in range(1,len(genres_u_all)):
    for j in genres_u_all[i]:
        if genres_u_all[i][j]>=average_u[i]:
            variance_u[i]+=1
    
for i in range(1,len(freq_u)+1):
    _type=user_freq_type[i]
    genre_count[_type][0]+=1
    genre_count[_type][1]+=variance_u[i]
    
for i in genre_count:
    print float(i[1])/i[0]








#%%

"""analysis  code for u-TWMF and p-TWMF """

#%%

hours_count=[0 for i in range(24)]

with open("dataset/ratings.dat","r") as f:
    for i in f:
        data=i.split("::")
        time=int(data[3])
        second=time%86400
        hour=second/3600
        hours_count[hour]+=1
        
        

#%%
        
hours_count=np.array(hours_count)
x=np.arange(24)




plt.rcParams['figure.dpi']=300
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'
plt.plot(x,hours_count,linewidth=0.5,markersize=0.6,
             markerfacecolor='w',markeredgewidth=0.8)


#plt.title("movie interaction counts in a day")
plt.xlabel("24 hours in a day")
plt.ylabel("interaction counts")

plt.plot(x,hours_count)
#%%


hours_count_occupation=[[] for i in range(len(occupations_dict))]
for i in range(len(hours_count_occupation)):
    hours_count_occupation[i]=[0 for j in range(24)]
    
with open("dataset/ratings.dat","r") as f:
    for i in f:
        data=i.split("::")
        time=int(data[3])
        second=time%86400
        hour=second//3600
        user_id=data[0]
        occupation=int(user_features[user_id][2])
        hours_count_occupation[occupation][hour]+=1

x=np.arange(24)

for i in range(len(hours_count_occupation)):
    plt.plot(x,hours_count_occupation[i],fmt[i],label=occupations_dict[i],linewidth=0.3,markersize=0.6,
             markerfacecolor='w',markeredgewidth=0.8)    

#plt.title("movie interaction counts in a day(different occupations)")
plt.xlabel("24 hours in a day")
plt.ylabel("interaction counts")
plt.legend(fontsize="4.5")
#plt.savefig("movie_features/genre",format="jpg",quality=90,dpi=200)

        
#%%


hours_count_age={}
for i in ages_dict:
    hours_count_age[i]=[0 for j in range(24)]

    
with open("dataset/ratings.dat","r") as f:
    for i in f:
        data=i.split("::")
        time=int(data[3])
        second=time%86400
        hour=second//3600
        user_id=data[0]
        age=int(user_features[user_id][1])
        hours_count_age[age][hour]+=1

x=np.arange(24)
j=0
for i in hours_count_age:
    plt.plot(x,hours_count_age[i],fmt[j],label=ages_dict[i],linewidth=0.3,markersize=0.6,
             markerfacecolor='w',markeredgewidth=0.8)
    j+=1    

#plt.title("movie interaction counts in a day(different ages)")
plt.xlabel("24 hours in a day")
plt.ylabel("interaction counts")
plt.legend(fontsize="6.5")

#%%

hours_count_gender={'M':[],'F':[]}

for i in hours_count_gender:
    hours_count_gender[i]=[0 for j in range(24)]
    
with open("dataset/ratings.dat","r") as f:
    for i in f:
        data=i.split("::")
        time=int(data[3])
        second=time%86400
        hour=second//3600
        user_id=data[0]
        gender=user_features[user_id][0]
        hours_count_gender[gender][hour]+=1

x=np.arange(24)


plt.plot(x,hours_count_gender['M'],fmt[0],label='male',linewidth=0.3)
plt.plot(x,hours_count_gender['F'],fmt[1],label='female',linewidth=0.3)  

#plt.title("movie interaction counts in a day(different gender)")
plt.xlabel("24 hours in a day")
plt.ylabel("interaction counts")
plt.legend()



#%%
plt.rcParams['figure.dpi']=100
hours_count_genre={}
for i in genre_list:
    hours_count_genre[i]=[0 for j in range(24)]
 
with open("dataset/ratings.dat","r") as f:
    for i in f:
        data=i.split("::")
        time=int(data[3])
        second=time%86400
        hour=second/3600
        movie_id=int(data[1])
        genre=movie_genre[movie_id][0].strip()
        hours_count_genre[genre][hour]+=1


x=np.arange(24)
j=0
for i in hours_count_genre:
    plt.plot(x,hours_count_genre[i],fmt[j],label=i,linewidth=0.3,markersize=0.6,
             markerfacecolor='w',markeredgewidth=0.8)
    j+=1
    
#plt.title("movie interaction counts i a day(different movie genres)")
plt.xlabel("24 hours in a day")
plt.ylabel("interaction counts")
plt.legend(fontsize=4)





#%%


occupation_count={}
for i in occupations_dict:
    occupation_count[occupations_dict[i]]=0

for i in user_features:
    _occupation=int(user_features[i][2])
    occupation_count[occupations_dict[_occupation]]+=1

#%%
plt.rcParams['figure.dpi']=300
index=np.arange(len(occupation_count))
plt.xticks(index,tuple(occupation_count.keys()),rotation='vertical')
plt.bar(index,occupation_count.values(),0.3)
#plt.title("occupation distribution")
plt.ylabel("Number of people")
plt.show()

#%%
plt.rcParams['figure.dpi']=300
hours_count_gender={'M':[],'F':[]}

for i in hours_count_gender:
    hours_count_gender[i]=[[] for j in range(24)]
    
with open("dataset/ratings.dat","r") as f:
    for i in f:
        data=i.split("::")
        time=int(data[3])
        second=time%86400
        hour=second//3600
        user_id=data[0]
        gender=user_features[user_id][0]
        hours_count_gender[gender][hour].append(user_id)

x=np.arange(24)
#male_num=len(hours_count_gender['M'])/len(set(hours_count_gender['M']))
#female_num=len(hours_count_gender['F'])/len(set(hours_count_gender['F']))
gender_count={'M':[],'F':[]}


for i in hours_count_gender:
    for j in hours_count_gender[i]:
        gender_count[i].append(len(j)/len(set(j)))
        

plt.plot(x,gender_count['M'],fmt[0],label='male',linewidth=0.3)
plt.plot(x,gender_count['F'],fmt[1],label='female',linewidth=0.3)  

#plt.title("movie interaction counts in a day(different gender)")
plt.xlabel("24 hours in a day")
plt.ylabel("interaction average")
plt.legend()

#%%
plt.rcParams['figure.dpi']=300
hours_avg_count={}

for i in ages_dict:
    hours_avg_count[i]=[[] for j in range(24)]
hours_count_age={}
for i in ages_dict:
    hours_count_age[i]=[0 for j in range(24)]

    
with open("dataset/ratings.dat","r") as f:
    for i in f:
        data=i.split("::")
        time=int(data[3])
        second=time%86400
        hour=second//3600
        user_id=data[0]
        age=int(user_features[user_id][1])
        hours_avg_count[age][hour].append(user_id)
        hours_count_age[age][hour]+=1

hours_avg_count_ages={}
for i in ages_dict:
    hours_avg_count_ages[i]=[0 for j in range(24)]
    
for i in ages_dict:
    for j in range(24):
        hours_avg_count_ages[i][j]=hours_count_age[i][j]/len(set(hours_avg_count[i][j]))
        
        
        
x=np.arange(24)
j=0
for i in ages_dict:
    plt.plot(x,hours_avg_count_ages[i],fmt[j],label=ages_dict[i],linewidth=0.3,markersize=0.6,
             markerfacecolor='w',markeredgewidth=0.8)
    j+=1    

#plt.title("movie interaction counts in a day(different ages)")
plt.xlabel("24 hours in a day")
plt.ylabel("interaction average")
plt.legend(fontsize="6.5")


#%%
plt.rcParams['figure.dpi']=100
    

hours_count_occupation=[[] for i in range(len(occupations_dict))]
for i in range(len(hours_count_occupation)):
    hours_count_occupation[i]=[0 for j in range(24)]

hours_avg_occupation=[[] for i in range(len(occupations_dict))]
for i in range(len(hours_avg_occupation)):
    hours_avg_occupation[i]=[[] for j in range(24)]
    
with open("dataset/ratings.dat","r") as f:
    for i in f:
        data=i.split("::")
        time=int(data[3])
        second=time%86400
        hour=second//3600
        user_id=data[0]
        occupation=int(user_features[user_id][2])
        hours_avg_occupation[occupation][hour].append(user_id)
        hours_count_occupation[occupation][hour]+=1


hours_avg_count_occupation=[[] for i in range(len(occupations_dict))]
for i in range(len(hours_avg_count_occupation)):
    hours_avg_count_occupation[i]=[0 for j in range(24)]
x=np.arange(24)

for i in range(len(hours_avg_occupation)):
    for j in range(24):
        if len(set(hours_avg_occupation[i][j]))==0:
            continue
        hours_avg_count_occupation[i][j]=hours_count_occupation[i][j]/len(set(hours_avg_occupation[i][j]))

for i in range(len(hours_count_occupation)):
    plt.plot(x,hours_avg_count_occupation[i],fmt[i],label=occupations_dict[i],linewidth=0.3,markersize=0.6,
             markerfacecolor='w',markeredgewidth=0.8)    

#plt.title("movie interaction counts in a day(different occupations)")
plt.xlabel("24 hours in a day")
plt.ylabel("interaction average")
plt.legend(fontsize="2.5")   
    
    
    
    