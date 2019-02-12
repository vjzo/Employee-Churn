import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as matplot
import seaborn as sns 
import os
import plotly.graph_objs as go#visualization
import plotly.offline as py#visualization
%matplotlib inline

os.chdir("C:/Users/vjred/Google Drive (vizo.datascience@gmail.com)/Hackathon 2018/Employee turnover")

df = pd.read_csv('HR.csv')
df.head()

#check for any missing values 
df.isnull().any()

df = df.rename(columns={'satisfaction_level': 'satisfaction', 
                        'last_evaluation': 'evaluation',
                        'number_project': 'projectCount',
                        'average_montly_hours': 'averageMonthlyHours',
                        'promotion_last_5years': 'promotion',
                        'sales' : 'department',
                        'left' : 'Churn'
                        })

df = df.rename(columns={'time_spend_company': 'yearsAtCompany',
                       'Work_accident': 'workAccident'})

df.head()

#Data features and their formats
df.dtypes

#How many employees in the dataset ? 
df.shape

#rate of Churn of the company 
Churn_rate = df.Churn.value_counts()/df.shape[0]
Churn_rate

#Describe the Statistical overview of the employees 
df.describe()

#Display the mean summary of Employees (Churn vs Non-Churn)
Churn_Summary = df.groupby('Churn')
Churn_Summary.mean()

#Create a Correlation matrix. To explain what features correlate the most with Churn, what other correlations can be found 
corr = df.corr()
corr

corr = (corr)
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    sns.heatmap(corr,
               xticklabels=corr.columns.values,
               yticklabels=corr.columns.values, mask=mask, vmax=.3, square=True)
plt.title('Heatmap of Correlation Matrix')
corr


# EDA 1. Distribution of Satisfaction, Evaluation, and Project Count
#setup the matplotlib
f, axes = plt.subplots(ncols=3, figsize=(15, 6))

#Graph Employee Satisfaction 
sns.distplot(df.satisfaction, kde=False, color="g", ax=axes[0]).set_title('Employee Satisfaction Distribution')
axes[0].set_ylabel('Employee Count')

#Graph Employee Evaluation 
sns.distplot(df.evaluation, kde=False, color="r", ax=axes[1]).set_title('Employee Evaluation Distribution')
axes[1].set_ylabel('Employee Count')

#Graph Employee Average Monthly Hours
sns.distplot(df.averageMonthlyHours, kde=False, color="b", ax=axes[2]).set_title('Employee Average Monthly Hours Distribution')
axes[2].set_ylabel('Employee Count')

sns.lmplot(x='satisfaction', y='evaluation', data=df, 
          fit_reg=False, #No regression line
          hue='Churn') #color by evolution stage

# EDA 2. K-Means Clustering of Employee Churn
from sklearn.cluster import KMeans
#Graph and create 3 clusters of Employee Churn 
kmeans = KMeans(n_clusters=3, random_state=2)
kmeans.fit(df[df.Churn==1][["satisfaction","evaluation"]])


kmeans_colors = ['green' if c==0 else 'blue' if c==2 else 'red' for c in kmeans.labels_]
fig = plt.figure(figsize=(10,6))
plt.scatter(x="satisfaction",y="evaluation", data = df[df.Churn==1],
            alpha=0.25, color = kmeans_colors) 
plt.xlabel("Satisfaction")
plt.ylabel("Evaluation")
plt.scatter(x=kmeans.cluster_centers_[:,0],y=kmeans.cluster_centers_[:,1],color="black",marker="X",s=100)
plt.title("Clusters of Employee Churn")
plt.show()

#EDA 3. Identifying cluster properties

sns.set(style="whitegrid")


# Draw a scatter plot while assigning point colors and sizes to different
# variables in the dataset
f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, right=True, top=True)
sns.scatterplot(x="satisfaction", y="averageMonthlyHours",
                hue="projectCount",
                palette="ch:r=-.2,d=.3_r",
                sizes=(1, 8), linewidth=0,
                data=df, ax=ax)

#EDA 4. Employee Satisfaction
#KDEPlot: Kernel Density Estimate Plot

fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['Churn'] == 0),'satisfaction'] , color='b',shade=True, label='no Churn')
ax=sns.kdeplot(df.loc[(df['Churn'] == 1),'satisfaction'] , color='r',shade=True, label='Churn')
plt.title('Employee Satisfaction Distribution - Churn V.S. No Churn')


fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['Churn'] == 0),'evaluation'] , color='b',shade=True, label='no Churn')
ax=sns.kdeplot(df.loc[(df['Churn'] == 1),'evaluation'] , color='r',shade=True, label='Churn')
plt.title('Employee Satisfaction Distribution - Churn V.S. No Churn')

# EDA 5. Employee Project Count
ax = sns.barplot(x="projectCount", y="projectCount", hue="Churn", data=df, estimator=lambda x: len(x) / len(df) * 100)
ax.set(ylabel="Percent")

# EDA 6. Employee department distribution
hrleft = df[df['Churn']==1]

hrleft = pd.DataFrame(hrleft.department.value_counts()).reset_index()
hrstay = pd.DataFrame(df.department.value_counts()).reset_index()

hr_merge = pd.merge(hrleft, hrstay, how='inner', on='index')

hr_merge = hr_merge.rename(columns={"department_x":'left', "department_y":'stay', "index":'department' })
hr_merge
    

sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(13, 7))

# Plot the total schools per city
sns.set_color_codes("pastel")
sns.barplot(x="stay", y='department', data=hr_merge,
            label="Total", color="b")

# Plot the total community schools per city
sns.set_color_codes("muted")
sns.barplot(x="left", y="department", data=hr_merge,
            label="Left", color="r")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set( ylabel="Department", title='Employees Per Department',
       xlabel="# of Employees")
sns.despine(left=True, bottom=True)

#Average Monthly Hours

#KDEPlot: Kernel Density Estimate Plot
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['Churn'] == 0),'averageMonthlyHours'] , color='b',shade=True, label='no Churn')
ax=sns.kdeplot(df.loc[(df['Churn'] == 1),'averageMonthlyHours'] , color='r',shade=True, label='Churn')
ax.set(xlabel='Employee Average Monthly Hours', ylabel='Frequency')
plt.title('Employee AverageMonthly Hours Distribution - Churn V.S. No Churn')


#Satisfaction
#KDEPlot: Kernel Density Estimate Plot
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['Churn'] == 0),'satisfaction'] , color='b',shade=True, label='no Churn')
ax=sns.kdeplot(df.loc[(df['Churn'] == 1),'satisfaction'] , color='r',shade=True, label='Churn')
ax.set(xlabel='Satisfaction', ylabel='Frequency')
plt.title('Satisfaction - Churn V.S. No Churn')

#Preprocessing

cat_var = ['department','salary','Churn','promotion']
num_var = ['satisfaction','evaluation','projectCount','averageMonthlyHours','yearsAtCompany', 'workAccident']
categorical_df = pd.get_dummies(df[cat_var], drop_first=True)
numerical_df = df[num_var]

new_df = pd.concat([categorical_df,numerical_df], axis=1)
new_df.head()

#Creating training and test datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve

# Create the X and y set
X = new_df.iloc[:,1:]
y = new_df.iloc[:,0]

# Define train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=123, stratify=y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Balancing datasets based on different sampling techniques
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE 

# Upsample using SMOTE
sm = SMOTE(random_state=12, ratio = 1.0)
x_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)


# Upsample minority class
X_train_u, y_train_u = resample(X_train[y_train == 1],
                                y_train[y_train == 1],
                                replace=True,
                                n_samples=X_train[y_train == 0].shape[0],
                                random_state=1)

X_train_u = np.concatenate((X_train[y_train == 0], X_train_u))
y_train_u = np.concatenate((y_train[y_train == 0], y_train_u))


# Downsample majority class
X_train_d, y_train_d = resample(X_train[y_train == 0],
                                y_train[y_train == 0],
                                replace=True,
                                n_samples=X_train[y_train == 1].shape[0],
                                random_state=1)
X_train_d = np.concatenate((X_train[y_train == 1], X_train_d))
y_train_d = np.concatenate((y_train[y_train == 1], y_train_d))


print("Original shape:", X_train.shape, y_train.shape)
print ("Upsampled SMOTE shape:", x_train_sm.shape, y_train_sm.shape)
print("Upsampled shape:", X_train_u.shape, y_train_u.shape)
print("Downsampled shape:", X_train_d.shape, y_train_d.shape)

# Applying Logistic regression on CV of different sampled data

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Create the Original, Upsampled, and Downsampled training sets
methods_data = {"Original": (X_train, y_train),
                "Upsampled": (X_train_u, y_train_u),
                "SMOTE":(x_train_sm, y_train_sm),
                "Downsampled": (X_train_d, y_train_d)}

# Loop through each type of training sets and apply 5-Fold CV using Logistic Regression
# By default in cross_val_score StratifiedCV is used

for method in methods_data.keys():
    lr_results = cross_val_score(LogisticRegression(), methods_data[method][0], methods_data[method][1], cv=10, scoring='f1')
    print(f"The best F1 Score for {method} data:")
    print (lr_results.mean())
 
print("Cross validation score: ",cross_val_score(LogisticRegression(class_weight='balanced'), X_train, y_train, cv=10, scoring='f1'))
print("Cross validation score mean: ",cross_val_score(LogisticRegression(class_weight='balanced'), X_train, y_train, cv=10, scoring='f1').mean())

# ROC results on smote sampling data with logistic regression

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

lr = LogisticRegression()

# Fit the model to the Upsampling data
lr = lr.fit(x_train_sm, y_train_sm)

print ("\n\n ---Logistic Regression Model---")
lr_auc = roc_auc_score(y_test, lr.predict(X_test))

print ("Logistic Regression AUC = %2.2f" % lr_auc)

lr2 = lr.fit(x_train_sm, y_train_sm)
print(classification_report(y_test, lr.predict(X_test)))

# Support Vector Machine (SVM)
from sklearn.svm import SVC  

svc = SVC(gamma='auto', probability=True)
svc.fit(x_train_sm, y_train_sm)
#print(svc.predict(X_test))

print ("\n\n ---Support Vector machine---")
svm_auc = roc_auc_score(y_test, svc.predict(X_test))

print ("SVM AUC = %2.2f" % svm_auc)

svm2 = svc.fit(x_train_sm, y_train_sm)
print(classification_report(y_test, svc.predict(X_test)))

# SVM model 

svc_result = cross_val_score(svc, x_train_sm, y_train_sm, cv=9, scoring='f1')

svc_result
print(svc_result.mean())


# Random Forest Model
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf_result = cross_val_score(rf, x_train_sm, y_train_sm, cv=9, scoring='f1')

rf_result
print(rf_result.mean())



from sklearn.metrics import roc_auc_score

rf = rf.fit(x_train_sm, y_train_sm)

print ("\n\n ---Random Forest Model---")
rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test))
print ("Random Forest AUC = %2.2f" % rf_roc_auc)
print(classification_report(y_test, rf.predict(X_test)))

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()  

gbc = gbc.fit(x_train_sm,y_train_sm)

gbc



gbc_result = cross_val_score(gbc, x_train_sm, y_train_sm, cv=9, scoring='f1')
gbc_result.mean()



from sklearn.metrics import roc_auc_score

print ("\n\n ---Gradient Boosting Model---")
gbc_auc = roc_auc_score(y_test, gbc.predict(X_test))
print ("Gradient Boosting Classifier AUC = %2.2f" % gbc_auc)
print(classification_report(y_test, gbc.predict(X_test)))


# Create ROC Graph
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(X_test)[:,1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
gbc_fpr, gbc_tpr, gbc_thresholds = roc_curve(y_test, gbc.predict_proba(X_test)[:,1])
svc_fpr, svc_tpr, svc_thresholds = roc_curve(y_test, svc.predict_proba(X_test)[:,1])


plt.figure()

# Plot Logistic Regression ROC
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % lr_auc)

# Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest Classifier (area = %0.2f)' % rf_roc_auc)

# Plot Decision Tree ROC
plt.plot(gbc_fpr, gbc_tpr, label='Gradient Boosting Classifier (area = %0.2f)' % gbc_auc)

# Plot SVM ROC 
plt.plot(svc_fpr, svc_tpr, label='SVM Classifier (area = %0.2f)' %svm_auc)


# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()

# Confusion Matrix for Linear Regresion
confusion_matrix(y_test, lr.predict(X_test))

confusion_matrix(y_test, gbc.predict(X_test))

confusion_matrix(y_test, svc.predict(X_test))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, rf.predict(X_test))

#Feature importances shows which feature is most important to determine the Churn rate



# Get Feature Importances
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
feature_importances = feature_importances.reset_index()
feature_importances

sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(13, 7))

# Plot the Feature Importance
sns.set_color_codes("pastel")
sns.barplot(x="importance", y='index', data=feature_importances,
            label="Total", color="b")

rf.predict_proba(X_test)
