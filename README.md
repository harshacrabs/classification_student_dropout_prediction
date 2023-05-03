# classification_student_dropout_prediction


<h4>Authored by: Team Zenith</h4>
This project aims to contribute to the reduction of dropouts in higher education by using machine learning techniques.
It identifies students at risk of dropping out early in their academic path, allowing support strategies to be
implemented. The dataset used for this project is from a higher education institution in Portugal and contains
information about students at the time of enrollment like demographics, socio-economic factors along with their academic
performance during the first year(2 semesters). There is also the status of the student(Dropout/Enrolled/Graduate) for
each instance. For this project, the target variable is a binary classification model (Drop-out or Not Drop-out).

After necessary data cleaning and preprocessing, classification models are trained and evaluated using the relevant
machine learning algorithms.

Our prediction task is to determine whether the student will drop-out or not based on various attributes:

<h3>Install and/or Import Necessary Packages</h3>
<ol>
    <li>Data Manipulation: pandas, numpy</li>
    <li>Plot Handling/Visualization: matplotlib.pyplot, seaborn</li>
    <li>Preprocessing Data: StandardScaler, OneHotEncoder, train_test_split, cross_val_score, GridSearchCV,
        RandomizedSearchCV
    </li>
    <li>Models: DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
        XGBClassifier, KNeighborsClassifier, LogisticRegression, MLPClassifier
    </li>
    <LI>Performance Measures: accuracy_score, precision_score, recall_score, f1_score, fbeta_score, make_scorer,
        confusion_matrix, classification_report, plot_roc_curve, roc_auc_score, roc_curve, auc, RocCurveDisplay,
        PrecisionRecallDisplay, precision_recall_curve
    </LI>


</ol>


<h3>Load, Clean, and Prepare Data for Analysis</h3>

<ol>
    <li>Load the dataset into pandas DataFrame</li>
    <li>Clean the column names by removing whitespaces and replacing them with underscores
    </li>
    <li>Transform multi-class to binary class</li>
    <li>Create a new column with binary class values and drop the multi-class column from the dataframe
    </li>
    <li>Encode the target variable
    </li>
    <li>Transform the datatypes of categorical variables
    </li>

</ol>


<h3>Clean and Transform Data</h3>

<ol>
    <li>Drop macro-economic data attributes like GDP, Unemployment rate, and Inflation</li>
    <li>Encode dummies for categorical variables</li>
    <li>Visualize class counts to check for imbalance</li>
</ol>



<h3>Partition the Data into Training and Test Sets for Model Evaluation</h3>

<ol>
    <li>Utilize a training/test split of the data at 70% training and 30% testing</li>
    <li>Allocate as much data as possible to training</li>
</ol>


<h3>Address Data Imbalances with Oversampling Technique</h3>
<ol>
    <li>Use oversampling technique to address data imbalances</li>
    <li>Count the number of dropouts and non-dropouts</li>
    <li>Sample randomly from the dropout class and oversample it to match the number of non-dropouts</li>
    <li>Concatenate the oversampled dropout data to the original data</li>
    <li>Split the data into training and testing sets</li>
</ol>




<h3>Train the Models</h3>

<ol>
    <li>Decision Tree with Hyper-Parametric Tuning</li>
    <li>Random Forest- Ensemble Technique(Bagging)</li>
    <li>AdaBoost</li>
    <li>Gradient Boost</li>
    <li>XG Boost</li>
    <li>Logistic Regression with Hyperparameter Tuning</li>
    <li>Artificial Neural Networks(ANN)</li>
    <li></li>
</ol>



<h3>Conclusion</h3>

For each model, we perform the following steps:

<ul>
    <li>Define model and hyperparameters</li>
    <li>Fit the model with training data</li>
    <li>Predict target variable for test data</li>
    <li>Evaluate the model using confusion matrix and performance measures</li>
    <li>Store model results in a DataFrame</li>
</ul>



The project successfully predicts which students are at risk of dropping out. The model with the highest performance
measures is Artificial Neural Networks (ANN), followed by Logistic Regression and Gradient Boost. The project can be
utilized by higher education institutions to identify students at risk of dropping out early, enabling them to provide
support strategies to reduce dropouts
