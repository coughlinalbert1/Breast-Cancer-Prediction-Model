Predicting Breast Cancer
Albert Coughlin




1. INTRODUCTION 										
1.1 Background 
1.2 Significance 
1.3 Statement of Problem 
2. REVIEW OF RELATED LITERATURE 
3. METHODOLOGY 
3.1  Design 
4. PERFORMANCE EVALUATION 
4.1 Accuracy 
4.2 Precision 
4.3 Recall 
4.4 F1-Score 
4.5 confusion matrix
5. REFERENCES 
6. PYTHON CODE 







Introduction
1.1 Background
This project focuses on using classification algorithms to build a predictive model to accurately predict if characteristics of the cell nuclei from a breast mass is benign or malignant. In other words, if the mass is cancerous or not. The data used is from the UC Irvine Machine Learning Repository but is imported using the machine learning library in Python called Sckit Learn

1.2 Significance
Machine learning in the medical field has some promising research for it’s use. (1)  AWS has its influence in this field. AWS has Amazon SageMaker which is used in the healthcare field to build predictive models to help with the diagnostic process.

1.3 Statement of Problem
The goal of this project is to build a predictive model, train the model, and test it using a dataset. The preicions score, accuracy score, recall score, F1-score, and a confusion matrix will be used to see how well the Sckit-learn machine learning algorithm works with this dataset. A report will be generated as well at the end. 












Review of Related Literature
There is various research done on using machine learning or AI in the healthcare field. AWS also has their hands in the field with their various machine learning technologies. Some research even suggests that in some aspects, the applications of AI produces results as good or even better than human beings. (1)

Davenport, et al. (2019) dive into this subject in their paper The potential for artificial intelligence in healthcare. In their paper, they find neural networks, robots, rule-based expert systems,natural langue processing, deep learning, and automation being used in the healthcare field to aid the patient outcomes, administrative processes, and more. They do not think AI will destroy jobs, only augment the current jobs. Over time, AI will make more of an impact in the healthcare world. 

Antari’s paper Artificial Intelligence for Medical Diagnostics—Existing and Future AI Technology talks about some of the ethical concerns and potential problems with AI in the medical diagnostics field. The paper brings up the problems of the algorithms having bias that cannot be resolved because the training data is biased.















Methodology
There are many classifier algorithms that can be used to build a predictive model. The main ones are K-nearest neighbors, support vector machine, Naive Bayes classifier, K-means clustering, logistic regression, and decision tree (random forest expands this algorithm). Each algorithm has its specialization and works better for different problems.

For the task of classifying breast mass information into whether it is malignant or benign, the decision tree algorithm is a good choice. Decision trees defined by IBM is “a non-parametric supervised learning algorithm, which is utilized for both classification and regression tasks. It has a hierarchical, tree structure, which consists of a root node, branches, internal nodes and leaf nodes.” (2)

Decision trees are simple to use and understand the output. A decision tree algorithm works by creating a model to predict values based on the target by learning rules from the data features. The root node partitions the data based on the attribute value and how well it helps classification. Trees are very intuitive to understand because it follows how humans make decisions.  Decision trees are not effected heavily by outliers which is very important when it comes to healthcare decisions. 

Sckit-learn allows two metrics to be used for the decision tree algorithm, gini and entropy. You can also specify other parameters for the algorithm like alpha, class weight, criterion, max depth, max features, max leaf nodes, and more. I found results were not too different when changing between gini and entropy. A max depth of 5 gave me more accurate results as well. I used jupyter notebook on VSCode to build this model and ran it using Anaconda to simplify package management. 

When getting the accuracy score, precision score, recall score, and f1 score, I decided to do 10 trials and record these scores for all 10. I took the average and used that to represent the score I received using this model.















Performance Evaluation
4.1 Accuracy 
The accuracy score evaluates the predictions that the Classification model made. The number is the percentage of predictions made that were correct. The mean accuracy score from the model was 0.929078014 or  93%. A perfect accuracy score would be 1 and a terrible score would be 0. 

The formula for accuracy score is  correct predictionstotal predictions 
The formula can also be writtentrue positives + true negativestrue positives + true negatives + false positives + false negatives

4.2 Precision 
Precision is the true positives divided by all of the positives. In other words, the ratio of the two. This is important because this model is for predicting breast cancer and we dont want to incorrectly diagnose someone with breast cancer. The mean precision score from the model was 93%. 
The formal for precision is true positivestrue positives + false positives

4.3 Recall 
The recall score measures how well the model is able to diagnose a patient with breast cancer. This score is important to have high because we do not want someone with breast cancer being diagnosed as benign. The mean recall score from the model was 96%.
The formula for recall score is true positivetrue positive + false negative
4.4 F1-Score 
F1-score combines the recall and precision score to give an overall metric on how the model handles false positives (precision) and false negatives (recall).The mean F1-score from the model was 94%.
 2recall  precisionrecall + precision

4.5 Confusion Matrix
Confusion matrix are very helpful with interpreting the data that we get from building and using a predictive model. It shows us how many correct predictions the model made and how many incorrect. During this specific instance, the model correctly predicted 96% of malignant masses and correctly predicted 93% of the benign masses. In other words, it shows true positive, true negatives, false positives, and false negatives. This image was generated with the confusion matrix plot function.








References
Davenport, Thomas, and Ravi Kalakota. “The Potential for Artificial Intelligence in Healthcare.” Future healthcare journal, June 2019. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6616181/. 
“What Is a Decision Tree.” IBM. Accessed November 17, 2023. https://www.ibm.com/topics/decision-trees. 
Al-Antari, Mugahed A. “Artificial Intelligence for Medical Diagnostics-Existing and Future AI Technology!” Diagnostics (Basel, Switzerland), February 12, 2023. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9955430/. 
“API Reference.” scikit. Accessed November 17, 2023. https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics. 
“Decision Tree Learning.” Wikipedia, November 9, 2023. https://en.wikipedia.org/wiki/Decision_tree_learning. 
