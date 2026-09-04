## CHAPTER 3: CLASSIFICATION 

## BUILINDING A CLASIFICATION MODEL FROM THE MNIST DATASET

The classic example is the MNIST dataset—70,000 tiny handwritten digits from 0 to 9

## Binary Classification — The 5-Detector

To start simple, the chapter reduces the problem to one question: is this image a 5 or not? A classifier that answers only yes or no to one question is called a binary classifier.

The algorithm used is called ``Stochastic Gradient Descent Classifier``.
It is chosen because it is extremely fast and efficient on large datasets — it learns from one training instance at a time rather than the whole dataset at once, which also makes it good for situations where new data keeps arriving continuously.


## Training a Binary Classifier
The problem is simplified to one question: is this digit a 5 or not? This is called a binary classifier because it only distinguishes between two classes — 5 and not-5.

The algorithm chosen is the Stochastic Gradient Descent Classifier. It is picked for two reasons:

It handles very large datasets efficiently
It processes one training instance at a time, making it well suited for online learning where new data keeps arriving
Because it involves randomness during training, the random_state parameter is set to ensure results are reproducible every time you run it.

After training on the full training set, the classifier correctly identifies the sample digit as a 5.



## Perfomance Measures

The natural first instinct is to measure accuracy — what percentage of predictions were correct. The Stochastic Gradient Descent Classifier achieves over 93% accuracy using cross-validation. This looks great until you realize that a completely dumb classifier that always predicts "not-5" without even looking at the image achieves over 90% accuracy too.

This happens because only about 10% of images in the dataset are actually 5s. So blindly saying "not a 5" every single time is correct 90% of the time by pure probability. This situation is called a skewed dataset — where one class appears far more often than others. In such cases, accuracy is completely misleading and you need much better tools.


## Confusion Matrix

The confusion matrix is the foundation of proper classifier evaluation. It counts how many times each actual class was predicted as each possible class. For a binary classifier it produces four numbers:

True Negatives — the image was not a 5 and the classifier correctly said "not a 5." These are correct rejections.

False Positives — the image was not a 5 but the classifier wrongly said "it is a 5." These are false alarms.

False Negatives — the image was a 5 but the classifier wrongly said "not a 5." These are missed detections.

True Positives — the image was a 5 and the classifier correctly said "it is a 5." These are correct detections.

A perfect classifier would only have true positives and true negatives, with zeros everywhere else in the matrix. The confusion matrix gives you the full picture of where exactly your classifier is going wrong.

## Precision

Precision answers: when the classifier calls something positive, how often is it actually right?

It is calculated as true positives divided by the sum of true positives and false positives. The Stochastic Gradient Descent Classifier scored 72.9% precision, meaning when it said "that is a 5," it was correct about 73 times out of every 100 claims.

A trivial way to get perfect precision is to make only one single positive prediction ever and make sure it is correct. That gives 100% precision but is useless because it misses almost every actual positive. This is why precision is never used alone.

## Recall

Recall answers: out of all the actual positives in the entire dataset, how many did the classifier manage to catch?

It is calculated as true positives divided by the sum of true positives and false negatives. The Stochastic Gradient Descent Classifier scored 75.6% recall, meaning it caught about three quarters of all actual 5s in the dataset and missed the remaining quarter.

Recall is also called sensitivity or the true positive rate.

## The F1 Score

When you need one single number that reflects both precision and recall together, you use the F1 score. It is calculated as the harmonic mean of precision and recall. The harmonic mean is used rather than the regular average because it heavily penalizes extreme imbalances — if either precision or recall is very low, the F1 score collapses even if the other one is high. You only earn a high F1 score when both are reasonably good. The Stochastic Gradient Descent Classifier scored 0.742.

##  The Precision and Recall Tradeoff

Precision and recall are in constant tension with each other. You cannot freely maximize both at the same time. This is called the precision and recall tradeoff and it is one of the most important ideas in this chapter.

Every classifier internally computes a numerical decision score for each instance. If the score exceeds a set threshold, the instance is predicted positive. If it falls below, it is predicted negative.

Raising the threshold makes the classifier more selective — it only calls something positive when it is very confident. This increases precision because there are fewer false alarms, but decreases recall because some actual positives whose scores fall below the higher threshold now get missed.

Lowering the threshold makes the classifier more generous — it flags more instances as positive. This increases recall because more actual positives get caught, but decreases precision because more false alarms slip through.

You can retrieve raw decision scores, plot precision and recall against every possible threshold, and then pick the exact tradeoff your situation demands. For example, to achieve 90% precision you raise the threshold to around 7,816, but recall drops to about 44% as a consequence.

The practical rule: whenever someone demands very high precision, always ask at what recall that comes.

## The Receiver Operating Characteristic Curve and Area Under the Curve

The Receiver Operating Characteristic curve is another standard tool for binary classifiers. Instead of plotting precision against recall, it plots the true positive rate (which is the same as recall) on the vertical axis against the false positive rate on the horizontal axis, sweeping across all possible thresholds.

The false positive rate is the fraction of actual negatives that get wrongly flagged as positive.

A purely random classifier produces a straight diagonal line. A good classifier curves toward the top-left corner, meaning it achieves high recall while keeping false positives low.

The Area Under the Curve summarizes the entire curve in one number. A perfect classifier scores 1.0 and a random classifier scores 0.5. The Stochastic Gradient Descent Classifier scored 0.961, which looks great. But comparing it to a Random Forest Classifier which scored 0.9983 reveals it still has room to improve.

When to use which curve: Use the precision-recall curve when the positive class is rare or when false positives matter more than false negatives. Use the Receiver Operating Characteristic curve otherwise. On skewed datasets the Receiver Operating Characteristic curve can make a classifier look better than it actually is because the huge number of true negatives inflates the apparent performance.

## Multiclass Classification

A multiclass classifier distinguishes between more than two classes — for MNIST, all 10 digits at once.

Some algorithms like Random Forest and Naive Bayes handle multiple classes naturally. Others like Stochastic Gradient Descent and Support Vector Machines are strictly binary. Two strategies exist to extend binary classifiers to multiclass problems.

One Versus All — train one binary classifier per class. For 10 digits you train 10 classifiers. To classify a new image, run it through all 10, collect each classifier's decision score, and predict whichever class gave the highest score. This is the preferred strategy for most algorithms.

One Versus One — train one binary classifier for every possible pair of classes. For 10 classes this means 45 classifiers. To classify a new image, run it through all 45 classifiers and hold a vote — the class that wins the most head-to-head matchups wins. The advantage is that each classifier only trains on two classes at a time, keeping training sets small. This is preferred for algorithms like Support Vector Machines that slow down dramatically when training sets get large.

Scikit-Learn automatically picks the right strategy. Scaling the input features using StandardScaler boosts the Stochastic Gradient Descent Classifier's accuracy from around 84% to around 90% because the algorithm is sensitive to feature scales.

## Error Analysis

Rather than just looking at overall accuracy, you study the specific types of errors the model makes to find the most productive improvements.

You plot the full confusion matrix for all 10 classes as a grayscale image. Most images land on the main diagonal indicating correct predictions. To focus purely on errors, you normalize the matrix by dividing each row by the number of instances in that class, then zero out the diagonal entirely.

What the analysis reveals: The column for digit 8 is very bright, meaning many different digits are being misclassified as 8. Also, 3s and 5s are frequently confused with each other in both directions.

Why 3s and 5s get confused: The Stochastic Gradient Descent Classifier is a linear model. All it does is assign a weight to each pixel and sum weighted intensities to produce a score. The difference between a handwritten 3 and a 5 is only a few pixels — the small junction line between the top and bottom arc is in a slightly different position. A 3 drawn with that junction shifted slightly left looks like a 5 to this model. The classifier is highly sensitive to small shifts and rotations.

How to fix it: Preprocess images to center and de-rotate them. Engineer smarter features such as counting closed loops (digit 8 has two closed loops, digit 6 has one, digit 5 has none). Gather more training examples for the confused digit pairs.

## Multilabel Classification

Sometimes you want a classifier to attach multiple labels to a single instance at the same time. This is called multilabel classification.

A face recognition system is the clearest example — if it detects two people in a photo, it should output two tags simultaneously. For MNIST, you can label each digit with two independent things: whether the digit is large (7, 8, or 9) and whether it is odd. These are two separate binary outputs per image.

Not all classifiers support this natively. The K-Nearest Neighbors Classifier does. When asked about the digit 5 it correctly outputs: not large and odd.

Evaluation is done by computing the F1 score for each label separately and then averaging. A simple average treats all labels equally. A weighted average gives more importance to labels that appear more frequently in the dataset.

## Multioutput Classification

Multioutput classification is a further generalization where each output label is not just binary but can take many different values.

The chapter illustrates this with an image denoiser. Random noise is added to MNIST images to create corrupted inputs. The task is to output the clean original image. The output has 784 labels — one per pixel — and each label can be any value from 0 to 255. This is multioutput classification.

This blurs the boundary between classification and regression since predicting pixel intensity levels feels more like regression. The important point is that multioutput systems can have outputs that are both class labels and value labels mixed together.

A K-Nearest Neighbors Classifier trained on noisy-input to clean-output pairs successfully denoises test images, producing output close to the original digit.
