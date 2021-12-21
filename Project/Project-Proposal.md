# Predicting Diameter and Physical Harm of Asteroids using Machine Learning
**Authors** :
Colin Campbell (c_c953), Jake Worden (jrw294), Leah Lewis (lrl68) and Ryan Wakabayashi (rjw102)

**Abstract** : It is a standalone section. It is written to give the reader a summary of your work. Be sure to specific, yet brief. Even though the abstract comes first in your paper, it is sometimes easier to write the abstract last. (150-300 words)

## Introduction

Your project report is the formal description of your project. The format is similar to the presentation but we want you to fully elaborate on what you did. You could provide some background knowledge about the data you are analyzing, clearly and concisely present your research question, describe the dataset that can help answer your question. (0.5-1 page) 

**Note:** you can transform it to Jupyter Notebook and add sections below as markdown cells. 

## Problem Statement 
* Give a clear and complete statement of the problem. 
  What is the benchmark you are using.  Why?  Where does the data come from, what are its characteristics?
* Include informal success measures (e.g. accuracy on cross-validated data, without specifying ROC or precision/recall etc) that you planned to use.
* What do you hope to achieve? 

* Question: How to use machine learning to predict the diameter of asteroids and classify them as physically hazardous.
* Asteroid diameter prediction based upon Asteroid_Updated.csv from Kaggle.
* Predict whether an asteroid is physically hazardous to Earth. 

* Success measures:
	* 5 - 10 fold CV accuracy for all models
	* Regression models: R^2 score
	* Classification models: Precision, Recall, ROC/AUC
	
* Hope to achieve >85% R^2 for regression models (based upon kaggle responses) and then >=80% for the classification models (low goal based on amount of data for imbalanced classes).

### Related Work

* Include background material as appropriate: who cares about this problem, what impact it has, what implications better solutions might have.
* Included a brief summary of any related work you know about.
* Benchmark implementations - see [paperswithcode.com](paperswithcode.com) as a good start 

**Link to other work:** [Asteroid Diameter Estimators with added difficulty](https://www.kaggle.com/liamkesatoran/asteroid-diameter-estimators-with-added-difficulty)

## Data Management 

In this section you should address the questions of interest and interpret the results in terms of the questions of interest you proposed. (1-5 pages, including relevant tables and figures, please adjust your figures to appropriate sizes).

- Describe how did you evaluate your solution
- What evaluation metrics did you use?
- Describe a baseline system
- How much did your system outperform the baseline?
- Were there other systems evaluated on the same dataset? How did your system do in comparison to theirs?
- Show graphs/tables with results
- Error analysis
- Suggestions for future improvements

Description of the dataset (dimensions, names of variables with their description)

### Data Gathering

Answer the questions from *Motivation* (Sec 31.), *Composition* (Sec 3.2), and *Collection* (Sec 3.3) of the [Datasheets For Datasets](https://arxiv.org/abs/1803.09010) paper here. 
* If benchmarks, describe the data in details.
* If data collections, justify your methods in terms of data statement. 

(usually algorithms, or data cleaning or wrangling approaches). 

Justify your methods in terms of the problem statement. What did you consider but *not* use? In particular, be sure to include every method you tried, even if it didn't "work". When describing methods that didn't work, make clear how they failed and any evaluation metrics you used to decide so. How was that a data-driven decision? 

#### *Motivation*
* This database was acquired from the Jet Propulsion Laboratory at California Institute of Technology's "Solar System Dynamics" on behalf of NASA
* This information is related to the orbits, physical and characteristics, and discovery cirumstances for most known natural bodies in our solar system

#### *Composition*
* There are 839714 instances in the dataset representing asteroids in our solar system
	
| Feature | Description | Dtype | Null |
| ------- | ----------------- | ------ | :------: |
| a | Semi-major axis(au) | float64 | 2 |
| e | Eccentricity | float64 | 0 |
| i | Inclination with respect to x-y ecliptic plain(deg) | float64 | 0 |
| om | Longitude of the ascending node | float64 | 0 |
| w | Argument of perihelion | float64 | 0 |
| q | Perihelion distance(au) | float64 | 0 |
| ad | Aphelion distance(au) | float64 | 6 |
| per_y | Oribital period(YEARS) | float64 | 1 |
| data_arc | Data arc-span(d) | float64 | 15474 |
| condition_code | Orbit condition code | object | 867 |
| n_obs_used | Number of Observation used | int64 | 0 |
| H | Absolute magnitude parameter | float64 | 2689 |
| neo | Near Earth Object | object | 6 |
| pha | Physically Hazardous Asteroid | object | 16442 |
| diameter | Diameter of asteroid(Km) | object | 702078 |
| extent | Object bi/tri axial ellipsoid dimensions(Km) | object | 839696 |
| albedo | Geometric albedo | float64 | 703305 |
| rot_per | Rotation Period(h) | float64 | 820918 |
| GM | Standard gravitational parameter, Product of mass and gravitational constant | float64 | 839700 |
| BV | Color index B-V magnitude difference | float64 | 838693 |
| UB | Color index U-B magnitude difference | float64 | 838735 |
| IR | Color index I-R magnitude difference | float64 | 839713 |
| spec_B | Spectral taxonomic type(SMASSII) | object | 838048 |
| spec_T | Spectral taxonomic type(Tholen) | object | 838734 |
| G | Magnitude slope parameter | float64 | 839595 |
| moid | Earth minimum orbit intersection distance(au) | float64 | 16442 |
| class | Asteroid orbit class | object | 0 |
| n | Mean motion(deg/d) | float64 | 2 |
| per | Orbital period(d) | float64 | 6 |
| ma | Mean anomaly(deg) | float64 | 8 |

* Shape: (839714 , 31)
* Memory usage: 198.6+ MB

**Dataset found here:** [Asteroid_Updated.csv](https://www.kaggle.com/basu369victor/prediction-of-asteroid-diameter?select=Asteroid_Updated.csv)

### Data Pre-processing, Cleaning, Labeling, and Maintenance 

* Answer the questions from *Motivation, Composition, and Collection* sections of the [Datasheets For Datasets](https://arxiv.org/abs/1803.09010) paper here. 

- Read in the .csv and visualized .head() and .info()
- Checked the number of Null values. If the sum of null values are > 700,000, we dropped the column
- If the remaining column has only Nulls, it is dropped
- If the remaining rows contain any Nulls, it is dropped
### Exploratory Data Analysis 

What Data Acquisition, Cleaning, and Processing Tools have you used.  Why? 

* Describe the methods you explored (usually algorithms, or data cleaning or wrangling approaches). 
* Justify your methods in terms of the problem statement. 
* What did you consider but *not* use? In particular, be sure to include every method you tried, even if it didn't "work". 

**NOTE:** Move from .md to .ipynb format when you plan to show EDA (either project proposal or midterm checkpoint)

## Machine Learning Approaches

In this section, you could describe the methods you used in your analysis. For example, if you are doing classifications, you could introduce the methods like logistic regression, discriminant analysis, support vector machines. You don't have to write formulas if you don't want to do so. It is fine to describe the methods in words. This section basically is a description of the methodologies that you have used for analyzing your data. (up to 2pages)
Describe the choice of Machine Learning Tool.  Refer ro related work, if applicable.  

* Evaluate a primary model and in addition a "baseline" model. 
  * The baseline is typically the simplest model that's applicable to that data problem
    * Naive Bayes for classification
	* K-means on raw feature data for clustering.
* Evaluate state-of-art model 
  * Research gitHuib, paperswithcode, Kaggle and similar. 
  * If not applicable, talk to the instructor.  
  
**Hint** Goal is to have some sort of baseline evaluation by Nov 11th checkpoint to establish a scale by which to measure your project's performance. Compare the performance of your baseline model and primary model and explain the differences.

** This is where all the methods you have tried go, including state-of-art if any **

### Describe the ML methods that you used and the reasons for their choice. 
What is the family of machine learnign algorithms you are using and why? 
* Supervised or Unsupervised?
* Regression or classification?

### Justify ML algorithms in terms of the problem itself and the methods you want to use. 
* How did you employ them? 
* What features worked well and what didn't?
* Provide documentation for integration  

### Tools and Infrastructure Tried and Not Used

Describe any tools and infrastruicture that you tried and ended up not using.
What was the problem? 
Describe infrastructure used. 

## Experiments

Give a detailed summary of the results of your work.

 * Setup - Here is where you specify the exact performance measures you used.  
   * Describe the data used in experiment for presenting dataset: Datasheets for Dataset template 
   * Describe your accuracy or quality measure, and your performance (runtime or throughput) measure. 
   
 * Please use visualizations whenever possible. Include links to interactive visualizations if you built them. 
 
 * You can also submit a separated notebook as an appendix to your report if that makes the visualization/interaction task easier. 
   * It would be reasonable to submit your report as a notebook, but please make sure it runs on one of the two standard environments, and that you include any required files. 

## Conclusion
In this section give a high-level summary of your results. If the reader only reads one section of the report, this one should be it, and it should be self-contained.  You can refer back to the Experiments Section for elaborations. This section should be less than a page. In particular emphasize any results that were surprising.


## References
List the references that cited in your project.

## Appendix## 

Explain the contributions of each member to the project. Include all supporting materials, e.g., additional figures/tables, Python code technical derivations.
