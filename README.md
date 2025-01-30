# Training and Fine Tuning of ML models
## 1. Preface
A free tree is an undirected graph that represents the connections between the words in a sentence. This project is extremely useful because unsupervised models often fail to determine the direction of these links, resulting in this free tree. Once we know the root of the tree thanks to this model we can easily deduce the direction of these connections.
## 2. Description
The purpose of this project is to train multiple Machine Learning models to get the best one at finding the root of a sentence given a **free tree** based in a set of selected features. The features that have been tested and the ones that have been selected are in the following sections with their corresponding analysis in *section 4. Analysis of Features*.

The data that has been used to train these models has been extracted from [here](https://cqllab.upc.edu/lal/universal-dependencies/). It is given in multiple languages (to check them go to the section *3.1 List of Languages*), one file per language. The **features** extracted from this data have been analyzed to determine how relevant they are to the model's performance. It is an indispensable step, because the simpler the model the better if the performance is barely affected by the deletion of this feature.

Data has been divided into two parts: the training and the validation portions. The main reason is to avoid **overfitting** due to bias and variance, and this method is perfect for that. These portions have been modified using the **K-Fold Cross-Validation** method for k=10, which consists in saving a few portions that are not used for training but to evaluate how good the model is afterwards.

Different models have been tested to compare them and use the best one for this use case. You can find the list of models below in the section *5. Models*. For the training of each model, hyperparameters like alfa and others for each own model have been optimized. Before the training of each of the models feature data has been normalized.

The scoring methods used to evaluate the performance of the models are the **MSE** and the **R-squared** values, used to quantify the error rate and the accuracy rate of the models respectively.

## 3. Data used for Training and Feature extraction
The data used for training the different models can be found [here](https://cqllab.upc.edu/lal/universal-dependencies/). The data received are multiple rooted trees, one line per sentence. A part of this project concerns the transformation from a rooted tree to a free tree.
### 3.1 List of Languages
| | | | | |
| -------- | -------- | -------- | -------- | -------- |
| Arabic | French | Indonesian | Portuguese | Turkish |
| Chinese | Galician | Italian | Russian | |
| Czech | German | Japanese  | Spanish | |
| English | Hindi | Korean  | Swedish | |
| Finnish | Icelandic | Polish | Thai | |
### 3.2 Features Extracted
| Feature Name | Brief Description |
| -------- | -------- |
| vertex_degree | Number of vertices connected to a vertex. |
| avg_distance | Mean distance from the vertex to all the other vertices of the sentence. |
| is_center | If the vertex belongs to the center of the "graph" of the sentence. |
| is_centroidal | If the vertex belongs to the centroid of the "graph" of the sentence. |
| phrase_length | Length of the sentence to which the vertex belongs. |
| betweenness_centrality | Number of times a vertex happens to be in the shortest path between two other vertices. |
| closeness_centrality | Reciprocal of avg_distance. |
| eigenvector_centrality | Measure to quantify the influence of a vertex, also considers its neighbor's influence. |
| average_neighbor_degree | Mean of the degrees (vertex_degree) of the adjacent vertices of a vertex. |
| eccentricity | Maximum distance from one vertex to another in the graph. |
| diameter | Maximum value for eccentricity. |
| in_diameter | Indicates whether the vertex can be found within the diameter. |
| radius | Minimum value for eccentricity. |
| weiner_index | Sum of distances between all the pairs of vertices inside the graph. |

## 4. Analysis of Features
These plots have been created to determine the influence of each feature. Keep in mind that this mostly applies to **Linear Correlation**, so there may be other relations that we are not aware of and that we can not see using these plots.

Only the most relevant features are shown in this section, but for more information go to section *9.1 Tables and Plots for Descriptive and Distribution Analysis of the data*.

### 4.1 Descriptive and Distribution Analysis
The table to do the descriptive analysis of these features is in section 9.1. As we can see, the sentences are quite long as the average of the feature phrase length is 22.26 and the maximum is 70. As for the diameter the average is 7.69, meaning that words close to each other in the sentence are normally related. The feature betweenness centrality shows that vertices are not usually part of a shortest path between two other vertices, but when it does happen these vertices have a heavy weight in the decision of what the root of the tree is.

All the features show a persistent left tendency, which can be seen in the following plots and the ones found in section 9.1. It is the main reason why the normalization method that has been used is the robust normalizer.

![vertex_degree-boxplot](/images/vertex_degree-boxplot.png)
![vertex_degree-distribution](/images/vertex_degree-distribution.png)
![phrase_length-boxplot](/images/phrase_length-boxplot.png)
![phrase_length-distribution](/images/phrase_length-distribution.png)
![betweenness_centrality-boxplot](/images/betweenness_centrality-boxplot.png)
![betweenness_centrality-distribution](/images/betweenness_centrality-distribution.png)
![eigenvector_centrality-boxplot](/images/eigenvector_centrality-boxplot.png)
![eigenvector_centrality-distribution](/images/eigenvector_centrality-distribution.png)

Finally, the most important plot is the one for the target we are looking for as shown below. As we can see, it also has a left tendency and its distribution is not very homogeneous.

![target-boxplot](/images/target-boxplot.png)
![target-distribution](/images/target-distribution.png)

### 4.2 Linear Correlation of Features
In the following figure we can see that some features have very little linear correlation with others, so these are the first ones that have been deleted to improve the performance of the models.

The most important features at first glance seem to be betweenness centrality, eigenvector centrality and closeness centrality in the respective order. It is pretty accurate, as in the end we have kept two of these three most important features.

![correlation_matrix](/images/correlation_matrix.png)

Thanks to the following plot we can safely say that betweenness centrality has a big linear correlation with the target almost looking like a quadratic function.

![dimensionality_reduction](/images/dimensionality_reduction.png)

## 5. Models
These are the following models used for this project:
| | |
| -------- | -------- |
| Random | Elastic Net Regression
| Normal degree | Decision trees |
| Linear Regression | Random Forest |
| Ridge Regression | Gradient Boosting Machine |
| Lasso Regression | GMM |

The "models" Random and Normal degree are used as a baseline to determine the quality of the others. Random referes to a model that outputs a random number between 0 and 1. Normal degree refers to a model that normalizes the degree (*k*) of each vertex calculating *(k-1)/(n-1)*, being *n* the length of the sentence.

## 6. K-fold Cross-Validation and Data Normalization
We normalize the data from the features extracted that need normalization to avoid giving more importance to one feature than another solely for its value size. The chosen method is the **Robust Scaler** due to the fact that all the plots previously presented of the features' properties show a very persistent left tendency as mentioned previously.

## 7. Evaluation
As it has been mentioned in the description, the evaluation methods are the MSE and R-squared values.
| Scoring method | Brief description |
| -------- | -------- |
| MSE | Mean of all the squared differences between the model predictions and the correct expected results. The lower number for this value the better. |
| R-squared | Mean of all the R-squared values which measures the quality of the regression model. The higher number for this value the better. |

## 8. Results
### 8.1 Best model
The **Gradient Boosting Machine** is the one that has proven to be most effective for this case with these four hyperparameters being considered relevant: **vertex degree**, **phrase length**, **betweenness centrality** and **eigenvector centrality**. Other models were not too far from first place as shown here:
| Model | MSE | R-squared |
| -------- | -------- | -------- |
| Gradient Boosting Machine | 0.01873 | 0.72408 |
| Linear Regression | 0.02017 | 0.70292 |
| Ridge Regression | 0.02017 | 0.70292 |

For all models the best values for their own hyperparameters have been chosen, including the best value for alfa. Despite this, the models Lasso Regression, Elastic Net Regression and GMM fail to perform well on any situation with any combination of features, meaning that they are not fit for this use case.

After deleting all features except one it can be noticed that the most important one by far is the **betweenness centrality**. Go to table 2 in section *9.2 Tables of Results* to see the results for MSE and R-squared with using only the feature betweenness centrality.
### 8.2 Comparison of Models for different Languages
As mentioned before the data input refers to the rooted trees of sentences in different languages. From table 4 and 5 in section *9.2 Tables of Results* we get the following plots respectively:

![MSE_results](/images/MSE_results.png)

The first plot shows the MSE values for each language for each of the models. Remember that the lower the number the better, so in this case we can see that the best language these models are at is Japanese, which is interesting, since it is known to be a language quite difficult to learn. We get these results most probably because its syntax is quite simple leading to a much easier time to learn what the root of the sentence is.

![R2_results](/images/R2_results.png)

The second plot shows the R-squared values for each language for each of the models. Remember that the higher the number the better, so in this case we can see that only the Linear Regression, the Ridge Regression, the Decision Tree, the Random Forest and the Gradient Boosting Machine models have a positive value and thus are worth investigating. For this value it seems like once again Japanese is the best language that these models are at, just like with the MSE values. As mentioned before, it is probably due to its simple syntax. Chinese also has a simple syntax, reinforcing the previous hypothesis as Chinese is also one of the languages that these models are best at.

## 9. Annex
### 9.1 Tables and plots for Descriptive and Distribution Analysis of the data
#### Table 1.

Table that shows a data description for each feature.
| Measure/Feature | vertex_degree | avg_distance | is_center| is_centroidal | phrase_length | betweenness_centrality | closeness_centrality | eigenvector_centrality | average_neighbor_degree | eccentricity |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| Count | 371321 | 371321 | 371321 | 371321 | 371321 | 371321 | 371321 | 371321 | 371321 | 371321 | 371321 |
| Mean | 1.892815 | 3.691950 | 0.074765 | 0.074765 | 22.260737 | 0.146178 | 0.294866 | 0.185223 | 3.169671 | 6.173166 | 7.687349 | 0.581553 | 4.092378 | 1154.083453 |
| Standard Deviation | 1.296955 | 1.051328 | 0.263013 | 0.263013 | 8.934124 | 0.221502 | 0.092301 | 0.138870 | 1.324016 | 1.872693 | 2.058742 | 0.493305 | 1.065941 | 1224.120719 |
| Min | 1.0 | 1.0 | 0.0 | 0.0 | 3.0 | 0.0 | 0.094708 | 0.000042 | 1.0 | 1.0 |
| 25% | 1.0 | 2.944444 | 0.0 | 0.0 | 16.0 | 0.0 | 0.230769 | 0.082408 | 2.0 | 5.0 | 6.0 | 0.0 | 3.0 | 380.0 |
| 50% | 1.0 | 3.60 | 0.0 | 0.0 | 21.0 | 0.0 | 0.277778 | 0.152841 | 3.0 | 6.0 | 8.0 | 1.0 | 4.0 | 778.0 |
| 75% | 3.0 | 4.333333 | 0.0 | 0.0 | 28.0 | 0.235507 | 0.339623 | 0.241381 | 4.0 | 7.0 | 9.0 | 1.0 | 5.0 | 1490.0 |
| Max | 14.0 | 10.558824 | 1.0 | 1.0 | 70.0 | 1.0 | 1.0 | 0.707107 | 14.0 | 18.0 |18.0 |1.0 | 9.0 | 14966.0 |

#### Plots 1 & 2.
Boxplot and distribution plots for the feature avg_distance.

![avg_distance-boxplot](/images/avg_distance-boxplot.png)
![avg_distance-distribution](/images/avg_distance-distribution.png)

#### Plots 3 & 4.
Boxplot and distribution plots for the feature is_center.

![is_center-boxplot](/images/is_center-boxplot.png)
![is_center-distribution](/images/is_center-distribution.png)

#### Plots 5 & 6.
Boxplot and distribution plots for the feature is_centroidal.

![is_centroidal-boxplot](/images/is_centroidal-boxplot.png)
![is_centroidal-distribution](/images/is_centroidal-distribution.png)

#### Plots 7 & 8.
Boxplot and distribution plots for the feature closeness_centrality.

![closeness_centrality-boxplot](/images/closeness_centrality-boxplot.png)
![closeness_centrality-distribution](/images/closeness_centrality-distribution.png)

#### Plots 9 & 10.
Boxplot and distribution plots for the feature average_neighbor_degree.

![average_neighbor_degree-boxplot](/images/average_neighbor_degree-boxplot.png)
![average_neighbor_degree-distribution](/images/average_neighbor_degree-distribution.png)

#### Plots 11 & 12.
Boxplot and distribution plots for the feature eccentricity.

![eccentricity-boxplot](/images/eccentricity-boxplot.png)
![eccentricity-distribution](/images/eccentricity-distribution.png)

#### Plots 13 & 14.
Boxplot and distribution plots for the feature diameter.

![diameter-boxplot](/images/diameter-boxplot.png)
![diameter-distribution](/images/diameter-distribution.png)

#### Plots 15 & 16.
Boxplot and distribution plots for the feature in_diameter.

![in_diameter-boxplot](/images/in_diameter-boxplot.png)
![in_diameter-distribution](/images/in_diameter-distribution.png)

#### Plots 17 & 18.
Boxplot and distribution plots for the feature radius.

![radius-boxplot](/images/radius-boxplot.png)
![radius-distribution](/images/radius-distribution.png)

#### Plots 19 & 20.
Boxplot and distribution plots for the feature wiener_index.

![wiener_index-boxplot](/images/wiener_index-boxplot.png)
![wiener_index-distribution](/images/wiener_index-distribution.png)

### 9.2 Tables of Results
#### Table 1.
Results for using the four features vertex_degree, phrase_length, eigenvector and betweenness centrality with all languages as input data.
| Model | MSE | R-squared |
| -------- | -------- | -------- |
| Random | 0.285105422370349 | -3.20060265254352 |
| Normal degree | 5.18634950886863 | -75.4134422683945 |
| Linear Regression | 0.0201658979678067 | 0.702917218900158 |
| Ridge Regression | 0.0201658979672097 | 0.702917219011988 |
| Lasso Regression | 0.0678806703897772 | -0.0000147436483798779 |
| Elastic Net Regression | 0.0678806703897772 | -0.0000147436483798779 |
| Decision Tree | 0.0342242901324194 | 0.495803626639215 |
| Random Forest | 0.0224229087660543 | 0.669670546659429 |
| Gradient Boosting Machine | 0.0190093487013587 | 0.719960973566609
| GMM | 0.0859682726791761 | -0.266475292626924 |

#### Table 2.
Results for using only the feature betweenness centrality with all languages as input data. The model normal degree does not have a value because the feature vertex degree has not been selected.
| Model | MSE | R-squared |
| -------- | -------- | -------- |
| Random | 0.2843237930512497 | -3.1891087881428732 |
| Normal degree | - | - |
| Linear Regression | 0.0202809792052787 | 0.7012214301329455 |
| Ridge Regression | 0.020280979205731776 | 0.7012214302394947 |
| Lasso Regression | 0.06788067038977724 | -1.474364837987796e-05 |
| Elastic Net Regression | 0.06788067038977724 | -1.474364837987796e-05 |
| Decision Tree | 0.0201769848553593 | 0.7027557091184073 |
| Random Forest | 0.020006119966908216 | 0.7052721931142946 |
| Gradient Boosting Machine | 0.01963092111601815 | 0.7107976564435315 |
| GMM | 0.08596827267917613 | -0.2664752926269244 |

#### Table 3.
Results for using all the previously described features with all languages as input data. The random forest model does not have a value because it did not even finish the first fold within 3 hours.
| Model | MSE | R-squared |
| -------- | -------- | -------- |
| Random | 0.284882094822205 | -3.19733863530342 |
| Normal degree | 0.481941878181523 | -8.06081392231267 |
| Linear Regression | 0.0200146946282594 | 0.705144898427363 |
| Ridge Regression | 0.0200146945281936 | 0.705144899951571 |
| Lasso Regression | 0.0708827675005482 | -0.0000147436483798779 |
| Elastic Net Regression | 0.0678806703897772 | -0.000771276833173439 |
| Decision Tree | 0.0346982454858982 | 0.48878538067082 |
| Random Forest | - | - |
| Gradient Boosting Machine | 0.0187291471075987 | 0.724083492005656
| GMM | 0.0859682726791761 | -0.266475292626924 |

#### Table 4.
MSE results for using the four features vertex_degree, phrase_length, eigenvector and betweenness centrality for each of the languages.
| Language/Model	| Linear Regression	| Ridge Regression	| Lasso Regression	| Elastic Net Regression | Decision Tree	| Random Forest | Gradient Boosting Machine	| GMM |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| Arabic	| 0.023530201625890	| 0.023530200411524	| 0.070882767500548	| 0.070882767500548	| 0.041892520807356	| 0.025823790428371	| 0.024267148578674	| 0.091989009039923 |
| Chinese	| 0.017475723657304| 	0.017475721632455	| 0.064627450893589	| 0.064627450893589	| 0.032692279744135	| 0.019328221176870	| 0.018703343028974	| 0.081042325597481 |
| Czech	| 0.025574448197278	| 0.025574445833119	| 0.078177154985588	| 0.078177154985588	| 0.048093078733165	| 0.029779830043549	| 0.027950156894012	| 0.102286726935066 |
| English	| 0.019061243015400	| 0.019061243325417	| 0.067520685886209	| 0.067520685886209	| 0.036610787284008	| 0.021414126819886	| 0.021076577754712	| 0.084528933896009 |
| Finnish	| 0.024517010597128	| 0.024517011920720	| 0.083739351279073	| 0.083739351279073	| 0.044088388554561	| 0.028335792757142	| 0.027814677548983	| 0.110357044549104 |
| French	| 0.016501784295316	| 0.016501782976113	| 0.059571556814500	| 0.059571556814500	| 0.029866842065063	| 0.018137027910744	| 0.017609673332394	| 0.073301612864846 |
| Galician	| 0.019777566798655	| 0.019777565410495	| 0.063624767801684	| 0.063624767801684	| 0.035765205450456	| 0.021304230519486	| 0.020838523700073	| 0.079409123138615
| German	| 0.018239866311027	| 0.018239866232157	| 0.066861106723244	| 0.066861106723244	| 0.033758375540433	| 0.020233827558412	| 0.020209476403394	| 0.083123471884610 |
| Hindi	| 0.016713159607737	| 0.016713157034977	| 0.060505032821676	| 0.060505032821676	| 0.033064646527558	| 0.016713157034977	| 0.019113629368141	| 0.074282706005423 |
| Icelandic	| 0.021027155921022	| 0.021027152733755	| 0.073250571025165	| 0.073250571025165	| 0.038514529837016	| 0.023853714467744	| 0.023705403094476	| 0.093705268135159 |
| Indonesian	| 0.021330666147686	| 0.021330669463480	| 0.072454646672234	| 0.072454646672234	| 0.037335159052457	| 0.021330669463480	| 0.021824949736656	| 0.093466674979132
| Italian	| 0.018545206593930	| 0.018545205266345	| 0.062243298973675	| 0.062243298973675	| 0.034582007054110	| 0.020249406546336	| 0.019569865023368	| 0.077042968838995 |
| Japanese	| 0.013471720644726	| 0.013471720528667	| 0.052215240250456	| 0.052215240250456	| 0.023226176732437	| 0.013861670142528	| 0.013762476401944	| 0.062966194328831 |
| Korean	| 0.027020784531987	| 0.027020779520219	| 0.079481462845313	| 0.079481462845313	| 0.050074508490094	| 0.031548695948378	| 0.028994179149125	| 0.110736402898037 |
| Polish	| 0.025205430469470	| 0.025205428022444	| 0.077425257940242	| 0.077425257940242	| 0.047293626226512	| 0.029048296050948	| 0.027488949420920	| 0.101913891865650 |
| Portuguese	| 0.018536753873033	| 0.018536753677286	| 0.062928834342612	| 0.062928834342612	| 0.034291502520805	| 0.020814913227607	| 0.020136315358895	| 0.078312086849974 |
| Russian	| 0.023056755870021	| 0.023056755548817	| 0.074674921973112	| 0.074674921973112	| 0.043210964100487	| 0.026428132295294	| 0.024918838796479	| 0.097923184051334 |
| Spanish	| 0.019327903967912	| 0.019327902426048	| 0.063181759695345	| 0.063181759695345	| 0.035260489937938	| 0.021427567761218	| 0.020477824698236	| 0.078690048837189 |
| Swedish	| 0.020319383955141	| 0.020319381632665	| 0.071750172750841	| 0.071750172750841	| 0.037923665751459	| 0.023933493288610	| 0.023873935849547	| 0.090423077401137 |
| Thai	| 0.017851796897786	| 0.017851796603027	| 0.061178759436948	| 0.061178759436948	| 0.033053661037837	| 0.019769775458511	| 0.018596636636208	| 0.076548764958193 |
| Turkish	| 0.024909582654976	| 0.024909579083363	| 0.079029734724098	| 0.079029734724098	| 0.044595721160219	| 0.029325854753074	| 0.027701259404730	| 0.105722681385448 |
| Total	| 0.020165897967807	| 0.020165897967210	| 0.067880670389777	| 0.067880670389777	| 0.034224290132419	| 0.022422908766054	| 0.019009348701359	| 0.085968272679176 |

#### Table 5.
R-squared results for using the four features vertex_degree, phrase_length, eigenvector and betweenness centrality for each of the languages.
| Language/Model	| Linear Regression	| Ridge Regression	| Lasso Regression	| Elastic Net Regression | Decision Tree	| Random Forest | Gradient Boosting Machine	| GMM |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
Arabic	| 0.668231405361397	| 0.668231553988513	| -0.000771276833173	| -0.000771276833173	| 0.408498740198355	| 0.635472709088694	| 0.657495249031473	| -0.298280240990090
Chinese	| 0.729450626761948	| 0.729450792033005	| -0.000481425281323	| -0.000481425281323	| 0.492598198354606	| 0.700138548751708	| 0.709899243881501	| -0.254385102310142
Czech	| 0.672926694028515	| 0.672926725959813	| -0.000422420328038	| -0.000422420328038	| 0.384733511274243	| 0.619044261243735	| 0.642386933184047	| -0.308888908665246
English	| 0.707321592620632	| 0.707321638492237	| -0.000747441490780	| -0.000747441490780	| 0.471973954269712	| 0.661415598080704	| 0.667767550034253	| -0.318390522188571
Finnish	| 0.722742078711091	| 0.722742114127700	| -0.000323551055634	| -0.000323551055634	| 0.498224865670929	| 0.695424901766864	| 0.704141031411308	| -0.230717074076135
French	| 0.722742078711091	| 0.722742114127700	| -0.000323551055634	| -0.000323551055634	| 0.498224865670929	| 0.695424901766864	| 0.704141031411308	| -0.230717074076135
Galician	| 0.689017088376618	| 0.689017115553612	| -0.000955116302744	| -0.000955116302744	| 0.436460683582461	| 0.664672804756430	| 0.672054409119377	| -0.248586631624224
German	| 0.726762949392607	| 0.726763127401463	| -0.000426895077754	| -0.000426895077754	| 0.494358106753184	| 0.696789484663698	| 0.696820302434507	| -0.243779123542593
Hindi	| 0.723863522701020	| 0.723863620630227	| -0.000354968846125	| -0.000354968846125	| 0.452858865457517	| 0.723863620630227	| 0.684210535230182	| -0.227921967057732
Icelandic	| 0.712799159906727	| 0.712799306429452	| -0.000454823234870	| -0.000454823234870	| 0.474302510003958	| 0.674334796238256	| 0.676074232566568	| -0.279527905051973
Indonesian	| 0.705532771218490	| 0.705532819217927	| -0.000503091068536	| -0.000503091068536	| 0.484159680284077	| 0.705532819217927	| 0.698797999828881	| -0.290351155978656
Italian	| 0.702163239714396	| 0.702163320635012	| -0.000715681561900	| -0.000715681561900	| 0.443455091463391	| 0.674504473208440	| 0.685265206124121	| -0.238154963649812
Japanese	| 0.741624222059655	| 0.741624244176905	| -0.000323146729331	| -0.000323146729331	| 0.554939132079185	| 0.734211453781428	| 0.736082944031855	| -0.206097588914639
Korean	| 0.659720028186403	| 0.659720246051809	| -0.000366410957912	| -0.000366410957912	| 0.370054953654056	| 0.602683504555444	| 0.634749855740814	| -0.393773699591188
Polish	| 0.674140819554199	| 0.674140911822583	| -0.000222166202581	| -0.000222166202581	| 0.388374590240733	| 0.624146782547441	| 0.644403494936484	| -0.316518764727151
Portuguese	| 0.705097111166362	| 0.705097104110269	| -0.000435739476198	| -0.000435739476198	| 0.454290258083305	| 0.668778289588656	| 0.679813122652100	| -0.244782536482362
Russian	| 0.691565051927168	| 0.691565088526099	| -0.000404248385223	| -0.000404248385223	| 0.420662726909824	| 0.646082674198140	| 0.666504251285823	| -0.311660111791061
Spanish	| 0.694111902483234	| 0.694112072526492	| -0.001219560969910	| -0.001219560969910	| 0.440972603028769	| 0.660161123899398	| 0.675526734865881	| -0.311660111791061
Swedish	| 0.716661741916394	| 0.716661885841392	| -0.000743277962007	| -0.000743277962007	| 0.470179954007153	| 0.665842539619768	| 0.666851601647136	| -0.260698882611414
Thai	| 0.708254057149609	| 0.708254101559506	| -0.000745693805230	| -0.000745693805230	| 0.458682822166127	| 0.676617632575237	| 0.695856527406451	| -0.251702488646375
Turkish	| 0.684416825406463	| 0.684417069460645	| -0.000579268668109	| -0.000579268668109	| 0.433369178494998	| 0.628173484750927	| 0.648963384818978	| -0.338294458657109
Total	| 0.702917218900158	| 0.702917219011988	| -0.000014743648380	| -0.000014743648380	| 0.495803626639215	| 0.669670546659429	| 0.719960973566609	| -0.266475292626924