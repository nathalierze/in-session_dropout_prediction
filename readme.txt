In-session dropout prediction model

This project describes an in-session prediction model that predicts student early dropout from online learning exercises.
Dropout prediction models for Massive Open Online Courses (MOOCs) have shown high accuracy rates in
the past and make personalized interventions possible. While MOOCs have traditionally high dropout rates,
school homework and assignments are supposed to be completed by all learners. In the pandemic, online
learning platforms were used to support school teaching. In this setting, dropout predictions have to be designed differently as a simple dropout from the (mandatory) class is not possible. The aim of our work is to
transfer traditional temporal dropout prediction models to in-session dropout prediction for school-supporting
learning platforms. For this purpose, we used data from more than 164,000 sessions by 52,000 users of the
online language learning platform orthografietrainer.net. We calculated time-progressive machine learning
models that predict dropout after each step (completed sentence) in the assignment using learning process
data. The multilayer perceptron is outperforming the baseline algorithms with up to 87% accuracy. By extending the binary prediction with dropout probabilities, we were able to design a personalized intervention
strategy that distinguishes between motivational and subject-specific interventions. 
A random state is not set, thus, results might differ marginally.

Whole project described in: 
N. Rzepka, K. Simbeck, H.-G. Müller, and N. Pinkwart
Keep It Up: In-session Dropout Prediction to Support Blended Classroom Scenarios
Proceedings of the 14th International Conference on Computer Supported Education - Volume 2: CSEDU,
SciTePress, 2022, ISBN 978-989-758-562-3 

How to use:
The data files to run this project can be assessed via Zenodo:
10.5281/zenodo.7746395

Data to run these files:
- dropout_prediction_data.pkl