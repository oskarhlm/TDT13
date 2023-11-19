# TDT13 Project - Oskar Holm (F2023)

This project is based on the shared task related to Social Media Geolocation (SMG) from VarDial 2020 and 2021, specifically the Workshop on Natural Language Processing (NLP) for Similar Languages, Varieties, and Dialects. Unlike typical VarDial tasks that involve choosing from a set of variety labels, this task focuses on predicting the latitude and longitude from which a social media post was made.

The task remained the same in both 2020 and 2021, covering three language areas: Bosnian-Croatian-Montenegrin-Serbian, German (Germany and Austria), and German-speaking Switzerland. This project is limited to the German-speaking Switzerland area due to time constraints and resource availability.

The goal of the project is to replicate the results of a study that used a BERT-based classifier for this double regression task. The dataset from the 2020 VarDial challenge is chosen because it had more submissions compared to the 2021 dataset. 

**train.ipynb** is where most of the action takes place. The **lib/** folder includes helper functions used in **train.ipynb**.   