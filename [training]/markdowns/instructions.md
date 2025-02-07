#### This pipeline is used for loading the diamonds dataset from opendatasets (kaggle API) </br> </br>
###### - It'll perform data cleaning, saving cleaned data and model training.</br></br>
###### - You can choose if you'd like to save the cleaned dataset to postgres database by specifying the value of the global variable *export_data* it has either yes, no by default it's yes.</br></br>
###### - You can choose which models to train by providing them in the global variable *selected_models* in this format ["dt","knn"]. </br></br>
