# Car_Prediction

The process of the project is the following:
1.  To create the data the first time, it is necessary to have 
the subfolder raw_data inside the folder data.
2.  Secondly in the train main setting the parameters: "labels_prepare"=True and "splitting_data"=True.
3.  "Labels_prepare" should be True only the first time the model is run.

Il main ha un parametro chiamato "origin_data_path" che è il percorso da cui prende le immagini
per fare lo split. Per prendere i dati del dataset inserire 'data/labels/all_labels.csv'.
Per prendere quelli nuovi settarlo a 'data/labels/all_labels_new'.

Per fare previsione su dati nuovi:
1. Eseguire il train_main e come parametro _origin_data_path_ 
 il percorso: "data/labels/all_labels_new".

Per fare previsione su foto di Stanford tagliate:
1. Eseguire il train_main e mettere come parametro _get_cropped_data_stanford_ 
 True.

