Experiment # 1

Baseline pretrianed model:
Rotation Only Data Augmentation

Classification Report:
                             precision    recall  f1-score   support

     acacia_auriculiformis       0.88      0.78      0.82         9
            acacia_mangium       1.00      0.33      0.50         9
            acer_saccharum       1.00      1.00      1.00         9
dipterocarpus_grandiflorus       0.57      0.89      0.70         9
      endospermum_peltatum       0.67      0.67      0.67         9
  eucalyptus_camaldulensis       0.26      0.67      0.38         9
        falcataria_falcata       0.57      0.44      0.50         9
        fraxinus_americana       1.00      0.33      0.50         9
           gmelina_arborea       1.00      0.33      0.50         9
        hevea_brasiliensis       0.22      0.22      0.22         9
             instia_bijuga       1.00      0.33      0.50         9
             juglans_nigra       0.82      1.00      0.90         9
             juglans_regia       1.00      0.78      0.88         9
     leucaena_leucocephala       0.86      0.67      0.75         9
                 palosapis       0.83      0.56      0.67         9
     parashorea_malaanonan       1.00      1.00      1.00         9
           pometia_pinnata       0.60      0.33      0.43         9
       pterocarpus_indicus       0.82      1.00      0.90         9
             quercus_robur       1.00      0.22      0.36         9
             samanea_saman       1.00      0.38      0.55         8
             shorea_albida       1.00      1.00      1.00         9
           shorea_astylosa       1.00      0.33      0.50         9
           shorea_contorta       1.00      0.33      0.50         9
        shorea_negrosensis       0.82      1.00      0.90         9
              shorea_ovata       0.31      0.56      0.40         9
          shorea_palosapis       0.00      0.00      0.00         2
         shorea_parvifolia       1.00      1.00      1.00         9
         shorea_polysperma       0.60      0.33      0.43         9
     swietenia_macrophylla       1.00      0.44      0.62         9
           tectona_grandis       0.24      0.89      0.37         9
          vitex_parviflora       0.16      0.44      0.24         9

                  accuracy                           0.61       271
                 macro avg       0.75      0.59      0.60       271
              weighted avg       0.77      0.61      0.62       271

Experiment # 2:

Data Augmentation:

- Rotation
- With Jitter and Random Erasing

Classification Report:
                             precision    recall  f1-score   support

     acacia_auriculiformis       0.90      1.00      0.95         9
            acacia_mangium       0.70      0.78      0.74         9
            acer_saccharum       1.00      1.00      1.00         9
dipterocarpus_grandiflorus       1.00      1.00      1.00         9
      endospermum_peltatum       0.36      1.00      0.53         9
  eucalyptus_camaldulensis       0.78      0.78      0.78         9
        falcataria_falcata       0.78      0.78      0.78         9
        fraxinus_americana       1.00      0.44      0.62         9
           gmelina_arborea       1.00      0.33      0.50         9
        hevea_brasiliensis       1.00      0.56      0.71         9
             instia_bijuga       0.71      0.56      0.63         9
             juglans_nigra       1.00      1.00      1.00         9
             juglans_regia       1.00      1.00      1.00         9
     leucaena_leucocephala       1.00      0.56      0.71         9
                 palosapis       1.00      0.33      0.50         9
     parashorea_malaanonan       1.00      1.00      1.00         9
           pometia_pinnata       1.00      0.33      0.50         9
       pterocarpus_indicus       0.90      1.00      0.95         9
             quercus_robur       1.00      0.44      0.62         9
             samanea_saman       1.00      0.38      0.55         8
             shorea_albida       1.00      1.00      1.00         9
           shorea_astylosa       0.75      0.33      0.46         9
           shorea_contorta       1.00      0.33      0.50         9
        shorea_negrosensis       1.00      1.00      1.00         9
              shorea_ovata       0.58      0.78      0.67         9
          shorea_palosapis       1.00      1.00      1.00         2
         shorea_parvifolia       1.00      1.00      1.00         9
         shorea_polysperma       0.27      0.33      0.30         9
     swietenia_macrophylla       1.00      0.33      0.50         9
           tectona_grandis       0.23      1.00      0.38         9
          vitex_parviflora       0.47      0.78      0.58         9

                  accuracy                           0.71       271
                 macro avg       0.85      0.71      0.72       271
              weighted avg       0.85      0.71      0.72       271

Experiment # 2:

Data Augmentation:

- Rotation
- With Jitter and Random Erasing
- additional Mobile data augmentation

Classification Report:
                             precision    recall  f1-score   support

     acacia_auriculiformis       0.75      1.00      0.86         9
            acacia_mangium       0.50      1.00      0.67         9
            acer_saccharum       1.00      1.00      1.00         9
dipterocarpus_grandiflorus       1.00      1.00      1.00         9
      endospermum_peltatum       1.00      1.00      1.00         9
  eucalyptus_camaldulensis       0.67      0.67      0.67         9
        falcataria_falcata       1.00      0.78      0.88         9
        fraxinus_americana       1.00      1.00      1.00         9
           gmelina_arborea       0.75      1.00      0.86         9
        hevea_brasiliensis       1.00      0.56      0.71         9
             instia_bijuga       1.00      0.33      0.50         9
             juglans_nigra       1.00      1.00      1.00         9
             juglans_regia       1.00      1.00      1.00         9
     leucaena_leucocephala       0.75      0.33      0.46         9
                 palosapis       1.00      0.78      0.88         9
     parashorea_malaanonan       0.90      1.00      0.95         9
           pometia_pinnata       1.00      0.33      0.50         9
       pterocarpus_indicus       1.00      1.00      1.00         9
             quercus_robur       1.00      1.00      1.00         9
             samanea_saman       0.53      1.00      0.70         8
             shorea_albida       1.00      1.00      1.00         9
           shorea_astylosa       0.64      1.00      0.78         9
           shorea_contorta       1.00      1.00      1.00         9
        shorea_negrosensis       1.00      1.00      1.00         9
              shorea_ovata       1.00      0.33      0.50         9
          shorea_palosapis       1.00      1.00      1.00         2
         shorea_parvifolia       1.00      1.00      1.00         9
         shorea_polysperma       0.62      0.56      0.59         9
     swietenia_macrophylla       0.60      1.00      0.75         9
           tectona_grandis       1.00      1.00      1.00         9
          vitex_parviflora       1.00      0.78      0.88         9

                  accuracy                           0.85       271
                 macro avg       0.89      0.85      0.84       271
              weighted avg       0.89      0.85      0.84       271

Classification Report:
                             precision    recall  f1-score   support

     acacia_auriculiformis       0.82      1.00      0.90         9
            acacia_mangium       0.60      1.00      0.75         9
            acer_saccharum       0.82      1.00      0.90         9
dipterocarpus_grandiflorus       1.00      1.00      1.00         9
      endospermum_peltatum       1.00      1.00      1.00         9
  eucalyptus_camaldulensis       0.82      1.00      0.90         9
        falcataria_falcata       1.00      0.78      0.88         9
        fraxinus_americana       0.90      1.00      0.95         9
           gmelina_arborea       0.90      1.00      0.95         9
        hevea_brasiliensis       1.00      0.56      0.71         9
             instia_bijuga       0.75      0.33      0.46         9
             juglans_nigra       1.00      1.00      1.00         9
             juglans_regia       1.00      1.00      1.00         9
     leucaena_leucocephala       0.50      0.33      0.40         9
                 palosapis       1.00      0.56      0.71         9
     parashorea_malaanonan       1.00      1.00      1.00         9
           pometia_pinnata       0.75      0.33      0.46         9
       pterocarpus_indicus       1.00      1.00      1.00         9
             quercus_robur       0.82      1.00      0.90         9
             samanea_saman       0.57      1.00      0.73         8
             shorea_albida       1.00      1.00      1.00         9
           shorea_astylosa       0.78      0.78      0.78         9
           shorea_contorta       1.00      1.00      1.00         9
        shorea_negrosensis       1.00      1.00      1.00         9
              shorea_ovata       1.00      0.33      0.50         9
          shorea_palosapis       1.00      1.00      1.00         2
         shorea_parvifolia       1.00      1.00      1.00         9
         shorea_polysperma       0.71      0.56      0.63         9
     swietenia_macrophylla       0.90      1.00      0.95         9
           tectona_grandis       0.50      1.00      0.67         9
          vitex_parviflora       1.00      0.89      0.94         9

                  accuracy                           0.85       271
                 macro avg       0.88      0.85      0.84       271
              weighted avg       0.87      0.85      0.84       271

Experiment # 2:

- Rotation
- With Jitter and Random Erasing
- additional Mobile data augmentation
- Additional data on poorly classified images

Classification Report:
                             precision    recall  f1-score   support

     acacia_auriculiformis       0.53      1.00      0.69         9
            acacia_mangium       0.82      1.00      0.90         9
            acer_saccharum       1.00      1.00      1.00         9
dipterocarpus_grandiflorus       1.00      1.00      1.00         9
      endospermum_peltatum       1.00      0.67      0.80         9
  eucalyptus_camaldulensis       0.69      1.00      0.82         9
        falcataria_falcata       0.75      0.67      0.71         9
        fraxinus_americana       1.00      0.89      0.94         9
           gmelina_arborea       0.90      1.00      0.95         9
        hevea_brasiliensis       1.00      0.56      0.71         9
             instia_bijuga       1.00      0.33      0.50         9
             juglans_nigra       1.00      1.00      1.00         9
             juglans_regia       1.00      1.00      1.00         9
     leucaena_leucocephala       1.00      0.56      0.71         9
                 palosapis       0.89      0.89      0.89         9
     parashorea_malaanonan       1.00      1.00      1.00         9
           pometia_pinnata       1.00      0.33      0.50         9
       pterocarpus_indicus       0.75      1.00      0.86         9
             quercus_robur       1.00      1.00      1.00         9
             samanea_saman       0.62      1.00      0.76         8
             shorea_albida       1.00      1.00      1.00         9
           shorea_astylosa       0.90      1.00      0.95         9
           shorea_contorta       0.82      1.00      0.90         9
        shorea_negrosensis       1.00      1.00      1.00         9
              shorea_ovata       1.00      0.78      0.88         9
          shorea_palosapis       1.00      1.00      1.00         2
         shorea_parvifolia       1.00      1.00      1.00         9
         shorea_polysperma       0.57      0.44      0.50         9
     swietenia_macrophylla       1.00      0.89      0.94         9
           tectona_grandis       0.82      1.00      0.90         9
          vitex_parviflora       0.82      1.00      0.90         9

                  accuracy                           0.87       271
                 macro avg       0.90      0.87      0.86       271
              weighted avg       0.90      0.87      0.86       271

Experiment # 3:

Data Augmentation:

- Rotation
- With Jitter and Random Erasing
- additional Mobile data augmentation
- Additional data augmentation for the class with less samples
- Addition of spatial transformer localization network

                             precision    recall  f1-score   support

     acacia_auriculiformis       0.89      0.89      0.89         9
            acacia_mangium       0.90      1.00      0.95         9
            acer_saccharum       1.00      1.00      1.00         9
dipterocarpus_grandiflorus       1.00      1.00      1.00         9
      endospermum_peltatum       1.00      0.67      0.80         9
  eucalyptus_camaldulensis       0.73      0.89      0.80         9
        falcataria_falcata       0.83      0.56      0.67         9
        fraxinus_americana       1.00      1.00      1.00         9
           gmelina_arborea       0.82      1.00      0.90         9
        hevea_brasiliensis       0.83      0.56      0.67         9
             instia_bijuga       1.00      0.33      0.50         9
             juglans_nigra       1.00      1.00      1.00         9
             juglans_regia       1.00      1.00      1.00         9
     leucaena_leucocephala       1.00      0.44      0.62         9
                 palosapis       1.00      0.67      0.80         9
     parashorea_malaanonan       1.00      1.00      1.00         9
           pometia_pinnata       1.00      0.78      0.88         9
       pterocarpus_indicus       0.90      1.00      0.95         9
             quercus_robur       1.00      1.00      1.00         9
             samanea_saman       0.50      1.00      0.67         8
             shorea_albida       1.00      1.00      1.00         9
           shorea_astylosa       0.60      1.00      0.75         9
           shorea_contorta       0.80      0.89      0.84         9
        shorea_negrosensis       1.00      1.00      1.00         9
              shorea_ovata       1.00      0.89      0.94         9
          shorea_palosapis       1.00      1.00      1.00         2
         shorea_parvifolia       1.00      1.00      1.00         9
         shorea_polysperma       0.62      0.56      0.59         9
     swietenia_macrophylla       0.90      1.00      0.95         9
           tectona_grandis       0.75      1.00      0.86         9
          vitex_parviflora       0.82      1.00      0.90         9

                  accuracy                           0.87       271
                 macro avg       0.90      0.87      0.87       271
              weighted avg       0.90      0.87      0.87       271

Confusion Matrix:
 [[8 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 8 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0]
 [0 0 0 0 0 0 0 9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 8 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9 0 0 0 0]
 [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 3 0 0 0 4 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9]]

Classification Report:
                             precision    recall  f1-score   support

     acacia_auriculiformis       1.00      0.89      0.94         9
            acacia_mangium       0.82      1.00      0.90         9
            acer_saccharum       1.00      1.00      1.00         9
dipterocarpus_grandiflorus       1.00      1.00      1.00         9
      endospermum_peltatum       1.00      0.89      0.94         9
  eucalyptus_camaldulensis       1.00      1.00      1.00         9
        falcataria_falcata       1.00      0.78      0.88         9
        fraxinus_americana       1.00      1.00      1.00         9
           gmelina_arborea       1.00      1.00      1.00         9
        hevea_brasiliensis       0.90      1.00      0.95         9
             instia_bijuga       1.00      1.00      1.00         9
             juglans_nigra       1.00      1.00      1.00         9
             juglans_regia       1.00      1.00      1.00         9
     leucaena_leucocephala       1.00      0.89      0.94         9
                 palosapis       1.00      1.00      1.00         9
     parashorea_malaanonan       0.90      1.00      0.95         9
           pometia_pinnata       1.00      1.00      1.00         9
       pterocarpus_indicus       1.00      1.00      1.00         9
             quercus_robur       1.00      1.00      1.00         9
             samanea_saman       0.89      1.00      0.94         8
             shorea_albida       1.00      1.00      1.00         9
           shorea_astylosa       1.00      1.00      1.00         9
           shorea_contorta       1.00      1.00      1.00         9
        shorea_negrosensis       0.69      1.00      0.82         9
              shorea_ovata       1.00      1.00      1.00         9
          shorea_palosapis       1.00      1.00      1.00         2
         shorea_parvifolia       1.00      1.00      1.00         9
         shorea_polysperma       0.80      0.44      0.57         9
     swietenia_macrophylla       1.00      1.00      1.00         9
           tectona_grandis       1.00      1.00      1.00         9
          vitex_parviflora       1.00      1.00      1.00         9

                  accuracy                           0.96       271
                 macro avg       0.97      0.96      0.96       271
              weighted avg       0.97      0.96      0.96       271