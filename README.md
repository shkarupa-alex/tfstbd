# tfstbd

Sentence & Token boundary detector with deep learning and TensorFlow.
This is a model development part of project.


## Training custom model

1. Obtain dataset with separated sentences and tokens in CoNNL-U format.

    UniversalDependencies is a good choice. But maybe you have more data?
    Copy your *.conllu files (or just train part) to `data/source/` folder.

2. Convert *.conllu files into trainer-accepted dataset format (with auto-augmentation).

    ```bash
    tfstbd-dataset data/source/ data/dataset/
    ```
   For multiple languages:
    ```bash
    tfstbd-dataset data/source/ru data/source/en data/dataset/
    ```

3. Check dataset for correctness and estimate some useful hyperparams.

    ```bash
    tfstbd-check data/dataset/
    ```

4. Prepare configuration file with hyperparameters. Start from `config/default.json` in this module repository.

5. Extract most frequent char ngrams vocabulary from train dataset.  This will include "<start" and "end>" ngrams too.

    ```bash
    tfstbd-vocab config/default.json data/dataset/
    ```

6. Run training.

    First run will only compute start metrics, so you should run repeat this step at least twice.
    ```bash
    tfstbd-train config/default.json data/dataset/ model/
    ```

    Optionally use `--eval_data data/ready_eval/` to evaluate model and `--export_path export/` to export.

7. Test your model on plain text file.
    ```bash
    tfstbd-infer export/<model_version> some_text_document.txt
    ```


## No training
{'accuracy': 0.96256346, 'accuracy_baseline': 0.96256346, 'auc': 0.5837398, 'auc_precision_recall': 0.07934569, 'average_loss': 0.30893928, 'label/mean': 0.03743653, 'loss': 4676.206, 'precision': 0.0, 'prediction/mean': 0.23343459, 'recall': 0.0, 'global_step': 1, 'f1': 0.0}



https://github.com/Koziev/rutokenizer


аугментации:
 - замена букв на цифры (при4еска, пр1вет)
 - замена букв на символы (зах**чила, жоп@)
 - замена ! на 1
 - заменять смайла на эмотиконы и обратно https://github.com/NeelShah18/emot/blob/master/emot/emo_unicode.py
 ?- вставка - в произвольное место
 ?- замена запятых или произвольная вставка многоточия (многоточие считается концом предложения как и .. и …)
 - отрывать точку от сокращения (это все еще не конец предложения)
 - добавлять пробел до знака .!?
 - добавлять ссылки в произвольные места


