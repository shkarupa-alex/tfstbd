TODO: LAMB optimizer
TODO: short sentencies
У С Т А Н О В И Л:
Критика

# tfstbd

Sentence&Token boundary detector implemented with TensorfFlow.
This is a model development part of project.


## Training custom model

1. Obtain dataset with already splitted sentences and tokens in CoNNL-U format.

UniversalDependencies is a good choice. But maybe you have more data?
Copy your *.conllu files (or just train part) to "data/prepare/" folder.

2. Convert *.conllu files (with auto-augmentation) into trainer-accepted dataset format.
```bash
tfkstbd-dataset data/prepare/ data/ready/
```

3. Prepare configuration file with hyperparameters. Start from config/default.json in this module repository.


4. Extract most frequent non-alphanum ngrams vocabulary from train dataset. This will include "<start" and "end>" ngrams too.
```bash
tfkstbd-vocab data/ready/ config/default.json data/vocabulary.pkl
```

5. Run training.

First run will only compute start metrics, so you should run repeat this stem multiple times.
```bash
tfkstbd-train data/ready/ data/ready/vocabulary.pkl config/default.json model/
```
Optionally use `--eval_data data/ready_eval/` to evaluate model and `--export_path export/` to export.
You can also provide `--threads_count NN` flag if you have a lot (>8) of CPU cores.

6. Test your model on plain text file.
```bash
tfkstbd-infer export/<model_version> some_text_document.txt
```


## No training
{'accuracy': 0.96256346, 'accuracy_baseline': 0.96256346, 'auc': 0.5837398, 'auc_precision_recall': 0.07934569, 'average_loss': 0.30893928, 'label/mean': 0.03743653, 'loss': 4676.206, 'precision': 0.0, 'prediction/mean': 0.23343459, 'recall': 0.0, 'global_step': 1, 'f1': 0.0}

TODO: urldecode, entities?
г/кВт∙ч.
тонн/ТВт∙ч)
    КП. АМ.

TODO:
focal loss



https://github.com/Koziev/rutokenizer