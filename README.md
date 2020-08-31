# tfstbd

Sentence & Token boundary detector with deep learning and TensorFlow.
This is a model development part of project.


## Training custom model

1. Optional. Obtain additional datasets for sentence-only boundary detection.

    Sentence boundaries are more rare then token ones.
    You can boost your model with additional data where token boundaries are not matter (will have zero weight in loss and metrics estimation).

    To prepare such dataset you need a text files with sentences.

    If sentences order matters they should be separated by single `\n`. This is called "paragraph". Keep paragraphs short.
    Paragraphs (including single-sentence ones) should be separated with at least double `\n`.

    Here is a simple dataset example:
    ```
    Single sentence.
    
    First sentence in paragraph.
    Second sentence in paragraph.
    And finally third one!
    
    One more single sentence in paragraph.
    ```
   
    If you need to preserve `\n` inside the sentence, replace it with character `␊`.
    ```
    Single sentence␊with newline.
    ``` 

    Install UDPipe and download model for your language. Or train a new one, e.g.:
    ```bash
    udpipe --train data/ud/udpipe.model --tokenizer="allow_spaces=0;batch_size=256;dimension=64;learning_rate=0.02;segment_size=256;epochs=50" --parser=none --heldout=data/ud/test.conllu data/ud/train.conllu
    ```

    Install UDPipe binding for python:
    ```bash
    pip install -U ufal.udpipe
    ```

    Then convert sentences and paragraphs into CoNLL-U format.
    ```bash
    tfstbd-convert data/ud/udpipe.model data/sent/doc.txt data/source/
    ```

2. Obtain dataset with already separated sentences and tokens in CoNNL-U format.

    UniversalDependencies is a good choice. But maybe you have more data?
    Copy your *.conllu files (or just train part) to `data/source/` folder.

3. Convert *.conllu files into trainer-accepted dataset format (with auto-augmentation).

    ```bash
    tfstbd-dataset data/source/ data/dataset/
    ```

4. Check dataset for correctness and estimate some useful hyperparams.

    ```bash
    tfstbd-check data/dataset/
    ```

5. Prepare configuration file with hyperparameters. Start from `config/default.json` in this module repository.

6. Extract most frequent char ngrams vocabulary from train dataset.  This will include "<start" and "end>" ngrams too.

    ```bash
    tfstbd-vocab config/default.json data/dataset/
    ```

7. Run training.

    First run will only compute start metrics, so you should run repeat this step at least twice.
    ```bash
    tfstbd-train config/default.json data/dataset/ model/
    ```

    Optionally use `--eval_data data/ready_eval/` to evaluate model and `--export_path export/` to export.

8. Test your model on plain text file.
    ```bash
    tfstbd-infer export/<model_version> some_text_document.txt
    ```


## No training
{'accuracy': 0.96256346, 'accuracy_baseline': 0.96256346, 'auc': 0.5837398, 'auc_precision_recall': 0.07934569, 'average_loss': 0.30893928, 'label/mean': 0.03743653, 'loss': 4676.206, 'precision': 0.0, 'prediction/mean': 0.23343459, 'recall': 0.0, 'global_step': 1, 'f1': 0.0}



https://github.com/Koziev/rutokenizer


TODO:
dividers = '-./:_\'’%*−+=#&@`—―–·×′\\'
repeaters = '.-)!?*/(":^+>,\'\\=—'
wrappers = '()<>[]{}**--__++~~'
