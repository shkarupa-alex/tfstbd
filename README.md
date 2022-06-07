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
    tfstbd-vocab config/default.yaml data/dataset/
    ```

6. Run training.

    First run will only compute start metrics, so you should run repeat this step at least twice.
    ```bash
    tfstbd-train config/default.yaml data/dataset/ model/
    ```

    Optionally use `--eval_data data/ready_eval/` to evaluate model and `--export_path export/` to export.

7. Test your model on plain text file.
    ```bash
    tfstbd-infer export/<model_version> some_text_document.txt
    ```


TODO:
если начинается и заканчивается на маленькую букву - делать с большой и заканчивать точкой

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


# TODO
# Удалять все пробелы которые не употребляются в одиночку + u200b
# sure_word = {'\u200e', '\u200f', '\ufffd'}
# sure_spaces = {'\t', '\n', '\x0b', '\x0c', '\r', '\x1c', '\x1d', '\x1e', '\x1f', ' ', '\x85', '\xa0', '\u1680', '\u2000', '\u2001', '\u2002', '\u2003', '\u2004', '\u2005', '\u2006', '\u2007', '\u2008', '\u2009', '\u200a', '\u2028', '\u2029', '\u200b', '\u202f', '\u205f', '\u2060', '\u2061', '\u2800', '\u3000', '\ufeff'}
# 00ad - удалять?
# fffd - удалять
# feff - заменять на 2060


tfstbd-dataset data/conllu/russian data/dataset_russ

tfstbd-dataset data/conllu/russian data/conllu/english data/dataset_ruen

tfstbd-dataset data/conllu/multilang/afrikaans data/conllu/multilang/albanian data/conllu/multilang/amharic data/conllu/multilang/ancient_greek data/conllu/multilang/apurina data/conllu/multilang/arabic data/conllu/multilang/armenian data/conllu/multilang/assyrian data/conllu/multilang/bambara data/conllu/multilang/basque data/conllu/multilang/belarusian data/conllu/multilang/bengali data/conllu/multilang/bhojpuri data/conllu/multilang/breton data/conllu/multilang/bulgarian data/conllu/multilang/buryat data/conllu/multilang/cantonese data/conllu/multilang/catalan data/conllu/multilang/chinese data/conllu/multilang/chukchi data/conllu/multilang/coptic data/conllu/multilang/croatian data/conllu/multilang/czech data/conllu/multilang/danish data/conllu/multilang/dutch data/conllu/multilang/erzya data/conllu/multilang/estonian data/conllu/multilang/faroese data/conllu/multilang/finnish data/conllu/multilang/french data/conllu/multilang/galician data/conllu/multilang/german data/conllu/multilang/greek data/conllu/multilang/guajajara data/conllu/multilang/hebrew data/conllu/multilang/hindi data/conllu/multilang/hungarian data/conllu/multilang/icelandic data/conllu/multilang/indonesian data/conllu/multilang/irish data/conllu/multilang/italian data/conllu/multilang/japanese data/conllu/multilang/javanese data/conllu/multilang/kangri data/conllu/multilang/karelian data/conllu/multilang/kazakh data/conllu/multilang/khunsari data/conllu/multilang/kiche data/conllu/multilang/komi_permyak data/conllu/multilang/komi_zyrian data/conllu/multilang/korean data/conllu/multilang/kurmanji data/conllu/multilang/latin data/conllu/multilang/latvian data/conllu/multilang/ligurian data/conllu/multilang/lithuanian data/conllu/multilang/livvi data/conllu/multilang/low_saxon data/conllu/multilang/maltese data/conllu/multilang/manx data/conllu/multilang/marathi data/conllu/multilang/mbya_guarani data/conllu/multilang/moksha data/conllu/multilang/munduruku data/conllu/multilang/nayini data/conllu/multilang/neapolitan data/conllu/multilang/north_sami data/conllu/multilang/norwegian data/conllu/multilang/old_east_slavic data/conllu/multilang/old_french data/conllu/multilang/persian data/conllu/multilang/polish data/conllu/multilang/portuguese data/conllu/multilang/romanian data/conllu/multilang/sanskrit data/conllu/multilang/scottish_gaelic data/conllu/multilang/serbian data/conllu/multilang/skolt_sami data/conllu/multilang/slovak data/conllu/multilang/slovenian data/conllu/multilang/soi data/conllu/multilang/south_levantine_arabic data/conllu/multilang/spanish data/conllu/multilang/swedish data/conllu/multilang/swiss_german data/conllu/multilang/tagalog data/conllu/multilang/tamil data/conllu/multilang/tatar data/conllu/multilang/telugu data/conllu/multilang/tupinamba data/conllu/multilang/turkish data/conllu/multilang/turkish_german data/conllu/multilang/ukrainian data/conllu/multilang/upper_sorbian data/conllu/multilang/urdu data/conllu/multilang/uyghur data/conllu/multilang/vietnamese data/conllu/multilang/warlpiri data/conllu/multilang/welsh data/conllu/multilang/western_armenian data/conllu/multilang/wolof data/conllu/multilang/xibe data/conllu/multilang/yoruba data/conllu/multilang/yupik data/dataset_mult
