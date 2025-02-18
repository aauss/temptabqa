# Evaluate temporal error

## Motivation

The goal is to calculate the temporal error in the TempTabQA dataset beyond token-level similarity. Usually, in QA benchmarks, F1, Acc., or EM are used to evaluate the LLM's performance. Assume that the expected and predicted answer were "12 months" and "one year" respectively. With the usual metrics, the performance would be 0 but when comparing predicted and expected answer temporally, they are both the same.

Using the TempTabQA dataset, I want to measure this error. How much better are LLMs when evaluated on temporal correctness? Does the temporal error reveal interesting patterns? For which questions is the temporal error particularly high?

Some exploratory work is documented in a [quarto report](../quarto_reports/supervision_temp_err_ana_2025-02-07.qmd).

## Setup

I will pick the head dataset. Most LLM's performed better there and it is bigger. I want to assess the benefit of temporal error analysis in the optimal setting.

## Work done

### Pre-label temporal answer

In a [notebook](./label_temp_answers_head_dataset/01_prelabel_temporal_questions.ipynb), I loaded the [head dataset](../../data/maindata/qapairs/head-set/head-set.csv). I filtered it for temporal question. I used the authors [data analysis script](../../data/dataanalysis/dataAnalysis.py) to classify answer by answer type. I removed all answer types that were clearly not temporal.

### Correct miss-classifications

#### Correct not-temporal answers

I double checked all QA-pairs that had a non-temporal answer in Excel and corrected a few. The resulting [dataset](./label_temp_answers_head_dataset/data/01a_prelabel_temp_q_head_data_corrected.csv) is saved in this folder.

#### Correct temporal answers

There were much more answers pre-labeled as temporal. Therefore, I used Label Studio to double check the labels. The [interface setting](./label_temp_answers_head_dataset/data/02_label_studio_interface) is saved in this folder. I used the dataset produced in the [step before](#correct-not-temporal-answers), replaced booleans with strings for better handling in label studio. I then only labeled data that was assigned the temporal label before. The [result](./label_temp_answers_head_dataset/data/02_man_label_temp_q_head_data.json) is saved in this folder.

I found a few issues with the dataset. This includes:

- duplicates,
- grammatical,
- erroneous answers,
- inconsistencies in date/time format
- string and digit representations are mixed
- Questions not asking for a time unit and answer not providing it too, e.g., Q: "How long ago did Faith die?", A: "46". More examples in this [notebook](./label_temp_answers_head_dataset/02_etl_manually_labeled_data.ipynb)

### Clean up labeled head-dataset

In this [notebook](./label_temp_answers_head_dataset/02_etl_manually_labeled_data.ipynb), I loaded the data from Label Studio and transformed it into a pandas-friendly CSV. I also removed duplicates and replaced string representations of digits to digits.

## Ongoing

This is a description of the work that I am currently conducting.

### Measure temporal error

Where the expected answer is temporal, I can conduct a temporal error analysis. I am using the cleaned, labeled [dataset](./label_temp_answers_head_dataset/data/notebook_output/02_man_label_temp_q_head_data_clean.csv) produced in the [prior step](#clean-up-labeled-head-dataset) I need to identify the expected time unit first. This allows me to normalize the LLM's response so that I can calculate the error measured in time.

#### Answers with time unit "year"
In the first step, I will work on answers that expect the answer to be a number of years. Based on the labeling work, I assume this to be the largest chunk of data.
