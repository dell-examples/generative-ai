# PubmedQA Dataset Preprocessing

Delve into the details of the [PubMedQA medical dataset](https://huggingface.co/datasets/pubmed_qa) and the preprocessing techniques essential for effective fine-tuning.

## Table of Contents

* [PubMedQA Dataset](#pubmedqa-dataset)
    * [Why PubMedQA Dataset](#why-pubmedqa-dataset)
* [Dataset Preprocessing](#dataset-preprocessing)
    * [Dataset Cleaning](#dataset-cleaning)
    * [Applying Prompts](#applying-prompts)
        * [Text Prompt Format](#text-prompt-format)
        * [Prompt with Tokens Format](#prompt-with-tokens-format)
        * [Special Tokens Prompt Format](#special-tokens-prompt-format)

## PubMedQA Dataset

The PubMed question-answering dataset is a valuable collection of questions and answers linked to scientific articles from PubMed, one of the most extensive and respected databases of biomedical literature. It was designed to support research in natural language understanding and information retrieval in the context of healthcare and medicine, and it was the first QA dataset to emphasize reasoning over the qualitative contents of biomedical research to answer questions.

It consists of 1k expert annotated, 61.2k unlabeled, and 211.3k artificially generated QA instances. Each PubMedQA instance consists of a question, context (corresponding abstract), a long answer (conclusion of abstract), and a yes/no/maybe answer that summarizes the conclusion.
The size of the PubMedQA dataset is 2.06 GB with 273,518 rows.  A sample from the dataset looks like below:

| pubid    | question | context | long_answer | final_decision |
| -------- | ---------| --------|-------------|----------------|
| 25,423,883| Is enhancement of in vitro activity of tuberculosis drugs by addition of thioridazine reflected by improved in vivo therapeutic efficacy?| { "contexts": [ "Assessment of the activity of thioridazine towards Mycobacterium tuberculosis (Mtb), in vitro and in vivo as a single drug and in combination with tuberculosis (TB) drugs.", "The in vitro activity of thioridazine as single drug or in combination with TB drugs was assessed in terms of MIC and by use of the time-kill kinetics assay. Various Mtb strains among which the Beijing genotype strain BE-1585 were included. In vivo, mice with TB induced by BE-1585 were treated with a TB drug regimen with thioridazine during 13 weeks. Therapeutic efficacy was assessed by the change in mycobacterial load in the lung, spleen and liver during treatment and 13 weeks post-treatment."], "labels": [ "OBJECTIVE", "METHODS", "RESULTS" ], "meshes": [] }   |Thioridazine is bactericidal towards Mtb in vitro, irrespective of the mycobacterial growth rate and results in enhanced activity of the standard regimen. The in vitro activity of thioridazine in potentiating isoniazid and rifampicin is not reflected by improved therapeutic efficacy in a murine TB-model | no |



### Why PubMedQA Dataset

Although Llama2 is a state-of-the-art open-access LLM, it doesn’t have a domain-specific dataset, so the PubMedQA dataset was used to finetune the model to better generate responses in the medical domain. The PubMedQA dataset was chosen for several reasons, including:

1. PubMedQA features expert-annotated questions, making it a reliable and accurate resource for training and evaluating AI models in the biomedical domain.
2. PubMedQA requires reasoning/comprehension of biomedical literature for the model to answer questions.
3. Unlike similar QA datasets where crowd workers ask questions about existing contexts, PubMedQA’s contexts and questions are written by the same authors. This ensures that the data is consistent since the contexts and questions are directly related, making PubMedQA an ideal benchmark for testing scientific reasoning abilities.

## Dataset Preprocessing

### Dataset Cleaning

First the dataset needs to cleaned. The steps involved in cleaning the PubMedQA is as follows.

* Loading the Dataset: Firstly, the `pubmed_qa` dataset is loaded from the hugging face hub into the system.
* Removing Unnecessary Columns: The `pubid`, `context.labels`, `context.meshes` and `final_decision` columns are dropped from the dataset as they are unnecessary.
* Renaming Columns: The `context.contexts` column is renamed to `context` and `long_answer` is renamed to `answer` according to the required format.
* Data Type Conversion: Since, the `context` column is of datatype `Sequence`, it is converted to `String` datatype for easier processing. This is mapped to all the rows in the dataset.
* Creation of JSON File: The dataset is converted into a JSON file and stored in a directory.
* Creation of Train & Test Split: The dataset is split into train and test data in the 80:20 ratio or according to the test split given as input by the user.

### Applying Prompts
These dataset splits are then converted into prompts of different formats according to the user's input and are mapped to the entire train and test dataset and all the rows within those splits.

#### Text Prompt Format

```
Below is an instruction that describes a task, paired with an input that provides further context.
    Write a response that appropriately completes the request.

    ### Instruction:
    {data_point["question"]}

    ### Input:
    {data_point["context"]}

    ### Response:
    {data_point["answer"]}
```

E.g. A sample of the dataset created into the above prompt template is as follows:

```
Below is an instruction that describes a task, paired with an input that provides further context.
    Write a response that appropriately completes the request.

    ### Instruction:

    Is enhancement of in vitro activity of tuberculosis drugs by addition of thioridazine reflected by improved in
    vivo therapeutic efficacy?

    ### Input:
	Assessment of the activity of thioridazine towards Mycobacterium tuberculosis (Mtb), in vitro and in vivo as a
    single drug and in combination with tuberculosis (TB) drugs. In vitro, thioridazine showed a concentration-dependent
    and time-dependent bactericidal activity towards both actively-replicating and slowly-replicating Mtb. Thioridazine at
    high concentrations could enhance the activity of isoniazid and rifampicin, and in case of isoniazid resulted in
    elimination of mycobacteria and prevention of isoniazid-resistant mutants. Thioridazine had no added value in combination
    with moxifloxacin or amikacin.

    ### Response:
   	Thioridazine is bactericidal towards Mtb in vitro, irrespective of the mycobacterial growth rate and results in
    enhanced activity of the standard regimen. The in vitro activity of thioridazine in potentiating isoniazid and
    rifampicin is not reflected by improved therapeutic efficacy in a murine TB-model
```

#### Prompt with Tokens Format

```<START_Q>  {data_point["question"]}  {data_point["context"]} <END_Q> <START_A> {data_point["answer"]} <END_A>```


E.g. A sample of the dataset created into the above prompt template is as follows:

```
<START_Q> Is enhancement of in vitro activity of tuberculosis drugs by addition of thioridazine reflected by
improved in vivo therapeutic efficacy? Assessment of the activity of thioridazine towards Mycobacterium tuberculosis (Mtb),
in vitro and in vivo as a single drug and in combination with tuberculosis (TB) drugs. In vitro, thioridazine showed a
concentration-dependent and time-dependent bactericidal activity towards both actively-replicating and slowly-replicating Mtb.
Thioridazine at high concentrations could enhance the activity of isoniazid and rifampicin, and in case of isoniazid resulted in
elimination of mycobacteria and prevention of isoniazid-resistant mutants. Thioridazine had no added value in combination with
moxifloxacin or amikacin. <END_Q> <START_A>Thioridazine is bactericidal towards Mtb in vitro, irrespective of the mycobacterial
growth rate and results in enhanced activity of the standard regimen. The in vitro activity of thioridazine in potentiating
isoniazid and rifampicin is not reflected by improved therapeutic efficacy in a murine TB-model <END_A>
```

#### Special Tokens Prompt Format

```<START_Q>  {data_point["question"]} <END_Q> <START_C>  {data_point["context"]} <END_C> <START_A> {data_point["answer"]} <END_A>```


E.g. A sample of the dataset created into the above prompt template is as follows:

```
<START_Q> Is enhancement of in vitro activity of tuberculosis drugs by addition of thioridazine reflected by
improved in vivo therapeutic efficacy?<END_Q> <START_C> Assessment of the activity of thioridazine towards Mycobacterium
tuberculosis (Mtb), in vitro and in vivo as a single drug and in combination with tuberculosis (TB) drugs. In vitro,
thioridazine showed a concentration-dependent and time-dependent bactericidal activity towards both actively-replicating
and slowly-replicating Mtb. Thioridazine at high concentrations could enhance the activity of isoniazid and rifampicin,
and in case of isoniazid resulted in elimination of mycobacteria and prevention of isoniazid-resistant mutants. Thioridazine
had no added value in combination with moxifloxacin or amikacin. <END_C> <START_A>Thioridazine is bactericidal towards Mtb
in vitro, irrespective of the mycobacterial growth rate and results in enhanced activity of the standard regimen. The in
vitro activity of thioridazine in potentiating isoniazid and rifampicin is not reflected by improved therapeutic efficacy
in a murine TB-model <END_A>

```
