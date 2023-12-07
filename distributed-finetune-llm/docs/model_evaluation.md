# Evaluating Fine-Tuned Llama 2 7B Model

As we are trying to train the model for a specific use-case, we must test the model on this use-case after training, and record how well the model does. This is the main  objective of evaluation. Evaluation is defined as the process of understanding the reliability of a model.

Evaluation is normally done by using a dataset similar to the use case that the model was trained on. The model is given an input (in our case of fine-tuning [Llama 2 7B](https://ai.meta.com/llama/) for a  question-answering use case, the input would be the question), and the model's output will be compared to the expected output using a metric, which normally outputs a number. This is done for all data points in the dataset, and the final metric value is returned as the evaluation of the model on the dataset.

## Evaluation Results of Fine-Tuned Llama 2 7B Model

We trained the Llama 2 7B model with the PubMedQA pqa_artificial subset on [our cluster setup](./raycluster_setup.md#our-test-cluster-for-llama-2-7b-fine-tuning) with 126 batch size per device for 15 epochs. The training and evaluation losses were 1.84 and 1.91 respectively.

> *Note: The result may vary for you based on the hyper parameter used for fine-tuning*

We gave both Llama 2 7B base and fine-tuned Llama 2 7B models several questions without context in the medical domain and recorded their answers. We didn't provide the context with the question for the following reasons:

* To eliminate the possibility that the base Llama 2 7B model was getting the answer just from the context.
* In real life question-answering situations, it is rarely the case that a context is given along with the question.

The following is an example where the base Llama 2 7B model gives the wrong answer, while the Llama 2 7B fine-tuned model gives the right answer:

| Input Prompt | Expected Answer | Base Llama 2 7B | Fine-Tuned Llama 2 7B |
| --- | --- | --- | --- |
| Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. <br> ### Instruction: <br> Is pentraxin 3 reduced in bipolar disorder? <br> ### Input: <br> ### Response: | Individuals with bipolar disorder have low levels of pentraxin 3 which may reflect impaired innate immunity. An increased understanding of the role of innate immunity in the etiopathogenesis of bipolar disorder might lead to new modalities for the diagnosis and treatment of this disorder. | Pentraxin 3 is a protein that is increased in bipolar disorder.	| Yes, pentraxin 3 is reduced in bipolar disorder. Bipolar disorder is a type of mood disorder that is characterized by extreme mood swings, including episodes of mania and depression. Studies have shown that people with bipolar disorder have lower levels of pentaxin 3. |

The following is an example of a question where the base Llama 2 7B model gives an insufficient answer, as compared to the fine-tuned model, which gives a complete answer:

| Input Prompt | Expected Answer | Base Llama 2 7B | Fine-Tuned Llama 2 7B |
| --- | --- | --- | --- |
|Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. <br> \### Instruction: <br> Is cD133-positive tumor cell content a predictor of early recurrence in colorectal cancer? <br> \### Input: <br> ### Response: | These findings indicate that CD133 is a valid marker for identifying cancer stem cells from fresh surgically resected colorectal cancer tissues. Furthermore, we successfully established a living tumor tissue bank using surgically resected colorectal tissues with a viability of >70%.	| The cD133-positive tumor cell content is a predictor of early recurrence in colorectal cancer. | Yes, cD133-positive tumor cell content is a predictor of early recurrence in colorectal cancer. In a study of 150 patients with colorectal cancer, cD133-positive tumor cell content was associated with an increased risk of early recurrence. |

The following is an example of where the base Llama 2 7B model and our fine-tuned model gave approximately the same right answer:

| Input Prompt | Expected Answer | Base Llama 2 7B | Fine-Tuned Llama 2 7B |
| --- | --- | --- | --- |
|Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. <br> \### Instruction: <br> Does obstructive sleep apnea affect aerobic fitness? <br> \### Input: <br> \### Response: | Overall, in a US Air Force population, OSA does not predict a decrease in aerobic fitness as measured by cycle ergometry. However, patients with an apnea-hypopnea index of greater than 20 have a statistically significant decrease in aerobic fitness compared to the normal population. This study demonstrates the effects of OSA on aerobic fitness. Further correlation of fitness testing results with OSA severity and treatment is needed.	| The effects of obstructive sleep apnea on aerobic fitness are not well understood. However, studies have shown that people with sleep apnea have lower levels of fitness than those without the condition. There are a few possible explanations for this. First, people with sleep apnea may not be getting enough sleep, which can lead to decreased aerobic fitness. Second, people with sleep apnea may be more likely to have other health problems that can affect aerobic fitness.	| Obstructive sleep apnea is a common sleep disorder in which breathing is interrupted during sleep. This can lead to lower aerobic fitness levels and an increased risk of heart disease and stroke. However, there is no clear evidence that obstructive sleep apnea affects aerobic fitness. |
