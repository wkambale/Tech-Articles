---
title: "PandasAI: The Generative AI Library"
seoTitle: "Understanding PandasAI: The Generative AI Library"
seoDescription: "Pandas AI, the Python library that enhances Pandas with generative artificial intelligence capabilities."
datePublished: Fri Jun 16 2023 12:45:47 GMT+0000 (Coordinated Universal Time)
cuid: cliykcdce000109ic0jc2byqi
slug: pandasai
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1686870501613/f3aef01b-8403-4f41-86c3-f738a1e749ec.png
tags: data-analysis, pandas, generative-ai

---

# Introduction

### What is PandasAI?

PandasAI is an advanced library built on top of the popular Pandas library, designed to provide enhanced functionality for data manipulation, analysis, and AI-driven tasks. With PandasAI, you can efficiently handle large datasets, perform complex operations, and leverage artificial intelligence techniques seamlessly. In this article, we will explore the key features of PandasAI with practical examples and code snippets.

> Read more about [Pandas](https://kambale.dev/pandas-the-gateway-to-data-exploration-and-visualization) here.

### Key Features of PandasAI

PandasAI extends the functionality of Pandas with additional features. Some of the key features are:

**Feature Engineering**: PandasAI offers a wide range of feature engineering techniques such as one-hot encoding, binning, scaling, and generating new features.

**AI-driven Operations**: PandasAI integrates with popular AI libraries like scikit-learn and TensorFlow, enabling seamless integration of machine learning and deep learning algorithms with Pandas data-frames.

**Exploratory Data Analysis (EDA)**: It provides various statistical and visualization tools for EDA, including descriptive statistics, correlation analysis, and interactive visualizations.

**Time Series Analysis**: PandasAI includes powerful tools for handling time series data, such as resampling, lagging, rolling computations, and date-based operations.

# Installation of PandasAI

To install PandasAI, you can use the `pip` package manager, which simplifies the process. Run the following command to install PandasAI:

```bash
pip install pandasai
```

If you wish to use PandasAI in a Google Colab Notebook like I am doing, you need to run the following commands to install PandasAI and other necessary modules:

```bash
!pip install pandasai
!pip install langchain
```

After successfully installing PandasAI, you need to import it along with the Pandas library to start using its enhanced functionalities.

```python
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
```

To use the OpenAI library, we will have to generate an OpenAI API Key [here](https://platform.openai.com/account/api-keys). Also, ensure that you set up a paid account for access to OpenAI's Large Language Models (LLM) which are priced per 1,000 tokens.

# Functionality of PandasAI

Before exploring the functionality of PandasAI, we will have to first set up an OpenAI environment and create an instance of PandasAI with the OpenAI environment we created.

```python
# Loading the API token to OpenAI environment
env = OpenAI(api_token='OpenAI API Key')

# Initializing an instance of PandasAI with OpenAI environment
pandasAi = PandasAI(env)
```

### Data Exploration

In this article, we are going to use the employee dataset containing information about employees, including their names, ages, salaries, and departments. However, this dataset has missing values in some of the columns. Below is a screenshot of the dataset:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1686866709849/6b89f818-c64e-44c4-b284-62ca5d7f3143.png align="center")

To explore our data, we will now prompt or ask `pandasAi` by passing in our dataset name and the question we wish to ask. In the code snippet below, we will ask PandasAI to tell us which employees have null values in the dataset.

```python
question = "Which employees have null values?"
pandasAi.run(data, prompt=question)
```

Output:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1686866175977/51decd4f-4589-471c-96b2-386c6c7d2c89.png align="center")

To check whether PandasAI is correct with the output given from our dataset, we can run a query for null values using Pandas itself with the `isna()` function.

```python
# Check for null values using Pandas
data = data[data.isna().any(axis=1)]
# Print the rows with the null values
print(data)
```

Output:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1686866485623/fea66a6e-724b-48bb-a865-205929a2e647.png align="center")

From the above output, we can see that both Pandas and PandasAI return the same rows that have null values in our dataset.

Next, we can prompt `pandasAi` to tell us which employee earns more than all the employees in the dataset. We will run the following code:

```python
question = "Who earns more than the others?"
pandasAi.run(data, prompt=question)
```

Output:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1686867077123/09762bd3-7f81-4e11-9fc1-e86335e80167.png align="center")

With a salary of 2000000, Marvin is the highest-paid employee. And we can see from the output above that PandasAI is correct.

Up next, we can ask PandasAI to fill in the null values for the employees without salaries. To do this, we are going to run the following code:

```python
question = "Fill in only the null values to 5 figure salaries"
pandasAi.run(data, prompt=question)
```

Output:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1686868481972/ae10a13f-3ba0-4d68-a85f-7fa2f729df83.png align="center")

From the above output, PandasAI can fill in the null values with a 5-figure salary for each employee who previously had a null value. The salary is uniform for each, but you can always twist the question/prompt to randomize the numbers.

In the next prompt, we can ask PandasAI to tell us how many employees are in different departments. This is crucial if one wishes to analyze employee data to know how employees are distributed across departments. In the code below, we prompt for the number of employees in the Sales department:

```python
question = "How many employees are in the Sales department?"
pandasAi.run(data, prompt=question)
```

Output:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1686869264768/a66fc227-636d-449f-99c7-f3f128ae91d6.png align="center")

Indeed there are 5 employees in the Sales department in our dataset.

There's more a data analyst can prompt PandasAI to do in terms of data exploration. As you can realize, we do not run the traditional Pandas codes and queries to analyze our dataset, but instead, supply English language prompts and the generative AI library will use OpenAI's LLM capabilities to return outputs.

### Data Visualization

As with Pandas, PandasAI can also be used to visualize data with simple prompts supplied to `pandasAi`. Below, we prompt PandasAI to plot visual charts of our data.

```python
question = "Plot a barplot of all employees and their salaries"
pandasAi.run(data, prompt=question)
```

Output:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1686869942147/67d4a043-c129-41b6-8166-804d6efe110a.png align="center")

We can plot the employee's salaries grouped by their departments and see which departments get more salary amounts compared to other departments. To do that, we twist the prompt as seen below:

```python
question = "Plot a barplot of all employees and their salaries grouped in their departments"
pandasAi.run(data, prompt=question)
```

Output:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1686870752231/7ce98ec4-bb0f-4bad-9d7b-bab6f1a5a4e8.png align="center")

Next, we can plot a boxplot for the employee's salary and age with the following prompt:

```python
question = "Plot a boxplot out of the employee salary and age"
pandasAi.run(data, prompt=question)
```

Output:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1686871372531/535ee4ca-5199-4bff-819a-546dae7a9550.png align="center")

We can also show the relationship between the employee's salaries and their age. To do that, we run the following prompt:

```python
question = "Plot a scatter graph from the employee data"
pandasAi.run(data, prompt=question)
```

Output:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1686871578521/01c665ca-a57f-43e0-a274-5b3f55c4c2f1.png align="center")

# Conclusion

PandasAI extends the capabilities of Pandas by providing advanced data manipulation, analysis, and AI-driven operations. In this article, we covered key features, and use cases, and provided examples and code snippets to illustrate the functionality of PandasAI. By leveraging PandasAI, you can streamline your data preprocessing pipeline and seamlessly integrate AI techniques into your Pandas workflows.

# Resources

[Google Colab Notebook](https://colab.research.google.com/drive/1sB1gAaz6GGlLkmkhP_foxZ3-x4qsZo-z?usp=sharing)

[Dataset](https://drive.google.com/file/d/12pV2KysMz0UPrH7GEgH8x3OwQwtJgJkw/view?usp=sharing)

[PandasAI Repository](https://github.com/gventuri/pandas-ai)

[PandasAI Docs](https://pandas-ai.readthedocs.io/en/latest/)