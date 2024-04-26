# FYP
This repo contains our final year project which which was DermAssistant a mobile based skin disease detection app.I have used MLOPS techniques to make this project scalable and automated. The following poster shows how we approached our porject:

<img width="541" alt="image" src="https://github.com/farazamjad/FYP/assets/81928514/4ac392a6-ea1c-46a4-8677-e46d506eff08">

# Results and Outputs:
## Classification Report

<img width="1416" alt="image" src="https://github.com/farazamjad/FYP/assets/81928514/aff39d63-0bcf-44d4-b742-7931639e317b">


## Confusion Matrix

<img width="1206" alt="image" src="https://github.com/farazamjad/FYP/assets/81928514/eac93f9c-2dbb-461f-82f8-870482e1b1a4">

## Using App to Classify skin disease
<img width="1145" alt="image" src="https://github.com/farazamjad/FYP/assets/81928514/226521c5-213c-4d53-935a-5fa8b126c59f">


# The following MLOPS techniques were used in this project:
# 1. DVC:
1.Versioning and Reproducibility: DVC allows you to version control your datasets, just like you version control your code with Git. It tracks changes to the dataset files, allowing you to easily switch between different versions and reproduce previous experiments.

2.Data Pipeline Management: DVC helps you manage complex data pipelines by defining dependencies and stages in your pipeline. It allows you to specify the input data, the processing steps, and the output data, ensuring that each step is executed in the correct order and only when necessary.

3.Storage Optimization: DVC provides features to optimize storage space when working with large datasets. Instead of storing multiple copies of the entire dataset, DVC uses a combination of hard links and symbolic links to efficiently manage data versions and minimize disk space usage.
# 2. Github Actions:
1.Continuous Integration and Delivery (CI/CD): GitHub Actions enables you to set up CI/CD pipelines for your projects. You can define workflows that automatically build, test, and deploy your code whenever changes are pushed to your repository. This helps ensure the quality and reliability of your codebase and streamlines the deployment process.

2.Code Quality and Linting: GitHub Actions allows you to integrate code quality and linting tools into your workflows. You can run static code analysis, code formatting, and other quality checks on your codebase to enforce coding standards and maintain code consistency across your project.
# 3. Mlflow
1.Experiment Tracking: MLflow allows you to log and track experiments, including parameters, metrics, and output artifacts. It helps you keep a record of different model iterations, compare their performance, and reproduce experiments. This is especially useful for iterative model development and hyperparameter tuning.

2.Model Packaging and Versioning: MLflow provides a standardized format for packaging machine learning models. It allows you to easily save and version your models, making it straightforward to share and reproduce them in different environments. You can also bundle model dependencies and associated files within the package.

3.Model Registry: MLflow includes a model registry where you can store and manage trained models. The model registry provides version control, model lineage, and metadata management. It enables collaboration and facilitates the deployment and promotion of models to production.

4.Model Performance Monitoring: MLflow provides functionality for monitoring model performance and tracking metrics over time. You can log and compare metrics such as accuracy, precision, recall, and custom business-specific metrics. Monitoring model performance helps identify degradation, trigger retraining, and make data-driven decisions.

# 4. Dockers
1.Portability: Docker containers provide a consistent runtime environment regardless of the underlying host system. This portability enables you to package an application and its dependencies into a container once and run it anywhere, whether it's a developer's machine, a testing environment, or a production server. This consistency eliminates the "it works on my machine" problem and simplifies deployment across different environments.

2.Portability: Docker containers provide a consistent runtime environment regardless of the underlying host system. This portability enables you to package an application and its dependencies into a container once and run it anywhere, whether it's a developer's machine, a testing environment, or a production server. This consistency eliminates the "it works on my machine" problem and simplifies deployment across different environments.


