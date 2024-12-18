library(tidymodels)
library(tidyverse)
library(vroom)
library(themis)



library(embed) # for target encoding

trainData <- vroom("train.csv")
testData <- vroom("test.csv")

trainData$ACTION <- as.factor(trainData$ACTION)

my_recipe <- recipe(ACTION ~ ., data = trainData) |> 
  step_mutate_at(all_numeric_predictors(), fn = factor) |>  # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) |>  # combines categorical values that occur
  step_dummy(all_nominal_predictors()) |>  # dummy variable encoding
  step_normalize(all_numeric_predictors()) |> 
  #step_mutate_at(ACTION, fn = factor) |> 
  step_lencode_mixed(all_nominal_predictors(), 
                     outcome = vars(ACTION)) |> 
  step_pca(all_predictors(), threshold = 0.8) |> 
  step_smote(all_outcomes(), neighbors = 10)


# NOTE: some of these step functions are not appropriate to use together

# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = trainData)



## LOGISTIC REGRESSION




logRegModel <- logistic_reg() %>% #Type of model
  set_engine("glm")

## Put into a workflow here
logReg_workflow <-
  workflow() |> 
  add_model(logRegModel) |> 
  add_recipe(my_recipe) |> 
  fit(data = trainData)


## Make predictions
amazon_predictions <- 
  predict(logReg_workflow,
                              new_data = testData,
                              type = "prob") |> 
  bind_cols(testData) |> 
  rename(ACTION = .pred_1) |> 
  select(id, ACTION)



# KAGGLE SUBMISSION


## Write out the file
vroom_write(x = amazon_predictions, 
            file="./AmazonPreds.csv", delim=",")
