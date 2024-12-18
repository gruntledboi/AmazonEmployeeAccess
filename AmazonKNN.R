library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(lme4)
library(parsnip)


trainData <- vroom("train.csv")
testData <- vroom("test.csv")

trainData$ACTION <- as.factor(trainData$ACTION)

my_recipe <- recipe(ACTION ~ ., data = trainData) |> 
  step_mutate_at(all_numeric_predictors(), fn = factor) |>  # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) |>  # combines categorical values that occur
  #step_dummy(all_nominal_predictors()) |>  # dummy variable encoding
  #step_mutate_at(ACTION, fn = factor) |> 
  step_lencode_mixed(all_nominal_predictors(), 
                     outcome = vars(ACTION)) |> 
  step_normalize(all_numeric_predictors()) |> 
  step_pca(all_predictors(), threshold = 0.8) |> 
  step_smote(all_outcomes(), neighbors = 10)

#target encoding (must
# also step_lencode_glm() and step_lencode_bayes()


# NOTE: some of these step functions are not appropriate to use together

# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = trainData)



## knn model
knn_model <- nearest_neighbor(neighbors = tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)




## Fit or Tune Model HERE

## Grid of values to tune over
tuning_grid <- grid_regular(neighbors(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(trainData, v = 5, repeats = 1)

## Run the CV - ERROR HERE TO FIX!!!
CV_results <- knn_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = NULL) #Or leave metrics NULL

## Find Best Tuning Parameter
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

## Finalize the Workflow & fit it
final_wf <-
  knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = trainData)




knn_pen_predictions <- 
  predict(final_wf,
          new_data = testData,
          type = "prob") |> 
  bind_cols(testData) |> 
  rename(ACTION = .pred_1) |> 
  select(id, ACTION)



vroom_write(x = knn_pen_predictions, 
            file="./AmazonKNNPreds.csv", delim=",")
