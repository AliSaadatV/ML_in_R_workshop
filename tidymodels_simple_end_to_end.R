library(readxl) # for importing file
library(tidymodels) # for ML
library(rpart.plot) # For decision trees
library(vip) # for variable importance plots
library(naivebayes) # for naive bayes
library(randomForest) # for random forest
library(stringr) # for data formatting
library(DALEXtra) # for explain_tidymodels
tidymodels::tidymodels_prefer()

data <- read_excel("/Users/stang/Desktop/R_analysis/tidymodelsworkshop/Table2.XLSX", sheet = "Sheet1")
colnames(data) <- data[3,]
data_clean <- data[-c(1:3),] # Remove the first three rows, which are just titles
data_clean <- data_clean[,-c(1:2)] # Remove the first two columns, as they don't contribute anything
data_clean[, 1:5] <- apply(data_clean[, 1:5], 2, as.numeric)

chosen_metric <- "f_meas"
all_metrics <- metric_set(accuracy, roc_auc, f_meas, mcc, sens, spec, bal_accuracy)


# Split data, let's keep 25% for the test.
set.seed(123)
splits <- initial_split(data_clean, prop = 0.75, strata = judge)
data_train <- training(splits)
data_test  <- testing(splits)

# We build a recipe.
data_rec_knn <-
  recipe(judge ~ ., data = data_clean) %>%
  update_role(Pubmed, DCDB, new_role = "ID") %>%
  step_zv(all_predictors()) %>% 
  step_impute_knn(all_predictors(), neighbors = 5) %>% 
  step_YeoJohnson(all_predictors()) %>% 
  step_normalize(all_predictors()) 

# We build a model.
randomforest_spec <-
  rand_forest(mtry = tune(),
              trees = tune(),
              min_n = tune()
  ) %>%
  set_engine("randomForest") %>%
  set_mode("classification")

# We create the workflow.
workflow <- 
  workflow() %>% 
  add_model(randomforest_spec) %>% 
  add_recipe(data_rec_knn)

# Hyperparameter tuning setting
grid_ctrl <- 
  control_grid(
    save_pred = TRUE,
    parallel_over = "everything",
    save_workflow = TRUE
  )

# 10-fold CV
set.seed(123)
data_folds <- vfold_cv(data_train, v = 10, strata = judge)

# Train model on train set
grid_results_impute <- 
  workflow %>% 
  tune_grid(resamples = data_folds, # cross-fold validation groupings
            grid = 50,
            control = grid_ctrl,
            metrics = all_metrics
  )

# Select best hyperparameters of best model
best_results_parameters <- select_best(grid_results_impute, metric = "f_meas")

### Finalise our workflow and test on unseen data.
final_wf <- 
  grid_results_impute %>% 
  extract_workflow(best_model_id) %>% 
  finalize_workflow(best_results_parameters) %>%
  last_fit(split = splits, metrics = all_metrics) %>% #fit it against our test data
  collect_metrics() # collect metrics about predictive model

