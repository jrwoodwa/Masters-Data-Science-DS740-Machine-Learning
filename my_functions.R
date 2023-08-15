#--------------------------------------------------------------
# Function to augment data
augment_data <-
  function(df,
           duplicates = 3,
           perturb_sd = 1,
           perturb_mean = 0) {
    # Create perturbed duplicates (normal distribution)
    duplicates_df <- purrr::map_dfr(seq_len(duplicates), ~ df %>%
                                      mutate(across(everything(), ~ pmax(
                                        0, . + (
                                          rnorm(1, mean = perturb_mean, sd = perturb_sd) * sample(c(-1, 1), size = 1)
                                        )
                                      ))))
    return(duplicates_df)  # Return only duplicates
  }

#--------------------------------------------------------------
# function for Single Cross-Validation
single_CV_rpart_VS_xgbTree <- function(dataused,
                                       modelApplied,
                                       tunegrid_rpart,
                                       tunegrid_xgbTree,
                                       kfolds = 5,
                                       OuterFold = 1, # outerloop (Default: 1 single-CV))
                                       xgbTreeRuns = 3, # best of three, the slower one only gets 2
                                       rpartRuns =1,
                                       ExcludeLabel = NULL
                                       )  # decision tree runs 17 times as fast as xgbTree for my data
  
{
  ############ Single Cross Validation 5-fold (default) ############
  # Initials
  results_df_function <- # Create an empty dataframe
    data.frame(
      Parameter = character(),
      Value = numeric(),
      Kappa = numeric(),
      methodApplied = character(),
      modelApplied = factor(),
      OuterLoop = numeric(),
      RunNumber = numeric(),
      stringsAsFactors = FALSE
    )
  rpart_models <- list()
  xgBoost_models <- list()
  
  RunMax_xgbTree = xgbTreeRuns
  RunMax_rpart = rpartRuns # non-random
  RunMax = max(RunMax_rpart, RunMax_xgbTree)
  
  bestrun_rpart = 0 # initialize minimums per model (ONLY save models if they're a new record, per model & method)
  bestrun_xgBoost = 0
  kappa_rpart = -10000
  kappa_xgBoost = -10000
  
  # Loop through dataframes
  for (dd in seq_along(Classification_Labels)) {
    if (identical(as.numeric(dd), ExcludeLabel)) {
      next  # Skip loop if condition met
    }
    # combine the 2/3 class label
    # (no need to subset/index the data, it is already just traindata)
    dataused <- cbind(SleepQuality = Classification_Labels[[dd]],
                      data.out)
    dataused_def <-
      as.factor(ifelse(dd == 1, "3-class", "2-class")) # magic numbers for the dataframe data's description
    #kk = 1 #testing
    
    # after data provided stratified k-fold
    cvIndex <-
      createFolds(dataused$SleepQuality, # magic number but OK
                  kfolds,
                  returnTrain = TRUE) # Create stratified k-folds
    training_general <- # k-fold
      trainControl(index = cvIndex,
                   method = 'cv',
                   number = kfolds)
    training_xgbTree <- # use parallel processing RAM
      trainControl(
        index = cvIndex,
        method = 'cv',
        number = kfolds,
        allowParallel = TRUE
      )
    
    for (kk in 1:RunMax) {
      
      if (kk <= RunMax_rpart) {
        # Decision Tree (no randomness)
        fit_caret_rpart <- train(
          modelApplied,
          data = dataused,
          na.action = na.pass,
          # doing nothing, Decision Tree uses NAs!~
          method = "rpart",
          trControl = training_general,
          tuneGrid = tunegrid_rpart,
          metric = "Kappa" # Set Kappa as optimization metric
        )
        
        # store result
        kappa <- max(fit_caret_rpart$results$Kappa)
        
        df_insert <- # get long version for a df (neat!)
          as.data.frame(gather(
            fit_caret_rpart$bestTune,
            key = Parameter,
            value = Value
          )) %>% # dplyr for rows (plural)
          mutate(
            Kappa = max(fit_caret_rpart$results$Kappa),
            methodApplied = fit_caret_rpart$method,
            modelApplied = dataused_def,
            OuterLoop = OuterFold,
            RunNumber = kk
          )
        results_df_function <- # insert rows
          rbind(results_df_function,
                df_insert)
        
        # save best run
        if (kappa > kappa_rpart) {
          bestrun_rpart <- kk
          kappa_rpart <- kappa
          rpart_models[[1]] <-
            fit_caret_rpart  # Save to the list, best run
        }
      } # end of rpart
      
      if (kk <= RunMax_xgbTree) {
        #Xtreme Gradient Boosting
        fit_caret_xgBoost <- train(
          modelApplied,
          data = dataused,
          na.action = na.pass,
          # doing nothing, Decision Tree uses NAs!~
          method = "xgbTree",
          trControl = training_xgbTree,
          tuneGrid = tunegrid_xgbTree,
          verbosity = 0,
          # quiet
          metric = "Kappa"  # Set Kappa as optimization metric
        )
        # store result
        kappa <- max(fit_caret_xgBoost$results$Kappa)
        
        df_insert <- # get long version for a df (neat!)
          as.data.frame(gather(
            fit_caret_xgBoost$bestTune,
            key = Parameter,
            value = Value
          )) %>% # dplyr for rows (plural)
          mutate(
            Kappa = max(fit_caret_xgBoost$results$Kappa),
            methodApplied = fit_caret_xgBoost$method,
            modelApplied = dataused_def,
            OuterLoop = OuterFold,
            RunNumber = kk
          )
        results_df_function <- # insert rows
          rbind(results_df_function,
                df_insert)
        
        # save best run
        if (kappa > kappa_xgBoost) {
          bestrun_xgBoost <- kk
          kappa_xgBoost <- kappa
          xgBoost_models[[1]] <-
            fit_caret_xgBoost  # Save to the list, best run
        }
      } # end of xgb
    } # end of random runs kk
    
  } # end of dd
  
  if (kappa_xgBoost > kappa_rpart) {
    bestModel = xgBoost_models[[1]] # if better Kappa, better method
    dataused_def = dataused_def
  }
  else {
    bestModel = rpart_models[[1]] # else the other method is better
    dataused_def = dataused_def
  }
  
  return(list(
    df = results_df_function,
    bestModel = bestModel,
    dataused_def = dataused_def
  ))
} # end of function






#--------------------------------------------------------------

## Storing results

# Name= 2-class, DecisionTree
# ExcludeLabel = 1,xgbTreeRuns = 0,rpartRuns = 1
#"Double CV: 0.676 (1)"
#"Double CV: 0.721 (2)"
#"Double CV: 0.752 (3)"
#"Double CV: 0.606 (4)"
#"Double CV: 0.7 (5)"
#"Double CV: 0.7 (6)"
#"Double CV: 0.667 (7)"
#"Double CV: 0.691 (8)"
#"Double CV: 0.667 (9)"
#"Double CV: 0.755 (10)"
#"Double CV: 0.648 (11)"
#"Double CV: 0.694 (12)"
#"Double CV: 0.673 (13)"
#"Double CV: 0.639 (14)"
#"Double CV: 0.648 (15)"
#"Double CV: 0.682 (16)"
#"Double CV: 0.688 (17)"
#"Double CV: 0.67 (18)"
#"Double CV: 0.63 (19)"
#"Double CV: 0.627 (20)"
#"Double CV: 0.655 (26)"
#"Double CV: 0.627 (27)"
#"Double CV: 0.6 (28)"
#"Double CV: 0.6 (29)"
#"Double CV: 0.636 (30)"
#"Double CV: 0.682 (31)"
#"Double CV: 0.636 (34)"
#"Double CV: 0.648 (35)"
#"Double CV: 0.664 (36)"
#"Double CV: 0.645 (37)"
#"Double CV: 0.673 (38)"
#"Double CV: 0.6 (39)"
#"Double CV: 0.655 (40)"
#"Double CV: 0.606 (41)"
#"Double CV: 0.682 (42)"
#"Double CV: 0.588 (43)"
#"Double CV: 0.648 (44)"
#"Double CV: 0.694 (45)"
#"Double CV: 0.627 (46)"
#"Double CV: 0.633 (47)"
#"Double CV: 0.658 (48)"
#"Double CV: 0.733 (49)"
#"Double CV: 0.688 (50)"
#"Double CV: 0.679 (51)"
#"Double CV: 0.642 (52)"
#"Double CV: 0.685 (53)"
#"Double CV: 0.67 (56)"
#"Double CV: 0.727 (57)"
#"Double CV: 0.7 (58)"
#"Double CV: 0.621 (59)"
#"Double CV: 0.618 (60)"
#"Double CV: 0.667 (61)"
#"Double CV: 0.661 (62)"
#"Double CV: 0.7 (63)"
#"Double CV: 0.618 (64)"
#"Double CV: 0.658 (65)"
#"Double CV: 0.706 (66)"
#"Double CV: 0.642 (67)"
#"Double CV: 0.633 (68)"
#"Double CV: 0.636 (69)"
#"Double CV: 0.673 (70)"
#"Double CV: 0.667 (71)"
#"Double CV: 0.612 (72)"
#"Double CV: 0.63 (73)"
#"Double CV: 0.712 (74)"
#"Double CV: 0.7 (75)"
#"Double CV: 0.658 (76)"
#"Double CV: 0.706 (77)"
#"Double CV: 0.703 (80)"
#"Double CV: 0.648 (81)"
#"Double CV: 0.597 (82)"
#"Double CV: 0.658 (83)"
#"Double CV: 0.648 (84)"
#"Double CV: 0.645 (85)"
#"Double CV: 0.664 (86)"
#"Double CV: 0.667 (87)"
#"Double CV: 0.652 (88)"
#"Double CV: 0.691 (89)"
#"Double CV: 0.7 (90)"
#"Double CV: 0.673 (91)"
#"Double CV: 0.585 (92)"
#"Double CV: 0.685 (93)"
#"Double CV: 0.682 (94)"
#"Double CV: 0.618 (95)"
#"Double CV: 0.603 (96)"
#"Double CV: 0.715 (97)"
#"Double CV: 0.606 (98)"
#"Double CV: 0.67 (99)"
#"Double CV: 0.594 (100)"

# Name= 2-class, Boosting
# ExcludeLabel = 1,xgbTreeRuns = 3,rpartRuns = 0
#"Double CV: 0.621 (1)"
#"Double CV: 0.606 (2)"
#"Double CV: 0.609 (3)"
#"Double CV: 0.63 (4)"
#"Double CV: 0.642 (5)"
#"Double CV: 0.682 (6)"


# Name= 2-class, Boosting + DecisionTree
# ExcludeLabel = 1,xgbTreeRuns = 3,rpartRuns = 1
#"Double CV: 0.615 (100)"
#"Double CV: 0.655 (101)"
#"Double CV: 0.609 (102)"
#"Double CV: 0.655 (103)"
#"Double CV: 0.645 (104)"
#"Double CV: 0.621 (105)"
#"Double CV: 0.648 (106)"
#"Double CV: 0.621 (107)"

# Name= 3-class, Decision Tree
# ExcludeLabel = 2,xgbTreeRuns = 0,rpartRuns = 1
#"Double CV: 0.661 (26)"
#"Double CV: 0.579 (27)"
#"Double CV: 0.579 (28)"
#"Double CV: 0.579 (29)"
#"Double CV: 0.63 (30)"
#"Double CV: 0.57 (31)"
#"Double CV: 0.6 (32)"
#"Double CV: 0.621 (33)"
#"Double CV: 0.67 (34)"
#"Double CV: 0.579 (35)"
#"Double CV: 0.558 (36)"
#"Double CV: 0.618 (37)"
#"Double CV: 0.606 (38)"
#"Double CV: 0.591 (39)"
#"Double CV: 0.579 (40)"
#"Double CV: 0.603 (41)"
#"Double CV: 0.579 (42)"
#"Double CV: 0.655 (43)"
#"Double CV: 0.655 (44)"
#"Double CV: 0.579 (45)"
#"Double CV: 0.579 (46)"
#"Double CV: 0.597 (47)"
#"Double CV: 0.579 (48)"
#"Double CV: 0.579 (49)"
#"Double CV: 0.609 (50)"
#"Double CV: 0.579 (51)"
#"Double CV: 0.597 (52)"
#"Double CV: 0.612 (53)"
#"Double CV: 0.579 (54)"
#"Double CV: 0.624 (55)"
#"Double CV: 0.579 (56)"
#"Double CV: 0.579 (57)"
#"Double CV: 0.603 (58)"
#"Double CV: 0.627 (59)"
#"Double CV: 0.642 (60)"
#"Double CV: 0.579 (61)"
#"Double CV: 0.615 (62)"
#"Double CV: 0.645 (63)"
#"Double CV: 0.579 (64)"
#"Double CV: 0.579 (65)"
#"Double CV: 0.612 (66)"
#"Double CV: 0.585 (67)"
#"Double CV: 0.63 (68)"
#"Double CV: 0.612 (69)"
#"Double CV: 0.579 (70)"
#"Double CV: 0.618 (71)"
#"Double CV: 0.591 (72)"
#"Double CV: 0.664 (73)"
#"Double CV: 0.688 (74)"
#"Double CV: 0.579 (75)"
#"Double CV: 0.652 (76)"
#"Double CV: 0.579 (77)"
#"Double CV: 0.576 (78)"
#"Double CV: 0.588 (79)"
#"Double CV: 0.579 (80)"
#"Double CV: 0.633 (81)"
#"Double CV: 0.567 (82)"
#"Double CV: 0.585 (83)"
#"Double CV: 0.618 (84)"
#"Double CV: 0.591 (85)"
#"Double CV: 0.612 (86)"
#"Double CV: 0.588 (87)"
#"Double CV: 0.591 (88)"
#"Double CV: 0.594 (89)"
#"Double CV: 0.585 (90)"
#"Double CV: 0.639 (91)"
#"Double CV: 0.615 (92)"
#"Double CV: 0.682 (93)"
#"Double CV: 0.606 (94)"
#"Double CV: 0.618 (95)"
#"Double CV: 0.579 (96)"
#"Double CV: 0.6 (97)"
#"Double CV: 0.664 (98)"
#"Double CV: 0.661 (99)"
#"Double CV: 0.603 (100)"

# Name= 3-class, Boosting
# ExcludeLabel = 2,xgbTreeRuns = 3,rpartRuns = 0
#"Double CV: 0.633 (100)"
#"Double CV: 0.642 (101)"
#"Double CV: 0.6 (102)"
#"Double CV: 0.618 (103)"
#"Double CV: 0.627 (104)"
#"Double CV: 0.597 (105)"
#"Double CV: 0.615 (106)"
#"Double CV: 0.624 (107)"

# Name= 3-class, Boosting + Decision Tree 
# ExcludeLabel = 2,xgbTreeRuns = 3,rpartRuns = 1
#"Double CV: 0.621 (200)"
#"Double CV: 0.594 (201)"
#"Double CV: 0.612 (202)"
#"Double CV: 0.615 (203)"
#"Double CV: 0.627 (204)"
#"Double CV: 0.594 (205)"
#"Double CV: 0.63 (206)"
#"Double CV: 0.639 (207)"

# Name= 2 or 3-class, Decision Tree 
# ExcludeLabel = NULL,xgbTreeRuns = 0,rpartRuns = 1
# "Double CV: 0.612 (1)"
# "Double CV: 0.7 (2)"
# "Double CV: 0.464 (3)"
# "Double CV: 0.47 (4)"
# "Double CV: 0.4 (5)"
# "Double CV: 0.597 (6)"
# "Double CV: 0.552 (7)"
# "Double CV: 0.476 (8)"
# "Double CV: 0.579 (9)"
# "Double CV: 0.4 (10)"
# "Double CV: 0.497 (11)"
# "Double CV: 0.609 (12)"
# "Double CV: 0.524 (13)"
# "Double CV: 0.545 (14)"
# "Double CV: 0.609 (15)"
# "Double CV: 0.503 (16)"
# "Double CV: 0.467 (17)"
# "Double CV: 0.452 (18)"
# "Double CV: 0.385 (19)"
# "Double CV: 0.439 (20)"
# "Double CV: 0.585 (21)"
# "Double CV: 0.415 (22)"
# "Double CV: 0.476 (23)"
# "Double CV: 0.527 (24)"
# "Double CV: 0.509 (25)"
# "Double CV: 0.667 (200)"
# "Double CV: 0.464 (201)"
# "Double CV: 0.53 (202)"
# "Double CV: 0.552 (203)"
# "Double CV: 0.445 (204)"
# "Double CV: 0.436 (205)"
# "Double CV: 0.276 (206)"
# "Double CV: 0.603 (207)"
# "Double CV: 0.455 (208)"
# "Double CV: 0.488 (209)"
# "Double CV: 0.37 (210)"
# "Double CV: 0.448 (211)"
# "Double CV: 0.512 (212)"
# "Double CV: 0.588 (213)"
# "Double CV: 0.485 (214)"
# "Double CV: 0.648 (215)"
# "Double CV: 0.548 (216)"
# "Double CV: 0.436 (217)"
# "Double CV: 0.536 (218)"
# "Double CV: 0.433 (219)"
# "Double CV: 0.094 (220)"
# "Double CV: 0.536 (221)"
# "Double CV: 0.43 (222)"
# "Double CV: 0.579 (223)"
# "Double CV: 0.5 (224)"
# "Double CV: 0.585 (225)"
# "Double CV: 0.627 (226)"
# "Double CV: 0.57 (227)"
# "Double CV: 0.148 (228)"
# "Double CV: 0.539 (229)"
# "Double CV: 0.579 (230)"
# "Double CV: 0.455 (231)"
# "Double CV: 0.467 (232)"
# "Double CV: 0.564 (233)"
# "Double CV: 0.382 (234)"
# "Double CV: 0.43 (235)"
# "Double CV: 0.439 (236)"
# "Double CV: 0.327 (237)"
# "Double CV: 0.294 (238)"
# "Double CV: 0.591 (239)"
# "Double CV: 0.512 (240)"
# "Double CV: 0.615 (241)"
# "Double CV: 0.509 (242)"
# "Double CV: 0.576 (243)"
# "Double CV: 0.448 (244)"
# "Double CV: 0.467 (245)"
# "Double CV: 0.436 (246)"
# "Double CV: 0.461 (247)"
# "Double CV: 0.491 (248)"
# "Double CV: 0.548 (249)"
# "Double CV: 0.497 (250)"
# "Double CV: 0.321 (251)"
# "Double CV: 0.421 (252)"
# "Double CV: 0.564 (253)"
# "Double CV: 0.488 (254)"
# "Double CV: 0.388 (255)"
# "Double CV: 0.509 (256)"
# "Double CV: 0.433 (257)"
# "Double CV: 0.555 (258)"
# "Double CV: 0.582 (259)"
# "Double CV: 0.536 (260)"
# "Double CV: 0.661 (261)"
# "Double CV: 0.53 (262)"
# "Double CV: 0.633 (263)"
# "Double CV: 0.633 (264)"
# "Double CV: 0.624 (265)"
# "Double CV: 0.612 (266)"
# "Double CV: 0.421 (267)"
# "Double CV: 0.539 (268)"
# "Double CV: 0.497 (269)"
# "Double CV: 0.573 (270)"
# "Double CV: 0.188 (271)"
# "Double CV: 0.227 (272)"
# "Double CV: 0.482 (273)"
# "Double CV: 0.373 (274)"
# "Double CV: 0.445 (275)"
# "Double CV: 0.615 (276)"
# "Double CV: 0.573 (277)"
# "Double CV: 0.333 (278)"
# "Double CV: 0.297 (279)"
# "Double CV: 0.415 (280)"
# "Double CV: 0.485 (281)"
# "Double CV: 0.53 (282)"
# "Double CV: 0.536 (283)"
# "Double CV: 0.452 (284)"
# "Double CV: 0.318 (285)"
# "Double CV: 0.406 (286)"
# "Double CV: 0.494 (287)"
# "Double CV: 0.412 (288)"
# "Double CV: 0.594 (289)"
# "Double CV: 0.558 (290)"
# "Double CV: 0.527 (291)"
# "Double CV: 0.373 (292)"
# "Double CV: 0.348 (293)"
# "Double CV: 0.315 (294)"
# "Double CV: 0.6 (295)"
# "Double CV: 0.524 (296)"
# "Double CV: 0.485 (297)"
# "Double CV: 0.252 (298)"
# "Double CV: 0.445 (299)"
# "Double CV: 0.418 (300)"

# Name= 2 or 3-class, Boosting
# ExcludeLabel = NULL,xgbTreeRuns = 3,rpartRuns = 0
#"Double CV: 0.621 (1)"
#"Double CV: 0.606 (2)"
#"Double CV: 0.609 (3)"
#"Double CV: 0.63 (4)"
#"Double CV: 0.612 (5)"
#"Double CV: 0.555 (6)"
#"Double CV: 0.588 (7)"


# Name= 2 or 3-class, Boosting + Decision Tree
# ExcludeLabel = NULL,xgbTreeRuns = 3, rpartRuns = 1
#"Double CV: 0.639 (1)"
#"Double CV: 0.609 (2)"
#"Double CV: 0.291 (3)"
#"Double CV: 0.43 (4)"
#"Double CV: 0.391 (5)"
#"Double CV: 0.315 (6)"
#"Double CV: 0.639 (777)"
#"Double CV: 0.645 (77)"


##-----------------


gf_partialPlot <- function(model,
                           modelType,
                           df,
                           x.var,
                           na.action = "na.fail",
                           which.class = NULL) {
  # Generates partial dependence plot.
  # Args:
  # model - caret object,
  # df - data frame,
  # x.var - quoted column name,
  # which.class - quoted level
  library(ggformula)
  
  observed_data = df[[x.var]]
  test_vals = seq(min(observed_data, na.rm = TRUE),
                  max(observed_data, na.rm = TRUE),
                  length = 50)
  pred_avg = numeric(50)
  
  if (modelType == "classification") {
    for (ii in 1:50) {
      comp_df <- df
      comp_df[x.var] = test_vals[ii]
      
      probs = predict(model, 
                      comp_df, 
                      type = "prob", 
                      na.action = na.action)
      if (is.null(which.class)) {
        pred_avg[ii] = (mean(log(probs[, 1] / (1 - probs[, 1]))))
      } else {
        pred_avg[ii] = (mean(log(probs[, which.class] / (1 - probs[, which.class]))))
      }
    } # end iteration test_vals
  } # end "if Classification"
  else {
    # Regression model
    for (ii in 1:50) {
      comp_df <- df
      comp_df[x.var] = test_vals[ii]
      
      pred_avg[ii] = mean(predict(model, 
                                  comp_df,
                                  na.action = na.action))
    }
  } #end "if Regression"
  
  Probability <- exp(pred_avg)/(1+exp(pred_avg))
  
  to_return = gf_line(Probability ~ test_vals,
                      title = paste("Partial Depend. on", x.var),
                      xlab = x.var,
                      ylab= paste0('"',which.class,'" Probability'))
  
  return(to_return)
} # end of function



#----------------------------------------

plot_missing_data_heatmap <- function(data) {
  # Create a missingness heatmap plot for each variable in the data
  missing_data <- data %>%
    mutate(across(everything(), ~ is.na(.)))
  
  missing_data$row_number <- seq_len(nrow(missing_data))
  
  # Reshape the dataframe to a long format for ggplot
  missing_data_long <-
    reshape2::melt(missing_data, id.vars = c("row_number"))
  
  # Plot the missingness heatmap using ggplot2
  ggplot(missing_data_long,
         aes(x = variable, y = row_number, fill = value)) +
    geom_tile() +
    scale_fill_manual(
      values = c("transparent", "grey10"),
      labels = c("Present", "Missing")
    ) +
    labs(x = " ", y = "Row") +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      axis.text.y = element_text(size = 8),
      panel.grid.major.x = element_line(color = "gray80"),
      # Add vertical gridlines
      panel.grid.minor = element_blank() # Remove minor gridlines
    ) +
    scale_y_reverse()  # Reverse the order of y-axis labels
}
# Replace outliers with NA values
replace_outliers_with_na <-
  function(data,
           column_names,
           multiplier = 1.5,
           ignore_lowerBound = FALSE,
           ignore_upperBound = FALSE) {
    for (column_name in column_names) {
      # Calculate the first and third quartiles
      q1 <- quantile(data[[column_name]], 0.25, na.rm = TRUE)
      q3 <- quantile(data[[column_name]], 0.75, na.rm = TRUE)
      iqr <- q3 - q1 # Calculate the IQR
      lower_bound <- q1 - multiplier * iqr
      upper_bound <- q3 + multiplier * iqr
      if (!ignore_lowerBound) {
        # if false, replace outliers with NA
        data[[column_name]][data[[column_name]] < lower_bound] <- NA
      }
      if (!ignore_upperBound) {
        # if false, replace outliers with NA
        data[[column_name]][data[[column_name]] > upper_bound] <- NA
      }
    }
    return(data)
  }

# Function to convert POSIXct to hours after midnight
posix_to_hours_after_midnight <- function(timestamp) {
  ifelse(
    hour(timestamp) >= 15,
    (hour(timestamp) + minute(timestamp) / 60) - 24,
    hour(timestamp) + minute(timestamp) / 60
  )
}