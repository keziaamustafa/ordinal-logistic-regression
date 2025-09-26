# Ordinal Logistic Regression for Credit Score
# Author: Kézia Alves Mustafá
# Date: 2025-09-26
# Description:
#   Clean, reproducible R script to model Credit_Mix (0=Poor, 1=Standard, 2=Good)
#   using ordinal logistic regression (MASS::polr). Includes EDA,
#   proportional odds test, performance metrics, and exportable artifacts.
#
# How to run:
#   1) Place your CSV in the project root. Default name is "ordinal_data.csv".
#   2) Run this script from the project root. Outputs will be saved under ./figs and ./outputs
#   3) You can pass a different CSV path via commandArgs, e.g.:
#        Rscript ordinal_credit_score.R --data "path/to/file.csv"

suppressPackageStartupMessages({
  if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
  pacman::p_load(
    tidyverse, magrittr, ggplot2, MASS, broom, caret, kableExtra, brant
  )
})

set.seed(1234)

options(stringsAsFactors = FALSE, scipen = 999)

dir.create("figs", showWarnings = FALSE)

dir.create("outputs", showWarnings = FALSE)

theme_set(theme_bw(base_size = 12))

# ------------------------------
# Parse CLI args
# ------------------------------
args <- commandArgs(trailingOnly = TRUE)
arg_df <- tibble(arg = args) %>% separate(arg, into = c("key","val"), sep = "=", fill = "right")
get_arg <- function(key, default) {
  v <- arg_df %>% filter(str_detect(key, fixed(key))) %>% pull(val)
  if (length(v) == 0 || is.na(v)) default else v
}

csv_path <- get_arg("--data", "ordinal_data.csv")

# ------------------------------
# Read data
# ------------------------------
stopifnot(file.exists(csv_path))
df <- read.csv(csv_path, header = TRUE, check.names = TRUE)

# Remove ID column

df <- dplyr::select(df, -dplyr::any_of("Customer_ID"))


# Basic sanity checks
stopifnot("Credit_Mix" %in% names(df))

# ------------------------------
# Target as ordered factor
# ------------------------------
# Expected encoding: 0 = Poor, 1 = Standard, 2 = Good
if (is.numeric(df$Credit_Mix)) {
  df$Credit_Mix <- factor(df$Credit_Mix, levels = c(0,1,2), ordered = TRUE)
} else {
  # Try to coerce common labels
  df$Credit_Mix <- df$Credit_Mix %>%
    as.character() %>%
    recode("Poor" = "0", "Standard" = "1", "Good" = "2") %>%
    as.numeric() %>% factor(levels = c(0,1,2), ordered = TRUE)
}
levels(df$Credit_Mix) <- c("Poor","Standard","Good")

# ------------------------------
# Quick EDA tables
# ------------------------------
eda_tbl <- list(
  summary = capture.output(summary(df[setdiff(names(df), "Credit_Mix")])),
  freq    = table(df$Credit_Mix),
  prop    = prop.table(table(df$Credit_Mix)) * 100
)


# Save frequency table
as.data.frame(eda_tbl$freq) %>%
  rename(Score = Var1, Freq = Freq) %>%
  write.csv("outputs/freq_credit_mix.csv", row.names = FALSE)

# ------------------------------
# Correlation matrix for numeric predictors
# ------------------------------
num_df <- df %>% dplyr::select(where(is.numeric))
if (ncol(num_df) >= 2) {
  cor_mat <- cor(num_df, use = "pairwise.complete.obs")
  write.csv(cor_mat, "outputs/correlation_matrix.csv", row.names = TRUE)
}

# ------------------------------
# Visual EDA
# ------------------------------
# Helper for boxplots by class
plot_box <- function(df, y, y_lab, title_lab) {
  ggplot(df, aes(x = Credit_Mix, y = .data[[y]])) +
    geom_boxplot(fill = "#A4A4A4", color = "#241160") +
    labs(x = "Credit Score", y = y_lab, title = title_lab) +
    theme(legend.position = "none")
}

maybe_plot <- function(var, y_lab, title_lab, file_stub) {
  if (var %in% names(df) && is.numeric(df[[var]])) {
    p <- plot_box(df, var, y_lab, title_lab)
    ggsave(filename = file.path("figs", paste0(file_stub, ".png")), p, width = 7, height = 5, dpi = 150)
  }
}

maybe_plot("Monthly_Inhand_Salary", "Monthly salary", "Monthly salary by credit score", "box_salary")
maybe_plot("Num_Bank_Accounts", "Number of bank accounts", "Bank accounts by credit score", "box_bank_accounts")
maybe_plot("Num_Credit_Card", "Number of credit cards", "Credit cards by credit score", "box_credit_cards")
maybe_plot("Credit_Utilization_Ratio", "Credit utilization ratio", "Credit utilization by credit score", "box_utilization")
maybe_plot("Num_of_Delayed_Payment", "Number of delayed payments", "Delayed payments by credit score", "box_delays")
maybe_plot("Changed_Credit_Limit", "Credit limit change (%)", "Credit limit change by credit score", "box_limit_change")

# ------------------------------
# Train-test split (stratified)
# ------------------------------
idx <- caret::createDataPartition(df$Credit_Mix, p = 0.7, list = FALSE)
train_data <- df[idx, ]
test_data  <- df[-idx, ]

# ------------------------------
# Fit baseline polr with all predictors
# ------------------------------
x_vars <- setdiff(names(train_data), "Credit_Mix")
formula_all <- as.formula(paste("Credit_Mix ~", paste(x_vars, collapse = " + ")))

ordinal_base <- MASS::polr(formula_all, data = train_data , Hess = TRUE)

# Stepwise selection by AIC (backward)
step_model <- step(ordinal_base, direction = "backward", trace = 0)

# Model summary and tidy table
summ <- summary(step_model)

coef_tbl <- broom::tidy(step_model, conf.int = TRUE, conf.level = 0.95) %>%
  mutate(
    p.value = 2 * pnorm(abs(statistic), lower.tail = FALSE),
    OR      = exp(estimate),
    OR_low  = exp(conf.low),
    OR_high = exp(conf.high)
  ) %>%
  dplyr::select(term, estimate, std.error, statistic, p.value, OR, OR_low, OR_high)


write.csv(coef_tbl, "outputs/model_coefficients_OR.csv", row.names = FALSE)

# Brant test for proportional odds assumption
brant_ok <- NA
brant_res <- NULL
try({
  brant_res <- brant(step_model$formula, data = step_model$model)
  capture.output(print(brant_res), file = "outputs/brant_test.txt")
  brant_ok <- all(brant_res[["Result"]][,"P.value"] > 0.05)
}, silent = TRUE)

# ------------------------------
# Evaluation on test set
# ------------------------------
# Class prediction
pred_class <- predict(step_model, newdata = test_data, type = "class")
# Probability prediction (for completeness)
pred_prob <- predict(step_model, newdata = test_data, type = "probs") %>% as.data.frame()

cm <- caret::confusionMatrix(pred_class, test_data$Credit_Mix)
acc <- cm$overall["Accuracy"] %>% unname()

# Macro F1 helper
f1_macro <- function(cm_table) {
  lv <- rownames(cm_table)
  f1s <- sapply(lv, function(l) {
    tp <- cm_table[l, l]
    fp <- sum(cm_table[l, ]) - tp
    fn <- sum(cm_table[, l]) - tp
    prec <- ifelse(tp + fp == 0, 0, tp/(tp+fp))
    rec  <- ifelse(tp + fn == 0, 0, tp/(tp+fn))
    if (prec + rec == 0) 0 else 2*prec*rec/(prec+rec)
  })
  mean(f1s)
}

f1 <- f1_macro(cm$table)

# Export metrics and confusion matrix
write.csv(as.data.frame(cm$table), "outputs/confusion_matrix.csv")
write.csv(tibble(metric = c("accuracy", "macro_f1"), value = c(acc, f1)),
          "outputs/metrics.csv", row.names = FALSE)


# ------------------------------
# Session info
# ------------------------------
writeLines(capture.output(sessionInfo()), "outputs/session_info.txt")
