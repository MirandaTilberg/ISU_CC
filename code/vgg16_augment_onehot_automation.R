library(magrittr)
library(lubridate)
library(stringr)
library(jpeg)
library(keras)

use_backend("tensorflow")
# install_keras()
if (!exists("classes")) {
  classes <- c(
    "bowtie", "chevron", "circle", "line", "polygon",
    "quad", "star", "text", "triangle"
  )
}

if (!exists("modeldir")) {
  modeldir <- here::here("model")
}

if (!exists("process_dir")) {
  process_dir <- list.files(file.path(modeldir, "RProcessedImages")) %>%
    as_datetime() %>%
    max(na.rm = T) %>%
    gsub("[^0-9\\ ]", "", .) %>%
    gsub(" ", "-", .)
} else {
  process_dir <- file.path(modeldir, "RProcessedImages", process_dir)
}

if (!exists("aug_multiple")) {
  aug_multiple <- 2
}

if (!exists("epochs")) {
  epochs <- 30
}

process_dir <- gsub("[[:punct:]]models[[:punct:]]RProcessedImages[[:punct:]]{1,}", "\\1", process_dir)
process_dir <- gsub("^[/\\\\]{1,}", "", process_dir)


work_dir <- file.path(modeldir, "TrainedModels")
start_date <- Sys.time() %>% gsub(" ", "_", .)
model_dir <- file.path(work_dir, process_dir)
dir.create(model_dir)

name_file <- function(date, ext) {
  pretrained_base <- "vgg16"
  mod_type <- "onehotaug"
  nclass <- paste0(length(classes), "class")
  pixel_size <- "256"

  filename <- paste(date, pretrained_base, mod_type, nclass,
    pixel_size,
    sep = "_"
  )

  file.path(model_dir, filename) %>%
    paste0(., ext)
}


base_dir <- file.path(modeldir, "RProcessedImages", process_dir)
train_dir <- file.path(base_dir, "train")
train_aug_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")

n_train <- length(list.files(train_dir))
n_validation <- length(list.files(validation_dir))
n_test <- length(list.files(test_dir))

img_names <- list.files(train_dir) %>% str_remove(., "\\.jpg")
img_loc <- list.files(train_dir, full.names = T)

augment_img <- function(filename, times = 3) {
  # Determine number of times augmentation should happen
  rep_num <- str_extract(basename(filename), "^\\d") %>% as.numeric()

  if (!is.na(rep_num)) {
    file_dir <- dirname(filename)
    base_file_name <- file.path(file_dir, str_remove(basename(filename), "^\\d{1,}\\+"))
    file.rename(filename, base_file_name)
    newfilepath <- base_file_name
    newfilename <- basename(base_file_name) %>% str_remove("\\.jpg")
  } else {
    rep_num <- 1
    newfilepath <- filename
    newfilename <- basename(filename) %>% str_remove("\\.jpg")
  }

  img <- readJPEG(newfilepath)
  dim(img) <- c(1, dim(img))

  aug_generator <- image_data_generator(
    samplewise_std_normalization = T,
    rotation_range = 40,
    width_shift_range = 0.05,
    height_shift_range = 0.05,
    shear_range = 60,
    zoom_range = 0.1,
    channel_shift_range = .1,
    zca_whitening = T,
    vertical_flip = T,
    horizontal_flip = TRUE
  )

  images_iter <- flow_images_from_data(
    x = img, y = NULL,
    generator = aug_generator,
    batch_size = 1,
    save_to_dir = train_aug_dir,
    save_prefix = paste("aug", newfilename, sep = "_"),
    save_format = "jpg"
  )

  iter_num <- times
  while (rep_num > 0) {
    while (iter_num > 0) {
      reticulate::iter_next(images_iter)
      iter_num <- iter_num - 1
    }
    rep_num <- rep_num - 1
  }
}

for (i in list.files(train_dir, "*.jpg", full.names = T)) {
  augment_img(i, times = aug_multiple)
}

conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(256, 256, 3)
)

extract_features2 <- function(directory, verbose = F) {

  files <- list.files(directory)
  sample_count <- length(files)

  features <- array(0, dim = c(sample_count, 8, 8, 512))
  labels <- array(0, dim = c(sample_count, length(classes)))

  cat(paste0("Sample count = ", sample_count, ". "))

  for (i in 1:sample_count) {

    if (verbose) {cat(paste0(i, ", "))}
    fname <- files[i]
    str <- substr(fname, 1, regexpr("-", fname) - 1)

    for (j in 1:length(classes)) {
      labels[i, j] <- grepl(classes[j], str)
    }

    img <- readJPEG(file.path(directory, files[i]))
    dim(img) <- c(1, 256, 256, 3)
    features[i, , , ] <- conv_base %>% predict(img)
  }

  list(
    features = features,
    labels = labels
  )
}

train <- extract_features2(train_dir, verbose = T)
validation <- extract_features2(validation_dir, verbose = T)
test <- extract_features2(test_dir, verbose = T)


reshape_features <- function(features) {
  array_reshape(features, dim = c(nrow(features), 8 * 8 * 512))
}

train$features <- reshape_features(train$features)
validation$features <- reshape_features(validation$features)
test$features <- reshape_features(test$features)


class_quantities <- colSums(train$labels)
class_proportions <- (1/class_quantities)/sum(1/class_quantities)
class_weights <- class_proportions %>%
  as.list() %>%
  set_names(as.character(1:length(class_proportions) - 1))
# class_weights <- (sqrt(1/class_proportions) / sum(sqrt(1/class_proportions))) %>%
#   as.list() %>%
#   set_names(as.character(1:length(class_quantities) - 1))

model <- keras_model_sequential() %>%
  layer_dense(
    units = 256, activation = "relu",
    input_shape = 8 * 8 * 512
  ) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = length(classes), activation = "sigmoid")

model %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% fit(
  train$features, train$labels,
  epochs = epochs,
  batch_size = 20,
  validation_data = list(validation$features, validation$labels),
  class_weight = class_weights
)


png(filename = name_file(start_date, ".png"), width = 1000, height = 1000, type = "cairo", pointsize = 16)
plot(history)
dev.off()

save_model_hdf5(model, name_file(start_date, ".h5"))
save_model_weights_hdf5(model, name_file(start_date, "-weights.h5"))

preds <- model %>% predict(test$features)
test_labs <- test$labels
colnames(preds) <- colnames(test_labs) <- classes

save(classes, preds, test_labs, file = name_file(start_date, ".Rdata"))
base::save.image(name_file(start_date, "fullimage.rdata"))

save(history, file = name_file(start_date, "-history.Rdata"))

