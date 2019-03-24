library(magrittr)
library(tidyverse)
library(ggplot2)
library(keras)

label_fixes <- c("triangles" = "triangle",
                 "trianglee" = "triangle",
                 "circlemtraignle" = "circle_triangle",
                 "circle triangle" = "circle_triangle",
                 "circleline" = "circle_line",
                 "circle text" = "circle_text",
                 "circle_elongated" = "circle",
                 "cricle|cirle" = "circle",
                 "chrevron" = "chevron",
                 "lie" = "line",
                 "texxt" = "text",
                 "star quad" = "star_quad",
                 "exc_idd" = "exclude",
                 "quad?" = "quad",
                 "qaud|qud" = "quad",
                 "stars|start" = "star",
                 "exlude" = "exclude",
                 "rounded" = "",
                 "smooth_texture" = "other",
                 "octagon" = "polygon",
                 "crepe" = "other",
                 "hex" = "polygon",
                 "smooth|hatching" = "other"
)

annotated_imgs <- select(dfunion, image, name) %>%
  mutate(base_image = basename(image)) %>%
  mutate(num_labels = str_count(name, "_") + 1,
         annot_num = 1:n()) %>%
  mutate(labels = str_remove_all(name, "\\(.{1,2}\\)") %>%
           str_replace_all(label_fixes) %>%
           str_split("_")) %>%
  unnest(labels) %>%
  mutate(labels = str_replace_all(labels, label_fixes)) %>%
  mutate(label_type = ifelse(num_labels == 1, "single", "multi")) %>%
  filter(labels != "") %>%
  filter(labels != "exclude") %>%
  filter(labels != "ribbon" & labels != "logo") %>%
  filter(labels %in% c(default_classes, "other"))

fix_annotation <- function(annot) {
  useful_vars <- c("name", "deleted", "date", "id", "username")
  new_vars <- set_names(useful_vars, c("name", "deleted", "date", "objID", "username"))

  get_vars <- new_vars[new_vars %in% names(annot)]

  if (!is.null(annot)) select(annot, !!get_vars) %>% as_tibble() else tibble()
}

# authors <- df %>%
#   filter(!is.null(fullannot)) %>%
#   mutate(base_image = basename(image)) %>%
#   select(id, base_image, image, fullannot) %>%
#   filter(!is.null(date)) %>%
#   mutate(miniannot = map(fullannot, fix_annotation)) %>%
#   select(-fullannot) %>%
#   unnest(miniannot)

new_img_dir <- dirname(newest_data_file)

ann_df <- select(annotated_imgs, base_image, name, num_labels, annot_num, labels, label_type) %>%
  group_by(base_image, name) %>%
  arrange(annot_num) %>%
  mutate(ln = 1:n())

img_df <- tibble(img = list.files(file.path(new_img_dir, c("test", "train", "validation")), full.names = T)) %>%
  mutate(type = str_extract(img, "(test|train|validation)"),
         image = basename(img),
         aug = grepl("^aug", image),
         name = str_extract(image, "^([a-z\\(\\)RE]*?_?){1,}-\\d{1,}-"),
         ln = str_extract(name, "-\\d{1,}-$") %>% gsub(pattern = "-", replacement = "") %>% as.numeric(),
         name = gsub(pattern = "-\\d{1,}-$", "", name),
         name = gsub(pattern = "aug_", "", name),
         name = gsub(pattern = "logo|ribbon|logo\\(R\\)|ribbon\\(R\\)", replacement = "other", name),
         base_image = gsub(image, pattern = "^([a-z\\(\\)RE]*?_?){1,}-\\d{1,}-", replacement = "") %>%
           gsub(., pattern = "_\\d_\\d{1,}.jpg", replacement = ".jpg")) %>%
  select(type, base_image, name, ln, aug, img_path = image) %>%
  mutate(num_labels = str_count(name, "_") + 1,
         annot_num = 1:n()) %>%
  mutate(labels = str_remove_all(name, "\\(.{1,2}\\)") %>%
           str_replace_all(label_fixes) %>%
           str_split("_")) %>%
  unnest(labels) %>%
  mutate(labels = str_replace_all(labels, label_fixes)) %>%
  mutate(label_type = ifelse(num_labels == 1, "single", "multi")) %>%
  filter(labels != "" & labels != "hatching" & labels != "exclude" & !is.na(labels))
