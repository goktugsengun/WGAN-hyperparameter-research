# Load the required libraries
library(keras)
library(tensorflow)
library(hms)


# Load the preprocessed data
path_to_file <-"/Users/goktug/Desktop/Neural Networks/data_code_descriptions/LC_EDA.RData"
load(path_to_file)



# Define the main training function    ---MODIFY FOR GRID SEARCH
train <- function(train_data, epochs, batch_size, d_steps, noise_dim, gp_weight, key, save_images = TRUE, plot_images = TRUE) {
  # This function handles the main training loop
  
  generator_total_loss <- 0
  discriminator_total_loss <- 0
  steps <- nrow(train_data)%/%batch_size
  
  old_total_time <- 0
  start_time <- Sys.time()
  
  loss_file <- file(paste0("model_autosaved/", key, "---loss_values.txt"), "w")
  
  for (epoch in 1:epochs) {
    # The main training loop is here
    
    # Save the models after each epoch
    if (epoch >= 2){
      save_model_hdf5(generator, paste0("model_autosaved/generator_autosaved---", key, ".hdf5"))
      save_model_hdf5(discriminator, paste0("model_autosaved/discriminator_autosaved---", key, ".hdf5"))
      cat("Epoch ", epoch, " Generator total loss: ", generator_total_loss$numpy(), " Discriminator total loss: ", discriminator_total_loss$numpy(), "\n", file = loss_file, append = TRUE)
    }
    
    start_epoch_time <- Sys.time()
    
    # Shuffle the training data
    train_data <- train_data[sample(1:nrow(train_data)),]
    train_data <- array_reshape(train_data, c(nrow(train_data), ncol(train_data)))
    
    start_index <- 1
    end_index <- batch_size  
    
    for (step in 1:steps) {
      
      
      
      real_data <- train_data[start_index:end_index,]
      real_data <- array_reshape(real_data, c(batch_size, ncol(train_data)))
      
      for (i in 1:d_steps){
        
        
        noise_shape <- tf$cast(c(batch_size, noise_dim), tf$int32)
        random_latent_vectors <- tf$random$normal(noise_shape) 
        
        with(tf$GradientTape() %as% tape,{
          
          fake_data <- generator(random_latent_vectors, training=TRUE)
          
          fake_logits <- discriminator(fake_data, training=TRUE)
          real_logits <- discriminator(real_data, training=TRUE)
          
          d_cost <- discriminator_loss(real_out=real_logits, fake_out=fake_logits)
          gp <- gradient_penalty(batch_size, real_data, fake_data)
          
          d_loss <- d_cost + gp * gp_weight 
          d_gradient <- tape$gradient(d_loss, discriminator$trainable_variables)
          d_optimizer$apply_gradients(purrr::transpose(
            list(d_gradient, discriminator$trainable_variables)))
        })
      }
      
      #
      
      noise_shape <- tf$cast(c(batch_size, noise_dim), tf$int32)
      random_latent_vectors <- tf$random$normal(noise_shape)
      
      with(tf$GradientTape() %as% tape,
           { 
             
             generated_data <- generator(random_latent_vectors, training=TRUE)
             fake_logits <- discriminator(generated_data, training=TRUE)
             
             g_loss <- generator_loss(fake_logits)
             gen_gradient <- tape$gradient(g_loss, generator$trainable_variables)
             g_optimizer$apply_gradients(
               purrr::transpose(list(gen_gradient, generator$trainable_variables))
             )
           }
      )
      
      generator_total_loss <- generator_total_loss + g_loss 
      discriminator_total_loss <- discriminator_total_loss + d_loss
      
      total_time <- Sys.time() - start_time
      epoch_time <- Sys.time() - start_epoch_time
      total_time_hms <- as_hms(round(as_hms(total_time)))
      epoch_time_hms <- as_hms(round(as_hms(epoch_time)))
      step_time <- total_time - old_total_time
      units(step_time) <-  "secs"
      step_time <- round(step_time, 2)
      
      # Display the current training progress
      cat("Current Grid: ", key, "\n")
      cat("Total train time: ", as.character(total_time_hms), "\n")
      cat("Epoch: ", epoch,"/", epochs, " Time: ", as.character(epoch_time_hms), "\n")
      cat("Step: ",step, "/", steps, " Time: ", as.character(step_time), " secs\n")
      cat("Generator total loss: ", generator_total_loss$numpy(), "\n")
      cat("Discriminator total loss: ", discriminator_total_loss$numpy(), "\n\n") 
      
      old_total_time <- total_time
      start_index <- start_index + batch_size
      end_index <- end_index + batch_size
    }
  }
  
  close(loss_file)
}


# Prepare the training data
train_data <- as.matrix(data[, c(2, 5:6, 11)])  # Select the relevant columns from the data
var_names <- colnames(train_data)  # Save the variable names for later use
MEAN <- apply(train_data, 2, mean)  # Calculate the mean of each column
STD <- apply(train_data, 2, sd)  # Calculate the standard deviation of each column
train_data <- sweep(train_data, 2, MEAN)  # Subtract the mean from each column
train_data <- sweep(train_data, 2, STD, FUN = "/")  # Divide each column by its standard deviation
train_data <- array_reshape(train_data, c(nrow(train_data), ncol(train_data)))  # Reshape the train data


# Define the hyperparameters to search over
learning_rates <- c(0.001, 0.0001)
batch_sizes <- c(32, 64)
epoch_numbers <- c(20, 30)  # Reduced number of epochs
d_step_numbers <- 3   # Reduced number of steps for the discriminator
noise_dims <- 50
gp_weights <- 5




# Perform grid search
for (lr in learning_rates) {
  for (batch_size in batch_sizes) {
    for (epochs in epoch_numbers) {
      
      g_optimizer <- optimizer_adam(learning_rate = lr, beta_1 = 0.5, beta_2 = 0.9)
      d_optimizer <- optimizer_adam(learning_rate = lr, beta_1 = 0.5, beta_2 = 0.9)
      batch_size <- batch_size
      epochs <- epochs
      d_steps <- d_steps
      noise_dim <- noise_dim
      gp_weight <- gp_weight
      
      # Create a unique key for the current hyperparameter combination
      key <- paste0("lr_", lr, "_batch_", batch_size, "_epochs_", epochs, "_dsteps_", d_steps, "_noise_", noise_dim, "_gp_", gp_weight)
      
      # Define the generator model
      generator_input <- layer_input(noise_dim)
      generator_output <- generator_input %>% 
        layer_dense(units = 128) %>% 
        layer_activation_leaky_relu(alpha = 0.2) %>% 
        layer_dropout(0.3) %>%
        layer_dense(units = 256) %>% 
        layer_activation_leaky_relu(alpha = 0.2) %>% 
        layer_dropout(0.3) %>%
        layer_dense(units = 512) %>% 
        layer_activation_leaky_relu(alpha = 0.2) %>% 
        layer_dropout(0.3) %>%
        layer_dense(units = ncol(train_data))
      
      generator <- keras_model(generator_input, generator_output)
      
      # Define the discriminator model
      discriminator_input <- layer_input(ncol(train_data))
      discriminator_output <- discriminator_input %>% 
        layer_dense(units = 512) %>% 
        layer_activation_leaky_relu(alpha = 0.2) %>% 
        layer_dense(units = 256) %>% 
        layer_activation_leaky_relu(alpha = 0.2) %>% 
        layer_dense(units = 128) %>% 
        layer_activation_leaky_relu(alpha = 0.2) %>% 
        layer_dense(units = 1, activation = "tanh")
      
      discriminator <- keras_model(discriminator_input, discriminator_output)
      
      d_optimizer <- optimizer_adam(learning_rate = lr, beta_1 = 0.5, beta_2 = 0.9)
      g_optimizer <- optimizer_adam(learning_rate = lr, beta_1 = 0.5, beta_2 = 0.9)
      
      # Define the loss functions 
      discriminator_loss <- function(real_out, fake_out){ # QUESTION: What are real_out and fake_out?
        # QUESTION: What type of loss do we have here? 
        real_loss <- tf$reduce_mean(real_out)
        fake_loss <- tf$reduce_mean(fake_out)
        return(fake_loss - real_loss) #
      }
      
      generator_loss <- function(fake_out){ 
        return(-tf$reduce_mean(fake_out))
      }
      
      gradient_penalty <- function(batch_size, real_data, fake_data){
        
        alpha_shape <- tf$cast(c(batch_size, ncol(train_data)), tf$int32)
        alpha <- tf$random$normal(alpha_shape)
        
        diff <- fake_data - real_data
        interpolated <- real_data + alpha * diff 
        
        with(tf$GradientTape() %as% gp_tape, { 
          gp_tape$watch(interpolated)
          pred <- discriminator(interpolated, training = TRUE)
          
          grads <- gp_tape$gradient(pred, interpolated) 
          normL2 <- tf$sqrt(tf$reduce_sum(tf$square(grads)))
        })
        gp <- tf$reduce_mean((normL2 - 1) ^ 2) 
        return(gp)
      }
      
      # Perform training
      train(train_data, epochs, batch_size, d_steps, noise_dim, gp_weight, key, save_images = TRUE, plot_images = TRUE)
      
    }
  }
}



noise_dim <- 50

# Define the function to generate new data after training
generate_data <- function(n= 100){
  # This function generates new samples from the trained generator
  
  noise_shape <- tf$cast(c(n, noise_dim), tf$int32)
  random_latent_vectors <- tf$random$normal(noise_shape)
  fake_dataset <- as.matrix(generator(random_latent_vectors, training=FALSE))
  
  # Rescale the generated data to the original scale
  fake_dataset <- sweep(fake_dataset,2, STD, FUN="*")
  fake_dataset <- sweep(fake_dataset,2, MEAN, FUN="+" )
  colnames(fake_dataset) <- var_names
  return(fake_dataset)
}


# Define the function to load the saved models
load_models <- function(generatorpath){
  generator <<- load_model_hdf5(generatorpath)
  discriminator <<- load_model_hdf5("/Users/goktug/Desktop/models2/discriminator_autosaved---lr_0.001_batch_32_epochs_30_dsteps_3_noise_50_gp_5.hdf5")
}

generator_path_1 <- "/Users/goktug/Desktop/models2/generator_autosaved---lr_0.001_batch_32_epochs_20_dsteps_3_noise_50_gp_5.hdf5"
generator_path_2 <- "/Users/goktug/Desktop/models2/generator_autosaved---lr_0.001_batch_32_epochs_30_dsteps_3_noise_50_gp_5.hdf5"
generator_path_3 <- "/Users/goktug/Desktop/models2/generator_autosaved---lr_0.001_batch_64_epochs_20_dsteps_3_noise_50_gp_5.hdf5"
generator_path_4 <- "/Users/goktug/Desktop/models2/generator_autosaved---lr_0.001_batch_64_epochs_30_dsteps_3_noise_50_gp_5.hdf5"
generator_path_5 <- "/Users/goktug/Desktop/models2/generator_autosaved---lr_1e-04_batch_32_epochs_20_dsteps_3_noise_50_gp_5.hdf5"
generator_path_6 <- "/Users/goktug/Desktop/models2/generator_autosaved---lr_1e-04_batch_32_epochs_30_dsteps_3_noise_50_gp_5.hdf5" ### best one
generator_path_7 <- "/Users/goktug/Desktop/models2/generator_autosaved---lr_1e-04_batch_64_epochs_20_dsteps_3_noise_50_gp_5.hdf5"
generator_path_8 <- "/Users/goktug/Desktop/models2/generator_autosaved---lr_1e-04_batch_64_epochs_30_dsteps_3_noise_50_gp_5.hdf5"
generator_path_9 <- "/Users/goktug/Downloads/generator.hdf5"
generator_path_10 <- "/Users/goktug/Desktop/models2/generator_autosaved---lr_0.001_batch_128_epochs_100_dsteps_3_noise_50_gp_5.hdf5"
generator_path_11 <- "/Users/goktug/Desktop/models2/generator_autosaved---lr_0.001_batch_128_epochs_100_dsteps_3_noise_50_gp_15.hdf5"


# Data Generation
noise_dim <- 50
# model1
load_models(generatorpath = generator_path_1)
model1 <- generate_data()   #-- no value
model1

# model2
load_models(generatorpath = generator_path_2)
model2 <- generate_data()   #good
model2
# model3
load_models(generatorpath = generator_path_3)
model3 <- generate_data()   #no value
model3
# model4
load_models(generatorpath = generator_path_4)
model4 <- generate_data()   #good
model4
# model5
load_models(generatorpath = generator_path_5)
model5 <- generate_data()   #good
model5

# model6
load_models(generatorpath = generator_path_6)
model6 <- generate_data()   #good
model6

# model7
load_models(generatorpath = generator_path_7)
model7 <- generate_data()   #good
model7

#model8
load_models(generatorpath = generator_path_8)
model8 <- generate_data()   #good
model8

noise_dim <- 50
#model9. ## find it from the computer
load_models(generatorpath = generator_path_9)
model9 <- generate_data()
model9

#model10
load_models(generatorpath = generator_path_10)
model10 <- generate_data()
model10

#model11
load_models(generatorpath = generator_path_11)
model11 <- generate_data()

# Stylized Facts
rm(list = ls(all.names = TRUE))

# Load the required libraries
library(keras)
library(tensorflow)
library(hms)
library(e1071)
library(corrplot)
library(ggplot2)
library(tidyr)

# Load the preprocessed data
load("/Users/goktug/Desktop/Neural Networks/data_code_descriptions/LC_EDA.RData")

# Prepare the training data
train_data <- as.matrix(data[,c(2, 5:6, 11)]) # Select the relevant columns from the data
var_names <- colnames(train_data) # Save the variable names for later use
colnames(train_data) <- var_names# name train data columns

# summary stats
sum <- summary(train_data)
mean <- apply(train_data, 2, mean)
skew <- apply(train_data, 2, skewness)
kurt <- apply(train_data, 2, kurtosis)
std <- apply(train_data, 2, sd)
table <- rbind(sum, std, skew, kurt)
table

round(data.frame(Mean = mean, 
                 Standard_Deviation = std,
                 Skewness = skew,
                 Kurtosis = kurt), 2)



# correlation matrix + heat map
correlation_matrix <- cor(train_data)
rounded_cor <- round(correlation_matrix, 2)
my_color_palette <- colorRampPalette(c("yellow", "orange", "red"))(100)
corrplot(correlation_matrix, method = "color", type = "upper", diag = FALSE,
         tl.col = "black", addCoef.col = "white", col = my_color_palette)

# prepare data for plotting
train_data_df <- as.data.frame(train_data)
loan_data_long <- gather(train_data_df, key = "Variable", value = "Value")


# Visualization


# Plot distribution
ggplot(loan_data_long, aes(x = Value)) +
  geom_density(aes(y = ..count..), color  = "blue") +
  facet_wrap(~ Variable, scales = "free", ncol = 2) +
  labs(x = "Value", y = "Frequency") +
  theme_classic() + 
  theme(strip.background = element_blank())+
  scale_x_continuous(labels = scales::comma)+
  scale_y_continuous(labels = scales::comma)


# Define the function to generate new data after training
generate_data <- function(n=100){
  # This function generates new samples from the trained generator
  
  noise_shape <- tf$cast(c(n, noise_dim), tf$int32)
  random_latent_vectors <- tf$random$normal(noise_shape)
  fake_dataset <- as.matrix(generator(random_latent_vectors, training=FALSE))
  
  # Rescale the generated data to the original scale
  fake_dataset <- sweep(fake_dataset,2, STD, FUN="*")
  fake_dataset <- sweep(fake_dataset,2, MEAN, FUN="+" )
  colnames(fake_dataset) <- var_names
  return(fake_dataset)
}


# Statistical Analysis

train_data <- as.matrix(data[,c(2, 5:6, 11)]) 
generated_data <- as.matrix(generate_data(n = nrow(train_data)))

generated_data <- as.matrix(model2)

train_data <- as.data.frame(train_data)
generated_data <- as.data.frame(generated_data)

summary(train_data) # Summary statistics for train_data
summary(generated_data) # Summary statistics for generated_data

# Distribution Analysis
library(ggplot2)


# Plotting linegraph for each column
library(ggplot2)
library(gridExtra)
library(scales)

look <- function(){
  loan_amnt <- ggplot() +
    geom_density(data = train_data, aes(x = loan_amnt, color = "Train Data"), alpha = 0.5, show.legend = FALSE) +
    geom_density(data = generated_data, aes(x = loan_amnt, color = "Generated Data"),  alpha = 0.5, show.legend = FALSE) +
    labs(x = "Loan Amount",
         y = "Density") +
    theme_minimal() +
    scale_x_continuous(labels = label_number_si())
  
  int_rate <- ggplot() +
    geom_density(data = train_data, aes(x = int_rate, color = "Train Data"), alpha = 0.5, , show.legend = FALSE) +
    geom_density(data = generated_data, aes(x = int_rate, color = "Generated Data"),  alpha = 0.5, , show.legend = FALSE) +
    labs(x = "Interest Rate",
         y = "Density", color = "") +
    theme_minimal()  +
    theme(axis.title.y = element_blank()) +
    scale_x_continuous(labels = label_number_si())
  
  installment <- ggplot() +
    geom_density(data = train_data, aes(x = installment, color = "Train Data"), alpha = 0.5, show.legend = FALSE) +
    geom_density(data = generated_data, aes(x = installment, color = "Generated Data"),  alpha = 0.5, show.legend = FALSE) +
    labs(x = "Installment",
         y = "Density")+
    theme_minimal() +
    scale_x_continuous(labels = label_number_si())
  
  annual_inc <- ggplot() +
    geom_density(data = train_data, aes(x = annual_inc, color = "Train Data"), alpha = 0.5, show.legend = FALSE) +
    geom_density(data = generated_data, aes(x = annual_inc, color = "Generated Data"), alpha = 0.5, show.legend = FALSE) +
    labs(x = "Annual Income",
         y = "Density", color = "") +
    scale_x_continuous(limits = c(0, 250000), labels = label_number_si()) +
    theme_minimal() +
    theme(axis.title.y = element_blank())
  
  grid.arrange(loan_amnt, int_rate, installment, annual_inc, nrow = 2, ncol = 2)
  
  grid_plots <- grid.arrange(loan_amnt, int_rate, installment, annual_inc, nrow = 2)
  
  return(grid_plots)
}

grid.arrange(loan_amnt, int_rate, installment, annual_inc, nrow = 2, ncol = 2)

grid_plots <- grid.arrange(loan_amnt, int_rate, installment, annual_inc, nrow = 2)




model <- model9
generated_data <- as.matrix(model)
train_data <- as.data.frame(train_data)
generated_data <- as.data.frame(generated_data)
look()

